"""Portable checkpoint interface shared by EHR linear estimators."""

import warnings
from importlib.metadata import version

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_preprocessing import fit_preprocessing, transform_features


def predict_checkpoint(checkpoint, features):
    values = transform_features(features, checkpoint["preprocessing"])
    coefficients = np.asarray(checkpoint["coefficients"], dtype=float)
    intercept = checkpoint["intercept"]
    require(coefficients.shape == (values.shape[1],) and np.isfinite(coefficients).all() and np.isfinite(intercept),
            "INVALID_CHECKPOINT", "Invalid linear coefficients or intercept")
    link = checkpoint.get("link", "identity")
    require(link in ("identity", "log"), "INVALID_CHECKPOINT", "Unsupported checkpoint link")
    with threadpool_limits(limits=1):
        prediction = values @ coefficients + intercept
    with np.errstate(over="ignore"):
        if link == "log":
            prediction = np.exp(prediction)
    require(np.isfinite(prediction).all(), "INVALID_CHECKPOINT", "Checkpoint predictions must be finite")
    return prediction


def fit_linear_fold(package, experiment, fold, folder, estimator):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y = package.samples.iloc[train].burden_score.to_numpy(float)
    settings = experiment["parameters"]["candidate"]["params"]
    model = estimator(**settings)
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(transform_features(x_train, state), y)
        require(not any(issubclass(item.category, ConvergenceWarning) for item in caught),
                "OPTIMIZER_DID_NOT_CONVERGE", "The configured estimator did not converge on this fold")
        expected = model.predict(transform_features(x_validation, state))
    folder.mkdir()
    path = folder / "model.json"
    checkpoint = {"preprocessing": state, "coefficients": model.coef_.tolist(), "intercept": float(model.intercept_)}
    if "link" in settings:
        checkpoint["link"] = settings["link"]
    write_json(path, checkpoint)
    np.testing.assert_allclose(predict_checkpoint(read_json(path), x_validation), expected, rtol=0, atol=1e-12)
    write_json(folder / "interface.json", {"fold": fold["fold"], "parameters": settings,
               "features": list(package.features.columns), "scikit_learn_version": version("scikit-learn"),
               "train_sample_ids": package.samples.iloc[train].sample_id.tolist(),
               "validation_sample_ids": package.samples.iloc[validation].sample_id.tolist()})
    return {"fold": fold["fold"], "fit_method": "single_fit"}


def checkpoint_prediction(store, manifest, package, fold):
    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    interface = read_json(output("interface.json"))
    train, validation = fold["train_indices"], fold["validation_indices"]
    require(interface["fold"] == fold["fold"] and interface["scikit_learn_version"] == version("scikit-learn")
            and interface["features"] == list(package.features.columns)
            and interface["train_sample_ids"] == package.samples.iloc[train].sample_id.tolist()
            and interface["validation_sample_ids"] == package.samples.iloc[validation].sample_id.tolist(),
            "INVALID_CHECKPOINT", "Linear checkpoint feature, version or fold identity differs")
    checkpoint = read_json(output("model.json"))
    require(checkpoint.get("link", "identity") == interface["parameters"].get("link", "identity"),
            "INVALID_CHECKPOINT", "Checkpoint link differs from its training interface")
    return predict_checkpoint(checkpoint, package.features.iloc[validation]), None
