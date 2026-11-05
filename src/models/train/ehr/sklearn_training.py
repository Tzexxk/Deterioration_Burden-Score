"""Shared training and checkpoint reload for EHR scikit-learn estimators."""

import warnings
from functools import partial
from importlib.metadata import version

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_fold import fold_interface, read_fold_interface
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.experiments import read_batch
from src.models.train.ehr.training import candidate_settings, train_model


def checked_prediction(model, features):
    with threadpool_limits(limits=1):
        scores = np.asarray(model.predict(features), dtype=float)
    require(scores.shape == (len(features),) and np.isfinite(scores).all(),
            "INVALID_CHECKPOINT", "Model predictions must be finite and match the fold")
    return scores


def load_estimator(path, estimator):
    # Only read locally generated training outputs from the validated manifest.
    model = joblib.load(path)
    require(type(model) is estimator, "INVALID_CHECKPOINT", "Unexpected checkpoint estimator")
    return model


def fit_fold(package, experiment, fold, folder, *, estimator, family):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y = package.samples.iloc[train].burden_score.to_numpy(float)
    settings = candidate_settings(experiment, family)
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        model = estimator(**settings)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(transform_features(x_train, state), y)
        require(not any(issubclass(item.category, ConvergenceWarning) for item in caught),
                "OPTIMIZER_DID_NOT_CONVERGE", "The configured estimator did not converge on this fold")
    values = transform_features(x_validation, state)
    expected = checked_prediction(model, values)
    folder.mkdir()
    path = folder / "model.joblib"
    joblib.dump(model, path, compress=3)
    np.testing.assert_allclose(checked_prediction(load_estimator(path, estimator), values), expected, rtol=0, atol=1e-12)
    write_json(folder / "preprocessing.json", state)
    write_json(folder / "interface.json", {**fold_interface(package, fold, model.get_params(deep=False)), "joblib_version": version("joblib")})
    return {"fold": fold["fold"], "fit_method": "single_fit"}


def checkpoint_prediction(store, manifest, package, fold, *, estimator):
    interface = read_fold_interface(store, manifest, package, fold)
    require(interface["joblib_version"] == version("joblib"), "INVALID_CHECKPOINT", "Joblib version differs")

    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    state = read_json(output("preprocessing.json"))
    values = transform_features(package.features.iloc[fold["validation_indices"]], state)
    model = load_estimator(output("model.joblib"), estimator)
    require(model.get_params(deep=False) == interface["parameters"] and model.n_features_in_ == values.shape[1],
            "INVALID_CHECKPOINT", "Model parameters or input width differ from the training interface")
    return checked_prediction(model, values), None


def run_training(layout, config_path, selected=None, *, model, family, estimator):
    experiments = read_batch(layout, config_path, selected)
    keys = estimator().get_params(deep=False).keys()
    for experiment in experiments:
        settings = candidate_settings(experiment, family)
        require(set(settings) <= keys, "INVALID_MODEL_CONFIG", "Unknown estimator parameter")
    require(version("scikit-learn") == "1.9.1" and version("joblib") == "1.5.3",
            "INVALID_MODEL_RUNTIME", "Use scikit-learn 1.9.1 and joblib 1.5.3")
    fit = partial(fit_fold, estimator=estimator, family=family)
    return train_model(layout, config_path, experiments, model, fit)
