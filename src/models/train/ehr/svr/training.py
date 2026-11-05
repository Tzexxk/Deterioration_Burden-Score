"""Fit RBF support vector regression candidates on the development folds."""

import warnings
from importlib.metadata import version

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_fold import fold_interface, read_fold_interface
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.experiments import read_batch
from src.models.train.ehr.training import train_model

FIXED_PARAMETERS = {
    "cache_size": 200, "coef0": 0.0, "degree": 3, "epsilon": 0.1, "gamma": 1.0,
    "kernel": "rbf", "max_iter": 100000, "shrinking": True, "tol": 0.0001, "verbose": False,
}


def validate_experiment(experiment):
    params = experiment["parameters"]
    require(set(params) == {"candidate"} and isinstance(params["candidate"], dict),
            "INVALID_SVR_CONFIG", "An SVR candidate object is required")
    candidate = params["candidate"]
    require(set(candidate) == {"name", "family", "params"} and candidate["family"] == "svr"
            and isinstance(candidate["params"], dict), "INVALID_SVR_CONFIG", "Invalid SVR candidate")
    settings = candidate["params"]
    require(set(settings) == FIXED_PARAMETERS.keys() | {"C"}
            and all(settings[name] == value for name, value in FIXED_PARAMETERS.items())
            and type(settings["C"]) in (int, float) and np.isfinite(settings["C"]) and settings["C"] > 0,
            "INVALID_SVR_CONFIG", "Use the RBF SVR protocol with positive finite C")


def predict_checkpoint(checkpoint, training_features, validation_features):
    validation = transform_features(validation_features, checkpoint["preprocessing"])
    support = checkpoint["support_indices"]
    coefficients = np.asarray(checkpoint["dual_coefficients"], dtype=float)
    gamma, intercept = checkpoint["gamma"], checkpoint["intercept"]
    require(isinstance(support, list)
            and all(type(i) is int and 0 <= i < len(training_features) for i in support)
            and len(set(support)) == len(support) and coefficients.shape == (len(support),)
            and np.isfinite(coefficients).all() and gamma == 1 / validation.shape[1] and np.isfinite(intercept),
            "INVALID_CHECKPOINT", "Invalid SVR support indices, coefficients, gamma or intercept")
    predicted = np.full(len(validation), intercept, dtype=float)
    if support:
        vectors = transform_features(training_features.iloc[support], checkpoint["preprocessing"])
        with threadpool_limits(limits=1):
            predicted += rbf_kernel(validation, vectors, gamma=gamma) @ coefficients
    require(np.isfinite(predicted).all(), "INVALID_CHECKPOINT", "Checkpoint predictions must be finite")
    return predicted


def fit_fold(package, experiment, fold, folder):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y = package.samples.iloc[train].burden_score.to_numpy(float)
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        values = transform_features(x_train, state)
        settings = {**experiment["parameters"]["candidate"]["params"], "gamma": 1 / values.shape[1]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = SVR(**settings).fit(values, y)
        require(model.fit_status_ == 0 and not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                "OPTIMIZER_DID_NOT_CONVERGE", "The configured SVR did not converge on this fold")
        expected = model.predict(transform_features(x_validation, state))
    folder.mkdir()
    path = folder / "model.json"
    write_json(path, {"preprocessing": state, "support_indices": model.support_.tolist(),
                      "dual_coefficients": model.dual_coef_[0].tolist(), "gamma": settings["gamma"],
                      "intercept": float(model.intercept_[0])})
    np.testing.assert_allclose(predict_checkpoint(read_json(path), x_train, x_validation), expected, rtol=0, atol=1e-12)
    write_json(folder / "interface.json", fold_interface(package, fold, settings))
    return {"fold": fold["fold"], "fit_method": "single_fit"}


def checkpoint_prediction(store, manifest, package, fold):
    interface = read_fold_interface(store, manifest, package, fold)
    ref = store.output_ref(manifest, f"fold_{fold['fold']}/model.json")
    checkpoint = read_json(store.layout.resolve(ref))
    require(checkpoint["gamma"] == interface["parameters"]["gamma"],
            "INVALID_CHECKPOINT", "SVR gamma differs from its training interface")
    train, validation = fold["train_indices"], fold["validation_indices"]
    return predict_checkpoint(checkpoint, package.features.iloc[train], package.features.iloc[validation]), None


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("scikit-learn") == "1.9.1", "INVALID_SKLEARN_VERSION", "Use scikit-learn 1.9.1 for this protocol")
    return train_model(layout, config_path, experiments, "svr", fit_fold)
