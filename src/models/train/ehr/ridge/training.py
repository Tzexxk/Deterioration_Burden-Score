"""Fit each Ridge candidate on its own five development training folds."""

from importlib.metadata import version

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.experiments import read_batch
from src.models.train.ehr.training import train_model

FIXED_PARAMETERS = {"copy_X": True, "fit_intercept": True, "max_iter": None, "positive": False,
                    "random_state": None, "solver": "svd", "tol": 0.0001}


def validate_experiment(experiment):
    params = experiment["parameters"]
    require(set(params) == {"candidate"} and isinstance(params["candidate"], dict),
            "INVALID_RIDGE_CONFIG", "A Ridge candidate object is required")
    candidate = params["candidate"]
    require(set(candidate) == {"name", "family", "params"} and candidate["family"] == "ridge"
            and isinstance(candidate["params"], dict), "INVALID_RIDGE_CONFIG", "Invalid Ridge candidate")
    settings = candidate["params"]
    require(set(settings) == FIXED_PARAMETERS.keys() | {"alpha"}
            and all(settings[name] == value for name, value in FIXED_PARAMETERS.items())
            and type(settings["alpha"]) in (int, float) and np.isfinite(settings["alpha"]) and settings["alpha"] >= 0,
            "INVALID_RIDGE_CONFIG", "Use the Ridge SVD protocol with nonnegative finite alpha")


def predict_checkpoint(checkpoint, features):
    values = transform_features(features, checkpoint["preprocessing"])
    coefficients = np.asarray(checkpoint["coefficients"], dtype=float)
    intercept = checkpoint["intercept"]
    require(coefficients.shape == (values.shape[1],) and np.isfinite(coefficients).all() and np.isfinite(intercept),
            "INVALID_CHECKPOINT", "Invalid Ridge coefficients or intercept")
    with threadpool_limits(limits=1):
        return values @ coefficients + intercept


def fit_fold(package, experiment, fold, folder):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y = package.samples.iloc[train].burden_score.to_numpy(float)
    settings = experiment["parameters"]["candidate"]["params"]
    model = Ridge(**settings)
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        model.fit(transform_features(x_train, state), y)
        expected = model.predict(transform_features(x_validation, state))
    folder.mkdir()
    path = folder / "model.json"
    write_json(path, {"preprocessing": state, "coefficients": model.coef_.tolist(), "intercept": float(model.intercept_)})
    np.testing.assert_allclose(predict_checkpoint(read_json(path), x_validation), expected, rtol=0, atol=1e-12)
    write_json(folder / "interface.json", {"fold": fold["fold"], "parameters": settings,
               "features": list(package.features.columns), "scikit_learn_version": version("scikit-learn"),
               "train_sample_ids": package.samples.iloc[train].sample_id.tolist(),
               "validation_sample_ids": package.samples.iloc[validation].sample_id.tolist()})
    return {"fold": fold["fold"], "fit_method": "single_fit"}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("scikit-learn") == "1.9.1", "INVALID_SKLEARN_VERSION", "Use scikit-learn 1.9.1 for this protocol")
    return train_model(layout, config_path, experiments, "ridge", fit_fold)
