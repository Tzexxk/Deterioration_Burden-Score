"""Fit each Ridge candidate on its own five development training folds."""

from importlib.metadata import version

import numpy as np
from sklearn.linear_model import Ridge

from src.common.paths import require
from src.models.ehr_linear import fit_linear_fold
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


def fit_fold(package, experiment, fold, folder):
    return fit_linear_fold(package, experiment, fold, folder, Ridge)


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("scikit-learn") == "1.9.1", "INVALID_SKLEARN_VERSION", "Use scikit-learn 1.9.1 for this protocol")
    return train_model(layout, config_path, experiments, "ridge", fit_fold)
