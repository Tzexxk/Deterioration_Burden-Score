"""Fit Tweedie candidates on the five development training folds."""

from importlib.metadata import version

import numpy as np
from sklearn.linear_model import TweedieRegressor

from src.common.paths import require
from src.models.ehr_linear import fit_linear_fold
from src.models.experiments import read_batch
from src.models.train.ehr.training import train_model

FIXED_PARAMETERS = {"fit_intercept": True, "link": "log", "solver": "lbfgs", "verbose": 0, "warm_start": False}


def validate_experiment(experiment):
    params = experiment["parameters"]
    require(set(params) == {"candidate"} and isinstance(params["candidate"], dict),
            "INVALID_TWEEDIE_CONFIG", "A Tweedie candidate object is required")
    candidate = params["candidate"]
    require(set(candidate) == {"name", "family", "params"} and candidate["family"] == "tweedie"
            and isinstance(candidate["params"], dict), "INVALID_TWEEDIE_CONFIG", "Invalid Tweedie candidate")
    settings = candidate["params"]
    require(set(settings) == FIXED_PARAMETERS.keys() | {"alpha", "power", "max_iter", "tol"}
            and all(settings[name] == value for name, value in FIXED_PARAMETERS.items())
            and all(type(settings[name]) in (int, float) and np.isfinite(settings[name]) for name in ("alpha", "power", "tol"))
            and settings["alpha"] >= 0 and 1 < settings["power"] < 2 and settings["tol"] > 0
            and type(settings["max_iter"]) is int and settings["max_iter"] > 0,
            "INVALID_TWEEDIE_CONFIG", "Use a log link and 1 < power < 2 for scores that can be zero")


def fit_fold(package, experiment, fold, folder):
    return fit_linear_fold(package, experiment, fold, folder, TweedieRegressor)


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("scikit-learn") == "1.9.1", "INVALID_SKLEARN_VERSION", "Use scikit-learn 1.9.1 for this protocol")
    return train_model(layout, config_path, experiments, "tweedie", fit_fold)
