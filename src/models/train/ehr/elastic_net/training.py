"""Fit Elastic Net candidates on the five development training folds."""

from importlib.metadata import version

import numpy as np
from sklearn.linear_model import ElasticNet

from src.common.paths import require
from src.models.ehr_linear import fit_linear_fold
from src.models.experiments import read_batch
from src.models.train.ehr.training import train_model

FIXED_PARAMETERS = {"copy_X": True, "fit_intercept": True, "positive": False, "precompute": False,
                    "random_state": None, "selection": "cyclic", "warm_start": False}


def validate_experiment(experiment):
    params = experiment["parameters"]
    require(set(params) == {"candidate"} and isinstance(params["candidate"], dict),
            "INVALID_ELASTIC_NET_CONFIG", "An Elastic Net candidate object is required")
    candidate = params["candidate"]
    require(set(candidate) == {"name", "family", "params"} and candidate["family"] == "elastic"
            and isinstance(candidate["params"], dict), "INVALID_ELASTIC_NET_CONFIG", "Invalid Elastic Net candidate")
    settings = candidate["params"]
    require(set(settings) == FIXED_PARAMETERS.keys() | {"alpha", "l1_ratio", "max_iter", "tol"}
            and all(settings[name] == value for name, value in FIXED_PARAMETERS.items())
            and all(type(settings[name]) in (int, float) and np.isfinite(settings[name]) and settings[name] > 0
                    for name in ("alpha", "l1_ratio", "tol"))
            and settings["l1_ratio"] <= 1 and type(settings["max_iter"]) is int and settings["max_iter"] > 0,
            "INVALID_ELASTIC_NET_CONFIG", "Use cyclic coordinate descent with valid regularization and convergence limits")


def fit_fold(package, experiment, fold, folder):
    return fit_linear_fold(package, experiment, fold, folder, ElasticNet)


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("scikit-learn") == "1.9.1", "INVALID_SKLEARN_VERSION", "Use scikit-learn 1.9.1 for this protocol")
    return train_model(layout, config_path, experiments, "elastic_net", fit_fold)
