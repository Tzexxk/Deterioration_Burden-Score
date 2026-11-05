"""Fit and load the full-development XGBoost checkpoint."""

from importlib.metadata import version

import numpy as np
from xgboost import XGBRegressor

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.train.ehr.xgboost.training import predict as predict_rounds


def fit(features, labels, schema, settings, rounds, folder):
    require(version("xgboost") == "3.2.0" and version("scikit-learn") == "1.9.1",
            "INVALID_MODEL_RUNTIME", "Use XGBoost 3.2.0 and scikit-learn 1.9.1")
    state = fit_preprocessing(features, schema)
    values = transform_features(features, state)
    params = {**settings["candidate"]["params"], "missing": np.nan, "n_estimators": rounds,
              "early_stopping_rounds": None}
    model = XGBRegressor(**params).fit(values, labels, verbose=False)
    model.save_model(folder / "model.ubj")
    write_json(folder / "preprocessing.json", state)
    return predict_rounds(model, values, rounds)


def predict(features, schema, rounds, folder):
    require(version("xgboost") == "3.2.0" and version("scikit-learn") == "1.9.1",
            "INVALID_MODEL_RUNTIME", "Use XGBoost 3.2.0 and scikit-learn 1.9.1")
    model = XGBRegressor(n_jobs=1)
    model.load_model(folder / "model.ubj")
    values = transform_features(features, read_json(folder / "preprocessing.json"))
    require(model.n_features_in_ == values.shape[1] and model.get_booster().num_boosted_rounds() == rounds,
            "INVALID_CHECKPOINT", "XGBoost refit width or round count differs")
    return predict_rounds(model, values, rounds)
