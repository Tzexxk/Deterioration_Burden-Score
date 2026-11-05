"""Fit and load the full-development Extra Trees checkpoint."""

from importlib.metadata import version

import joblib
from sklearn.ensemble import ExtraTreesRegressor

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.train.ehr.sklearn_training import checked_prediction, load_estimator


def fit(features, labels, schema, settings, rounds, folder):
    require(version("scikit-learn") == "1.9.1" and version("joblib") == "1.5.3",
            "INVALID_MODEL_RUNTIME", "Use scikit-learn 1.9.1 and joblib 1.5.3")
    state = fit_preprocessing(features, schema)
    values = transform_features(features, state)
    model = ExtraTreesRegressor(**settings["candidate"]["params"]).fit(values, labels)
    joblib.dump(model, folder / "model.joblib", compress=3)
    write_json(folder / "preprocessing.json", state)
    return checked_prediction(model, values)


def predict(features, schema, rounds, folder):
    require(version("scikit-learn") == "1.9.1" and version("joblib") == "1.5.3",
            "INVALID_MODEL_RUNTIME", "Use scikit-learn 1.9.1 and joblib 1.5.3")
    model = load_estimator(folder / "model.joblib", ExtraTreesRegressor)
    values = transform_features(features, read_json(folder / "preprocessing.json"))
    require(model.n_features_in_ == values.shape[1], "INVALID_CHECKPOINT", "Extra Trees refit input width differs")
    return checked_prediction(model, values)
