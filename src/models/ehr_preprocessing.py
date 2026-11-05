"""Fold-fitted preprocessing for the historical EHR cross-model protocol."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.paths import require


def fit_preprocessing(features, schema):
    numeric, categorical = schema["numeric_features"], schema["categorical_features"]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    filled = imputer.fit_transform(features[numeric])
    scaler = StandardScaler().fit(filled)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64).fit(features[categorical])
    return {"numeric": numeric, "categorical": categorical, "imputation": imputer.statistics_.tolist(),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "categories": {name: values.tolist() for name, values in zip(categorical, encoder.categories_, strict=True)}}


def transform_features(features, state):
    numeric, categorical = state["numeric"], state["categorical"]
    require(list(features.columns) == numeric + categorical, "INVALID_FEATURE_ORDER", "EHR feature order changed")
    values = features[numeric].to_numpy(copy=True)
    missing = np.isnan(values)
    np.copyto(values, np.asarray(state["imputation"], dtype=values.dtype), where=missing)
    # Match StandardScaler's float32 rounding before both operations.
    values -= np.asarray(state["mean"], dtype=values.dtype)
    values /= np.asarray(state["scale"], dtype=values.dtype)
    encoded = [features[name].to_numpy()[:, None] == np.asarray(state["categories"][name])[None, :]
               for name in categorical]
    result = np.concatenate([values, missing.astype(float), *encoded], axis=1)
    require(np.isfinite(result).all(), "INVALID_TRANSFORMED_FEATURES", "Preprocessing produced nonfinite features")
    return result
