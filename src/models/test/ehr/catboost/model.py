"""Fit and load the full-development CatBoost checkpoint."""

from importlib.metadata import version

from catboost import CatBoostRegressor

from src.common.paths import require
from src.models.train.ehr.catboost.training import create_model


def fit(features, labels, schema, settings, rounds, folder):
    require(version("catboost") == "1.2.10", "INVALID_CATBOOST_VERSION", "Use CatBoost 1.2.10")
    model = create_model(settings, schema["categorical_features"], rounds=rounds)
    model.fit(features, labels, use_best_model=False)
    model.save_model(str(folder / "model.cbm"), format="cbm")
    return model.predict(features)


def predict(features, schema, rounds, folder):
    require(version("catboost") == "1.2.10", "INVALID_CATBOOST_VERSION", "Use CatBoost 1.2.10")
    model = CatBoostRegressor()
    model.load_model(str(folder / "model.cbm"), format="cbm")
    require(model.feature_names_ == list(features.columns) and model.tree_count_ == rounds
            and model.get_cat_feature_indices() == [features.columns.get_loc(key) for key in schema["categorical_features"]],
            "INVALID_CHECKPOINT", "CatBoost refit width, round count or categorical features differ")
    return model.predict(features)
