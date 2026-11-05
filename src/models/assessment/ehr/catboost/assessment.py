"""Assess frozen CatBoost checkpoints on their development validation folds."""

from importlib.metadata import version

import numpy as np

from src.common.paths import read_json, require
from src.models.assessment.ehr.evaluation import assess_model
from src.models.assessment.plots import plot_history


def checkpoint_prediction(store, manifest, package, fold):
    from catboost import CatBoostRegressor

    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    interface = read_json(output("interface.json"))
    train = package.samples.iloc[fold["train_indices"]]
    validation = package.samples.iloc[fold["validation_indices"]]
    categorical = package.schema["categorical_features"]
    features = list(package.features.columns)
    require(interface["fold"] == fold["fold"] and interface["catboost_version"] == version("catboost")
            and interface["features"] == features and interface["categorical_features"] == categorical
            and interface["train_sample_ids"] == train.sample_id.tolist()
            and interface["validation_sample_ids"] == validation.sample_id.tolist(),
            "INVALID_CHECKPOINT", "Checkpoint input, CatBoost version or fold identity differs")
    model = CatBoostRegressor()
    model.load_model(str(output("model.cbm")), format="cbm")
    require(model.feature_names_ == features and model.tree_count_ == interface["tree_count"]
            and model.get_cat_feature_indices() == [features.index(name) for name in categorical],
            "INVALID_CHECKPOINT", "Saved CatBoost model differs from its declared interface")
    history = read_json(output("history.json"))
    learn, valid = (np.asarray(history[part]["RMSE"], dtype=float) for part in ("learn", "validation"))
    best_round = interface["best_round"]
    require(learn.shape == valid.shape and learn.ndim == 1 and learn.size > 0
            and np.isfinite(learn).all() and np.isfinite(valid).all()
            and best_round == int(np.argmin(valid)) + 1 == interface["tree_count"],
            "INVALID_TRAINING_HISTORY", "RMSE history does not match the checkpoint's best round")
    raw = model.predict(package.features.iloc[fold["validation_indices"]])
    return raw, {"fold": fold["fold"], "best_round": best_round, "learn": learn, "validation": valid}


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, "catboost", checkpoint_prediction, plot_history)
