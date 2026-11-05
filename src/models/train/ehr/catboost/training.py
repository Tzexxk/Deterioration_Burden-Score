"""Fit the CatBoost baseline independently for each configured development experiment."""

from importlib.metadata import version

import numpy as np

from src.common.manifest import write_json
from src.common.paths import require
from src.models.experiments import (
    read_batch,
)
from src.models.train.ehr.catboost.protocol import validate_training_config
from src.models.train.ehr.training import train_model


def create_model(config, categorical, *, rounds=None):
    from catboost import CatBoostRegressor

    policy = config["model"]
    candidate = config["candidate"]
    parameters = {k: v for k, v in policy.items() if k not in ("family", "use_best_model", "early_stopping_rounds")}
    parameters.update(depth=candidate["depth"], l2_leaf_reg=candidate["l2_leaf_reg"],
                      random_seed=config["seed"], cat_features=categorical)
    if rounds is not None:
        parameters["iterations"] = rounds
    return CatBoostRegressor(**parameters)


def fit_fold(package, experiment, fold, folder):
    from catboost import CatBoostRegressor

    config = experiment["parameters"]
    policy = config["model"]
    categorical = package.schema["categorical_features"]
    model = create_model(config, categorical)
    train, validation = fold["train_indices"], fold["validation_indices"]
    x, y = package.features, package.samples.burden_score
    model.fit(x.iloc[train], y.iloc[train].to_numpy(float),
              eval_set=(x.iloc[validation], y.iloc[validation].to_numpy(float)),
              use_best_model=policy["use_best_model"], early_stopping_rounds=policy["early_stopping_rounds"])
    best_round = int(model.get_best_iteration()) + 1
    require(best_round > 0, "INVALID_CHECKPOINT", "CatBoost did not select a validation round")
    folder.mkdir()
    checkpoint = folder / "model.cbm"
    model.save_model(str(checkpoint), format="cbm")
    restored = CatBoostRegressor()
    restored.load_model(str(checkpoint), format="cbm")
    np.testing.assert_allclose(restored.predict(x.iloc[validation]), model.predict(x.iloc[validation]), rtol=0, atol=1e-12)
    interface = {"fold": fold["fold"], "best_round": best_round, "tree_count": int(model.tree_count_),
                 "features": list(x.columns), "categorical_features": categorical, "catboost_version": version("catboost"),
                 "train_sample_ids": package.samples.iloc[train].sample_id.tolist(),
                 "validation_sample_ids": package.samples.iloc[validation].sample_id.tolist()}
    write_json(folder / "interface.json", interface)
    write_json(folder / "history.json", model.get_evals_result())
    write_json(folder / "parameters.json", model.get_all_params())
    return {"fold": fold["fold"], "best_round": best_round, "tree_count": interface["tree_count"]}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_training_config(experiment)
    require(version("catboost") == "1.2.10", "INVALID_CATBOOST_VERSION", "Use CatBoost 1.2.10 for this protocol")
    return train_model(layout, config_path, experiments, "catboost", fit_fold)
