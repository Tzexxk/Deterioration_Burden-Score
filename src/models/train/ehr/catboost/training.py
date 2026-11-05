"""Fit the CatBoost baseline independently for each configured development experiment."""

from importlib.metadata import version

import numpy as np

from src.common.manifest import file_lock, write_json
from src.common.paths import PipelineError, read_json, require
from src.models.ehr_data import checked_input_summary, load_package
from src.models.experiments import experiment_store, read_batch, update_index
from src.models.released_data import ReleasedData
from src.models.train.ehr.catboost.protocol import validate_training_config


def fit_fold(package, experiment, fold, folder):
    from catboost import CatBoostRegressor

    config = experiment["parameters"]
    policy = config["model"]
    candidate = config["candidate"]
    categorical = package.schema["categorical_features"]
    parameters = {k: v for k, v in policy.items() if k not in ("family", "use_best_model", "early_stopping_rounds")}
    parameters.update(depth=candidate["depth"], l2_leaf_reg=candidate["l2_leaf_reg"],
                      random_seed=config["seed"], cat_features=categorical)
    model = CatBoostRegressor(**parameters)
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


def publish_training(layout, config_path, experiment, package, input_summary):
    key = experiment["id"]
    store = experiment_store(layout, "train", "ehr", "catboost", key)
    ref = store.manifest_ref("train", "run")
    source = package.release.acceptance
    if layout.resolve(ref).is_file():
        manifest = store.validate(ref)
        snapshot = read_json(layout.resolve(store.output_ref(manifest, "effective_configuration.json")))
        require(snapshot == experiment and manifest["inputs"] == [{"manifest": source["self"], "generation": source["generation"]}],
                "EXPERIMENT_ID_CONFLICT", "Completed ID has a different configuration or data release; choose a new ID")
        return {"id": key, "status": "unchanged", "manifest": ref}
    with store.publish("train", "run", inputs=[source["self"]], configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "folds": len(package.cv["folds"])}) as work:
        results = []
        for fold in package.cv["folds"]:
            print(f"{key} fold={fold['fold']}: fitting development split", flush=True)
            results.append(fit_fold(package, experiment, fold, work / f"fold_{fold['fold']}"))
        write_json(work / "input_summary.json", input_summary)
        write_json(work / "effective_configuration.json", experiment)
        write_json(work / "training_summary.json", {"id": key, "folds": results, "test_consumed_for_fitting": False})
    return {"id": key, "status": "complete", "manifest": ref}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_training_config(experiment)
    require(version("catboost") == "1.2.10", "INVALID_CATBOOST_VERSION", "Use CatBoost 1.2.10 for this protocol")
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    input_summary = checked_input_summary(package)
    root = layout.path("outputs", "models/train/ehr/catboost")
    results = []
    with file_lock(root / ".batch.lock"):
        for experiment in experiments:
            try:
                result = publish_training(layout, config_path, experiment, package, input_summary)
            except PipelineError as error:
                result = {"id": experiment["id"], "status": "failed", "error": error.code}
            except Exception as error:
                result = {"id": experiment["id"], "status": "failed", "error": type(error).__name__}
            results.append(result)
        update_index(root, "train")
    return {"status": "failed" if any(item["status"] == "failed" for item in results) else "complete",
            "data_release": experiments[0]["data_release"], "experiments": results}
