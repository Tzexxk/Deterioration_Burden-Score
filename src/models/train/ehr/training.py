"""Shared publication of EHR development-fold training artifacts."""

from src.common.manifest import write_json
from src.common.paths import require
from src.models.ehr_data import checked_input_summary, load_package
from src.models.experiments import completed_result, experiment_store, run_model_batch
from src.models.released_data import ReleasedData


def candidate_settings(experiment, family):
    params = experiment["parameters"]
    require(set(params) == {"candidate"} and isinstance(params["candidate"], dict),
            "INVALID_MODEL_CONFIG", "A candidate object is required")
    candidate = params["candidate"]
    require(set(candidate) == {"name", "family", "params"} and candidate["family"] == family
            and isinstance(candidate["params"], dict), "INVALID_MODEL_CONFIG", "Invalid model candidate")
    return candidate["params"]


def publish_training(layout, config_path, experiment, package, input_summary, model, fit_fold):
    key = experiment["id"]
    store = experiment_store(layout, "train", "ehr", model, key)
    ref = store.manifest_ref("train", "run")
    source = package.release.acceptance
    existing = completed_result(store, ref, experiment, [{"manifest": source["self"], "generation": source["generation"]}])
    if existing:
        return existing
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


def train_model(layout, config_path, experiments, model, fit_fold):
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    input_summary = checked_input_summary(package)
    return run_model_batch(layout, "train", "ehr", model, experiments,
                           lambda experiment: publish_training(layout, config_path, experiment, package,
                                                               input_summary, model, fit_fold))
