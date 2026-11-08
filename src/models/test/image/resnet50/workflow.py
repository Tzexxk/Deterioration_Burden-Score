"""Refit on the frozen development split, then evaluate the fixed test split."""

import sys
from statistics import median

import numpy as np

from src.common.config import load_model_config
from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.evaluation import training_source, validate_parameters
from src.models.experiments import completed_result, experiment_store
from src.models.image_data import load_image_package
from src.models.released_data import ReleasedData
from src.models.test.evaluation import (
    SAMPLE_COLUMNS,
    frozen_thresholds,
    output,
    write_fixed_evaluation,
)
from src.models.test.image.resnet50 import model as adapter
from src.models.train.image.resnet50.training import MODEL
from src.models.train.image.training import image_runtime_versions


def checked_package(release, settings, source, shape, runtime):
    require(runtime == image_runtime_versions(settings), "IMAGE_RUNTIME_MISMATCH",
            "Use the image runtime recorded by the development training")
    package = load_image_package(release, settings["image_source"])
    require(package.source == source and list(package.shape) == shape, "TEST_IMAGE_SOURCE_MISMATCH",
            "Fixed testing must use the development image release and dimensions")
    return package


def run_refit(layout, config_path, config, release, store):
    key = config["id"]
    assessment_id = config["assessment_id"]
    assessment_store = experiment_store(layout, "assessment", "image", MODEL, assessment_id)
    assessment_ref = assessment_store.manifest_ref("assessment", "run")
    require(layout.resolve(assessment_ref).is_file(), "ASSESSMENT_NOT_COMPLETE", "Run the selected assessment first")
    assessment = assessment_store.validate(assessment_ref)
    require(assessment["scope"] == assessment_store.scope and assessment["details"]["data_release"] == config["data_release"],
            "ASSESSMENT_DATA_MISMATCH", "Selected assessment must use the configured frozen data release")
    assessment_config = read_json(output(assessment_store, assessment, "effective_configuration.json"))
    validate_parameters(assessment_config)
    train_id = assessment["details"]["train_id"]
    require(assessment_config["id"] == assessment_id and assessment_config["data_release"] == config["data_release"]
            and assessment_config["parameters"]["train_id"] == train_id,
            "ASSESSMENT_DATA_MISMATCH", "Published assessment configuration differs from its binding")
    train_store, train = training_source(layout, release, train_id, MODEL, modality="image")
    require(assessment["inputs"] == [{"manifest": train["self"], "generation": train["generation"]}],
            "ASSESSMENT_DATA_MISMATCH", "Assessment training binding differs")
    snapshot = read_json(output(train_store, train, "effective_configuration.json"))
    require(snapshot["id"] == train_id and snapshot["data_release"] == config["data_release"],
            "INVALID_TRAINING_CONFIG", "Training snapshot does not match the selected run")
    settings = snapshot["parameters"]
    package = checked_package(release, settings, snapshot["resolved_image_source"], snapshot["image_shape"],
                              snapshot["runtime_versions"])
    ref = store.manifest_ref("test", "refit")
    existing = completed_result(store, ref, config, [{"manifest": assessment_ref, "generation": assessment["generation"]}])
    if existing:
        return existing

    folds = read_json(output(train_store, train, "training_summary.json"))["folds"]
    require([f["fold"] for f in folds] == [f["fold"] for f in package.cv["folds"]]
            and all(type(f["best_epoch"]) is int and 0 < f["best_epoch"] <= f["epochs"] <= settings["max_epochs"]
                    for f in folds), "INVALID_REFIT_ROUNDS", "Expected valid development best epochs for every fold")
    rounds = int(median(f["best_epoch"] for f in folds))
    development = package.samples.iloc[package.cv["development_indices"]]
    mean = float(development.burden_score.to_numpy(float).mean())
    thresholds = frozen_thresholds(development, mean, read_json(output(assessment_store, assessment, "operating_points.json")))
    with store.publish("test", "refit", inputs=[assessment_ref], configs=[layout.ref("project", config_path)],
                       details={"data_release": config["data_release"], "assessment_id": assessment_id}) as work:
        print(f"{MODEL}/{key} refit: fitting {len(development)} development studies", file=sys.stderr, flush=True)
        device = adapter.refit(layout, package, settings, rounds, work, label=f"{MODEL}/{key} refit")
        write_json(work / "effective_configuration.json", config)
        write_json(work / "thresholds.json", thresholds)
        write_json(work / "fit_summary.json", {"train_sample_ids": development.sample_id.tolist(),
                   "development_studies": len(development), "development_patients": int(development.subject_id.nunique()),
                   "baseline_mean": mean, "refit_rounds": rounds, "training_parameters": settings,
                   "resolved_image_source": package.source, "image_shape": list(package.shape),
                   "runtime_versions": snapshot["runtime_versions"], "device": device, "test_consumed_for_fitting": False})
    return {"id": key, "status": "complete", "manifest": ref}


def run_evaluate(layout, config_path, config, release, store):
    key = config["id"]
    refit_ref = store.manifest_ref("test", "refit")
    require(layout.resolve(refit_ref).is_file(), "REFIT_NOT_COMPLETE", "Run refit before evaluate")
    refit = store.validate(refit_ref)
    require(refit["scope"] == store.scope and read_json(output(store, refit, "effective_configuration.json")) == config,
            "REFIT_CONFIG_MISMATCH", "Evaluation must use the frozen refit configuration")
    ref = store.manifest_ref("test", "evaluate")
    existing = completed_result(store, ref, config, [{"manifest": refit_ref, "generation": refit["generation"]}])
    if existing:
        return existing
    summary = read_json(output(store, refit, "fit_summary.json"))
    settings = summary["training_parameters"]
    package = checked_package(release, settings, summary["resolved_image_source"], summary["image_shape"],
                              summary["runtime_versions"])
    development = package.samples.iloc[package.cv["development_indices"]]
    require(summary["train_sample_ids"] == development.sample_id.tolist()
            and summary["baseline_mean"] == float(development.burden_score.to_numpy(float).mean())
            and summary["test_consumed_for_fitting"] is False,
            "INVALID_REFIT_SCOPE", "Refit must use the full development split")
    indices = package.cv["test_indices"]
    rows = package.samples.iloc[indices][SAMPLE_COLUMNS].copy()
    require(set(rows.subject_id).isdisjoint(development.subject_id), "PATIENT_LEAKAGE",
            "Fixed test and development patients must be disjoint")
    thresholds = read_json(output(store, refit, "thresholds.json"))
    with store.publish("test", "evaluate", inputs=[refit_ref], configs=[layout.ref("project", config_path)],
                       details={"data_release": config["data_release"], "assessment_id": config["assessment_id"]}) as work:
        print(f"{MODEL}/{key} evaluate: scoring {len(rows)} fixed-test studies", file=sys.stderr, flush=True)
        raw = adapter.predict(package, indices, settings, output(store, refit, "model.pt"), summary["device"])
        require(raw.shape == (len(rows),) and np.isfinite(raw).all(), "INVALID_CHECKPOINT", "Invalid fixed-test scores")
        write_fixed_evaluation(work, config, rows, raw, summary["baseline_mean"], thresholds)
    return {"id": key, "status": "complete", "manifest": ref}


def run_test(layout, config_path, mode):
    config = load_model_config(layout, config_path)
    require(set(config) == {"data_release", "id", "assessment_id"},
            "INVALID_TEST_CONFIG", "Specify data_release, id and assessment_id")
    identifier(config["id"])
    identifier(config["assessment_id"])
    require(mode in ("refit", "evaluate"), "INVALID_TEST_MODE", "Select refit or evaluate")
    print(f"{MODEL}/{config['id']} {mode}: checking frozen release {config['data_release']}",
          file=sys.stderr, flush=True)
    release = ReleasedData.load(layout, config["data_release"])
    store = experiment_store(layout, "test", "image", MODEL, config["id"])
    run = run_refit if mode == "refit" else run_evaluate
    return run(layout, config_path, config, release, store)
