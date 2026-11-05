"""Freeze full-development EHR fits and score the fixed test split."""

from statistics import median

import numpy as np
from threadpoolctl import threadpool_limits

from src.common.config import load_model_config
from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.ehr.comparison import BASELINE, RULES
from src.models.assessment.ehr.evaluation import metric_pair, training_source
from src.models.assessment.operating_points import (
    measure_operating_point,
    operating_point,
    rule_folder,
)
from src.models.assessment.plots import confusion_plot
from src.models.ehr_data import checked_input_summary, load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData

SAMPLE_COLUMNS = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]


def output(store, manifest, name):
    return store.layout.resolve(store.output_ref(manifest, name))


def run_refit(layout, config_path, config, package, model, fit, predict):
    key = config["id"]
    assessment_id = config["assessment_id"]
    assessment_store = experiment_store(layout, "assessment", "ehr", model, assessment_id)
    assessment_ref = assessment_store.manifest_ref("assessment", "run")
    require(layout.resolve(assessment_ref).is_file(), "ASSESSMENT_NOT_COMPLETE", "Run the selected assessment first")
    assessment = assessment_store.validate(assessment_ref)
    require(assessment["scope"] == assessment_store.scope and assessment["details"]["data_release"] == config["data_release"],
            "ASSESSMENT_DATA_MISMATCH", "Selected assessment must use the configured frozen data release")
    train_id = assessment["details"]["train_id"]
    train_store, train = training_source(layout, package, train_id, model)
    require(assessment["inputs"] == [{"manifest": train["self"], "generation": train["generation"]}],
            "ASSESSMENT_DATA_MISMATCH", "Assessment training binding differs")
    experiment = read_json(output(train_store, train, "effective_configuration.json"))
    require(experiment["id"] == train_id and experiment["data_release"] == config["data_release"],
            "INVALID_TRAINING_CONFIG", "Training snapshot does not match the selected run")

    store = experiment_store(layout, "test", "ehr", model, key)
    ref = store.manifest_ref("test", "refit")
    inputs = [{"manifest": assessment_ref, "generation": assessment["generation"]}]
    existing = completed_result(store, ref, config, inputs)
    if existing:
        return existing

    development = package.cv["development_indices"]
    samples = package.samples.iloc[development]
    features = package.features.iloc[development]
    y = samples.burden_score.to_numpy(float)
    mean = float(y.mean())
    baseline = np.full(len(y), mean)
    points = read_json(output(assessment_store, assessment, "operating_points.json"))
    thresholds = {rule: {"model": {}, BASELINE: {}} for rule in RULES}
    for rule in RULES:
        for endpoint in ENDPOINTS:
            model_point = points[rule]["model"][endpoint]
            target = model_point["target"]
            thresholds[rule]["model"][endpoint] = {field: model_point[field] for field in ("target", "threshold", "comparison")}
            point = operating_point(samples[endpoint].to_numpy(bool), baseline, rule, target)
            thresholds[rule][BASELINE][endpoint] = {field: point[field] for field in ("target", "threshold", "comparison")}

    rounds = None
    if model in {"catboost", "xgboost"}:
        summary = read_json(output(train_store, train, "training_summary.json"))
        values = [fold["best_round"] for fold in summary["folds"]]
        require(len(values) == len(package.cv["folds"]) and all(type(value) is int and value > 0 for value in values),
                "INVALID_REFIT_ROUNDS", "Five positive development best rounds are required")
        rounds = int(median(values))

    with store.publish("test", "refit", inputs=[assessment_ref], configs=[layout.ref("project", config_path)],
                       details={"data_release": config["data_release"], "assessment_id": assessment_id}) as work:
        with threadpool_limits(limits=1):
            expected = fit(features, y, package.schema, experiment["parameters"], rounds, work)
            restored = predict(features, package.schema, rounds, work)
        require(np.isfinite(restored).all() and restored.shape == y.shape, "INVALID_CHECKPOINT", "Invalid refit predictions")
        np.testing.assert_allclose(expected, restored, rtol=0, atol=1e-12)
        write_json(work / "effective_configuration.json", config)
        write_json(work / "thresholds.json", thresholds)
        write_json(work / "fit_summary.json", {"train_sample_ids": samples.sample_id.tolist(),
                   "development_studies": len(y), "development_patients": int(samples.subject_id.nunique()),
                   "baseline_mean": mean, "refit_rounds": rounds, "training_parameters": experiment["parameters"],
                   "test_consumed_for_fitting": False})
    return {"id": key, "status": "complete", "manifest": ref}


def run_evaluate(layout, config_path, config, package, model, predict):
    key = config["id"]
    store = experiment_store(layout, "test", "ehr", model, key)
    refit_ref = store.manifest_ref("test", "refit")
    require(layout.resolve(refit_ref).is_file(), "REFIT_NOT_COMPLETE", "Run refit before evaluate")
    refit = store.validate(refit_ref)
    require(refit["scope"] == store.scope and read_json(output(store, refit, "effective_configuration.json")) == config,
            "REFIT_CONFIG_MISMATCH", "Evaluation must use the frozen refit configuration")
    ref = store.manifest_ref("test", "evaluate")
    inputs = [{"manifest": refit_ref, "generation": refit["generation"]}]
    existing = completed_result(store, ref, config, inputs)
    if existing:
        return existing

    summary = read_json(output(store, refit, "fit_summary.json"))
    development = package.samples.iloc[package.cv["development_indices"]]
    require(summary["train_sample_ids"] == development.sample_id.tolist()
            and summary["baseline_mean"] == float(development.burden_score.to_numpy(float).mean())
            and summary["test_consumed_for_fitting"] is False,
            "INVALID_REFIT_SCOPE", "Refit must use the full development split")
    indices = package.cv["test_indices"]
    rows = package.samples.iloc[indices][SAMPLE_COLUMNS].copy()
    require(not set(rows.subject_id) & set(development.subject_id), "PATIENT_LEAKAGE", "Fixed test and development patients must be disjoint")
    threshold_payload = read_json(output(store, refit, "thresholds.json"))
    with store.publish("test", "evaluate", inputs=[refit_ref], configs=[layout.ref("project", config_path)],
                       details={"data_release": config["data_release"], "assessment_id": config["assessment_id"]}) as work:
        folder = layout.resolve(refit_ref).parent
        with threadpool_limits(limits=1):
            raw = predict(package.features.iloc[indices], package.schema, summary["refit_rounds"], folder)
            require(raw.shape == (len(rows),) and np.isfinite(raw).all(), "INVALID_CHECKPOINT", "Invalid fixed-test predictions")
        scores = {"model": raw, BASELINE: np.full(len(rows), summary["baseline_mean"])}
        predictions = rows.assign(predicted_score=raw, postprocessed_score=np.clip(raw, 0, 8),
                                  baseline_mean_score=scores[BASELINE])
        predictions.to_parquet(work / "test_predictions.parquet", index=False)
        write_json(work / "effective_configuration.json", config)
        write_json(work / "summary.json", {"prediction_role": "fixed_test", "studies": len(rows),
                   "patients": int(rows.subject_id.nunique()),
                   "positive_studies": {name: int(rows[name].sum()) for name in ENDPOINTS},
                   "baseline_mean": summary["baseline_mean"]})
        write_json(work / "metrics.json", {name: metric_pair(rows, values) for name, values in scores.items()})
        points = {}
        for rule in RULES:
            values = {name: {ep: measure_operating_point(rows[ep].to_numpy(bool), scores[name], point)
                             for ep, point in endpoints.items()} for name, endpoints in threshold_payload[rule].items()}
            target = values["model"][ENDPOINTS[0]]["target"]
            folder = work / "threshold_based" / rule_folder(rule, target)
            folder.mkdir(parents=True)
            confusion_plot(values, folder / "confusion_matrices.svg",
                           title=f"Fixed test · development {rule} target {target:.0%}")
            points[rule] = values
        write_json(work / "operating_points.json", points)
    return {"id": key, "status": "complete", "manifest": ref}


def run_test(layout, config_path, model, mode, fit, predict):
    config = load_model_config(layout, config_path)
    require(set(config) == {"data_release", "id", "assessment_id"},
            "INVALID_TEST_CONFIG", "Specify data_release, id and assessment_id")
    identifier(config["id"])
    identifier(config["assessment_id"])
    require(mode in ("refit", "evaluate"), "INVALID_TEST_MODE", "Select refit or evaluate")
    package = load_package(ReleasedData.load(layout, config["data_release"]))
    checked_input_summary(package)
    if mode == "refit":
        return run_refit(layout, config_path, config, package, model, fit, predict)
    return run_evaluate(layout, config_path, config, package, model, predict)
