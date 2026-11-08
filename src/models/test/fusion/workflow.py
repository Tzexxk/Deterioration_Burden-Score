"""Freeze development-only fusion rules and evaluate them on shared fixed-test scores."""

from statistics import median

import numpy as np
import pandas as pd

from src.common.config import load_model_config
from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.comparison import BASELINE, RULES, load_predictions
from src.models.assessment.fusion.learned_late.assessment import fit_coefficients
from src.models.assessment.operating_points import operating_point
from src.models.ehr_data import checked_input_summary, load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData
from src.models.test.comparison import load_results
from src.models.test.evaluation import (
    POINT_FIELDS,
    frozen_thresholds,
    output,
    write_fixed_evaluation,
)

CANDIDATES = {"weighted_late": "f1", "learned_late": "f3b"}
BASE_MODELS = {"ehr": "catboost", "image": "resnet50"}
TEST_HISTORY = "Both source test results were viewed and used for source selection."


def binding(manifest):
    return {"manifest": manifest["self"], "generation": manifest["generation"]}


def checked_manifest(store, stage, phase, release):
    manifest = store.validate(store.manifest_ref(stage, phase))
    require(manifest["scope"] == store.scope and manifest["details"]["data_release"] == release,
            "TEST_DATA_MISMATCH", "Selected source must match its scope and frozen release")
    return manifest


def load_refits(layout, config, package, development):
    refits, assessments = {}, {}
    for family, model in BASE_MODELS.items():
        store = experiment_store(layout, "test", family, model, config[f"{family}_refit_id"])
        manifest = checked_manifest(store, "test", "refit", config["data_release"])
        assessment_id = manifest["details"]["assessment_id"]
        source_store = experiment_store(layout, "assessment", family, model, assessment_id)
        source_ref = source_store.manifest_ref("assessment", "run")
        source = read_json(layout.resolve(source_ref))
        require(manifest["inputs"] == [binding(source)] and source["scope"] == source_store.scope
                and len(source["inputs"]) == 1, "INVALID_REFIT_SOURCE", "Refit must bind its exact source assessment")
        train = read_json(layout.resolve(source["inputs"][0]["manifest"]))
        train_config = read_json(output(store, train, "effective_configuration.json"))
        summary = read_json(output(store, manifest, "fit_summary.json"))
        fold_summary = read_json(output(store, train, "training_summary.json"))["folds"]
        lengths = [f["best_round" if family == "ehr" else "best_epoch"] for f in fold_summary]
        require([f["fold"] for f in fold_summary] == [f["fold"] for f in package.cv["folds"]]
                and all(type(n) is int and n > 0 for n in lengths)
                and summary["refit_rounds"] == int(median(lengths))
                and summary["training_parameters"] == train_config["parameters"]
                and summary["train_sample_ids"] == development.sample_id.tolist()
                and summary["baseline_mean"] == float(development.burden_score.to_numpy(float).mean())
                and summary["test_consumed_for_fitting"] is False,
                "INVALID_REFIT_SCOPE", "Refit must match the full development cohort, settings and median length")
        if family == "image":
            require(all(summary[k] == train_config[k] for k in ("resolved_image_source", "image_shape", "runtime_versions")),
                    "TEST_IMAGE_SOURCE_MISMATCH", "Refit image source or runtime differs from training")
        output(store, manifest, "model.cbm" if family == "ehr" else "model.pt")
        refits[family] = (store, manifest)
        assessments[family] = (model, assessment_id, source_store, source)
    return refits, assessments



def freeze_development(layout, config, package, model):
    development = package.samples.iloc[package.cv["development_indices"]]
    mean = float(development.burden_score.to_numpy(float).mean())
    refits, sources = load_refits(layout, config, package, development)
    base_inputs = [binding(item[3]) for item in sources.values()]
    key = config["assessment_id"]
    store = experiment_store(layout, "assessment", "fusion", model, key)
    assessment = checked_manifest(store, "assessment", "run", config["data_release"])
    predictions = pd.read_parquet(output(store, assessment, "oof_predictions.parquet"))
    predictions = predictions.sort_values("sample_id").reset_index(drop=True)
    dev_scores = {}
    for family, item in sources.items():
        rows, scores, _ = load_predictions(package.release, package.samples, package.cv, [item], modality=family)
        dev_scores[family] = scores[f"{item[0]}/{item[1]}"]
        base_points = read_json(output(item[2], item[3], "operating_points.json"))
        for rule in RULES:
            for endpoint in ENDPOINTS:
                point = operating_point(rows[endpoint].to_numpy(bool), dev_scores[family], rule, 0.8)
                require(base_points[rule]["model"][endpoint] == point,
                        "OPERATING_POINT_MISMATCH", "Base assessment point differs from its OOF scores")
        refit_store, refit = refits[family]
        require(read_json(output(refit_store, refit, "thresholds.json"))
                == frozen_thresholds(development, mean, base_points),
                "OPERATING_POINT_MISMATCH", "Base refit thresholds differ from its development assessment")
    folds = {sample: fold["fold"] for fold in package.cv["folds"]
             for sample in package.samples.iloc[fold["validation_indices"]].sample_id}
    rows = rows.assign(fold=rows.sample_id.map(folds))
    require(all(np.array_equal(predictions[col], rows[col]) for col in rows)
            and np.array_equal(predictions.baseline_mean_score, scores[BASELINE]),
            "INVALID_OOF_COVERAGE", "Fusion assessment must cover the frozen development OOF cohort")

    candidate = CANDIDATES[model]
    if model == "weighted_late":
        require(assessment["inputs"] == base_inputs,
                "INVALID_F1_SOURCE", "F1 must bind both selected base assessments")
        parameters = read_json(output(store, assessment, "effective_configuration.json"))["parameters"]
        alpha = parameters["image_weight"]
        require(alpha == config["image_weight"], "INVALID_F1_WEIGHT", "F1 image weight differs from the test plan")
        require(np.array_equal(predictions.predicted_score, (1-alpha)*dev_scores["ehr"] + alpha*dev_scores["image"])
                and np.array_equal(predictions.ehr_shrink_score, (1-alpha)*dev_scores["ehr"] + alpha*scores[BASELINE]),
                "INVALID_F1_PREDICTIONS", "F1 and its shrinkage control must match their published formula")
        coefficients = {candidate: {"alpha": alpha, "bias": 0.0}}
        controls = {"ehr_shrink": "ehr_shrink_score"}
    else:
        load_predictions(package.release, package.samples, package.cv,
                         [(model, key, store, assessment)], modality="fusion")
        train = read_json(layout.resolve(assessment["inputs"][0]["manifest"]))
        parameters = read_json(output(store, train, "effective_configuration.json"))["parameters"]
        require(train["inputs"] == [binding(package.release.acceptance), *base_inputs]
                and parameters["round_policy"] == "training_median"
                and np.array_equal(predictions.predicted_score, predictions.f3b),
                "INVALID_F3_SOURCE", "F3b must use the matched training-median protocol")
        require(np.isfinite(predictions[["ehr", "image"]].to_numpy(float)).all(),
                "INVALID_OOF_PREDICTIONS", "Nested base scores must be finite")
        coefficients = fit_coefficients(predictions.ehr.to_numpy(float), predictions.image.to_numpy(float),
                                        predictions.burden_score.to_numpy(float))
        controls = {"f3a": "f3a", "ehr_bias": "ehr_bias"}

    published = read_json(output(store, assessment, "operating_points.json"))
    thresholds = frozen_thresholds(development, mean, published)
    for name, column in {"model": "predicted_score", **controls}.items():
        values = predictions[column].to_numpy(float)
        require(np.isfinite(values).all(), "INVALID_OOF_PREDICTIONS", "Development scores must be finite")
        for rule in RULES:
            thresholds[rule][name] = {}
            for endpoint in ENDPOINTS:
                point = operating_point(rows[endpoint].to_numpy(bool), values, rule, 0.8)
                require(published[rule][name][endpoint] == point,
                        "OPERATING_POINT_MISMATCH", "Development point differs from its OOF scores")
                thresholds[rule][name][endpoint] = {field: point[field] for field in POINT_FIELDS}

    summary = {"development_studies": len(rows), "development_patients": int(rows.subject_id.nunique()),
               "baseline_mean": mean, "candidate": candidate, "mechanism_controls": list(controls),
               "coefficients": coefficients, "base_refits": {k: binding(v[1]) for k, v in refits.items()},
               "threshold_sources": {**dict.fromkeys((candidate, *controls), binding(assessment)),
                                     BASELINE: "complete_development_mean"},
               "prior_fixed_test_exposure": TEST_HISTORY, "test_consumed_for_fitting": False}
    if model == "weighted_late":
        summary["decision_margins"] = parameters["decision_rules"]
    inputs = [binding(item[1]) for item in refits.values()] + [binding(assessment)]
    return summary, thresholds, inputs


def fusion_scores(summary, scores):
    combined = {}
    for name, coef in summary["coefficients"].items():
        alpha, bias = coef["alpha"], coef["bias"]
        combined[name] = (1-alpha)*scores["ehr"] + alpha*scores["image"] + bias
    if "ehr_shrink" in summary["mechanism_controls"]:
        alpha = summary["coefficients"]["f1"]["alpha"]
        combined["ehr_shrink"] = (1-alpha)*scores["ehr"] + alpha*scores[BASELINE]
    return combined


def publish_refit(layout, config_path, config, package, store):
    summary, thresholds, inputs = freeze_development(layout, config, package, store.scope["model"])
    ref = store.manifest_ref("test", "refit")
    existing = completed_result(store, ref, config, inputs)
    if existing:
        return existing
    with store.publish("test", "refit", inputs=[i["manifest"] for i in inputs], configs=[layout.ref("project", config_path)],
                       details={"data_release": config["data_release"], "prediction_role": "development_freeze"}) as work:
        require(all(binding(read_json(layout.resolve(i["manifest"]))) == i for i in inputs),
                "UPSTREAM_REBUILT", "A source changed while preparing the freeze")
        write_json(work / "effective_configuration.json", config)
        write_json(work / "fit_summary.json", summary)
        write_json(work / "thresholds.json", thresholds)
    return {"id": config["id"], "status": "complete", "manifest": ref}


def publish_evaluate(layout, config_path, config, package, store):
    frozen = checked_manifest(store, "test", "refit", config["data_release"])
    require(read_json(output(store, frozen, "effective_configuration.json")) == config,
            "REFIT_CONFIG_MISMATCH", "Evaluation must use the frozen configuration")
    summary = read_json(output(store, frozen, "fit_summary.json"))
    thresholds = read_json(output(store, frozen, "thresholds.json"))
    sources = []
    for family, model in BASE_MODELS.items():
        source_store = experiment_store(layout, "test", family, model, config[f"{family}_refit_id"])
        evaluation = checked_manifest(source_store, "test", "evaluate", config["data_release"])
        require(evaluation["inputs"] == [summary["base_refits"][family]],
                "INVALID_TEST_EVALUATION", "Base test scores must bind the frozen refits")
        sources.append((family, source_store, evaluation))
    inputs = [binding(frozen), *(binding(item[2]) for item in sources)]
    ref = store.manifest_ref("test", "evaluate")
    existing = completed_result(store, ref, config, inputs)
    if existing:
        return existing
    rows, scores, _ = load_results(package, sources)
    dev = package.samples.iloc[package.cv["development_indices"]]
    require(set(rows.subject_id).isdisjoint(dev.subject_id)
            and np.array_equal(scores[BASELINE], np.full(len(rows), summary["baseline_mean"])),
            "INVALID_TEST_COVERAGE", "Test patients must be disjoint from development and use the frozen mean")
    raw = fusion_scores(summary, scores)[summary["candidate"]]
    primary_thresholds = {rule: {name: thresholds[rule][name] for name in ("model", BASELINE)} for rule in RULES}
    with store.publish("test", "evaluate", inputs=[i["manifest"] for i in inputs],
                       configs=[layout.ref("project", config_path)],
                       details={"data_release": config["data_release"], "prediction_role": "exploratory_fixed_test"}) as work:
        require(all(binding(read_json(layout.resolve(i["manifest"]))) == i for i in inputs),
                "UPSTREAM_REBUILT", "A source changed while loading fixed-test scores")
        write_fixed_evaluation(work, config, rows, raw, summary["baseline_mean"], primary_thresholds)
        report = read_json(work / "summary.json")
        report.update(prediction_role="exploratory_fixed_test", prior_fixed_test_exposure=TEST_HISTORY)
        write_json(work / "summary.json", report)
    return {"id": config["id"], "status": "complete", "manifest": ref}


def run_test(layout, config_path, mode, model):
    config = load_model_config(layout, config_path)
    identifiers = ("id", "assessment_id", "ehr_refit_id", "image_refit_id")
    required = {"data_release", *identifiers}
    if model == "weighted_late":
        required.add("image_weight")
    require(model in CANDIDATES and set(config) == required,
            "INVALID_TEST_CONFIG", "Specify a release, assessment ID, both base refit IDs and the fixed weight for F1")
    for key in identifiers:
        identifier(config[key])
    if model == "weighted_late":
        alpha = config["image_weight"]
        require(type(alpha) in (int, float) and 0 < alpha < 1, "INVALID_TEST_CONFIG", "Invalid image weight")
    require(mode in ("refit", "evaluate"), "INVALID_TEST_MODE", "Select refit or evaluate")
    package = load_package(ReleasedData.load(layout, config["data_release"]))
    checked_input_summary(package)
    store = experiment_store(layout, "test", "fusion", model, config["id"])
    run = publish_refit if mode == "refit" else publish_evaluate
    return run(layout, config_path, config, package, store)
