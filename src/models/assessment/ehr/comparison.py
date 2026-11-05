"""Compare completed EHR assessments using a shared patient bootstrap."""

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.bootstrap import (
    interval,
    patient_weights,
    rank_metrics,
    ratio,
    threshold_metrics,
)
from src.models.assessment.operating_points import fixed_alerts, rule_folder
from src.models.assessment.plots import interval_plot
from src.models.ehr_data import load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData

BASELINE = "baseline_mean"
RULES = ("sensitivity", "specificity")


def discover(layout):
    root = layout.path("outputs", "models/assessment/ehr")
    sources = []
    for path in sorted(root.glob("*/*/steps/assessment/run/manifest.json")):
        model, key = path.relative_to(root).parts[:2]
        if model == "comparison":
            continue
        store = experiment_store(layout, "assessment", "ehr", model, key)
        ref = store.manifest_ref("assessment", "run")
        manifest = store.validate(ref)
        require(manifest["scope"] == store.scope, "INVALID_ASSESSMENT_SOURCE", f"Invalid assessment scope: {model}/{key}")
        sources.append((model, key, store, manifest))
    require(sources, "NO_ASSESSMENTS", "No completed EHR assessments were found")
    return sources


def load_predictions(package, sources):
    columns = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]
    development = package.samples.iloc[package.cv["development_indices"]][columns].sort_values("sample_id").reset_index(drop=True)
    folds, means = {}, {}
    for fold in package.cv["folds"]:
        for sample in package.samples.iloc[fold["validation_indices"]].sample_id:
            folds[sample] = fold["fold"]
        means[fold["fold"]] = float(package.samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())
    fold_numbers = development.sample_id.map(folds)
    baseline = fold_numbers.map(means).to_numpy(float)
    scores, points = {BASELINE: baseline}, {}
    acceptance = package.release.acceptance
    for model, key, store, manifest in sources:
        name = f"{model}/{key}"
        layout = store.layout
        require(manifest["details"]["data_release"] == package.release.store.data.key,
                "ASSESSMENT_DATA_MISMATCH", f"Data release differs: {name}")
        train_store = experiment_store(layout, "train", "ehr", model, manifest["details"]["train_id"])
        train_ref = train_store.manifest_ref("train", "run")
        require(len(manifest["inputs"]) == 1 and manifest["inputs"][0]["manifest"] == train_ref,
                "ASSESSMENT_DATA_MISMATCH", f"Training binding differs: {name}")
        # The assessment manifest validation already checked the training dependency recursively.
        train = read_json(layout.resolve(train_ref))
        require(train["scope"] == train_store.scope and
                {"manifest": acceptance["self"], "generation": acceptance["generation"]} in train["inputs"],
                "ASSESSMENT_DATA_MISMATCH", f"Frozen training dataset differs: {name}")

        def output(filename):
            return layout.resolve(store.output_ref(manifest, filename))

        require(any(item["name"] == "operating_points.json" for item in manifest["outputs"]),
                "ASSESSMENT_UPGRADE_REQUIRED", f"Rebuild both operating points: {name}")
        actual = pd.read_parquet(output("oof_predictions.parquet")).sort_values("sample_id").reset_index(drop=True)
        require(actual.sample_id.is_unique and all(np.array_equal(actual[k], development[k]) for k in columns)
                and np.array_equal(actual.fold, fold_numbers) and np.array_equal(actual.baseline_mean_score, baseline),
                "INVALID_OOF_COVERAGE", f"OOF identities, labels, folds or baseline differ: {name}")
        values = actual.predicted_score.to_numpy(float)
        require(np.isfinite(values).all(), "INVALID_OOF_PREDICTIONS", f"OOF scores must be finite: {name}")
        scores[name] = values
        stored = read_json(output("operating_points.json"))
        baseline_points = {rule: stored[rule]["baseline_mean"] for rule in RULES}
        points.setdefault(BASELINE, baseline_points)
        require(points[BASELINE] == baseline_points, "OPERATING_POINT_MISMATCH", f"Baseline operating points differ: {name}")
        points[name] = {rule: stored[rule]["model"] for rule in RULES}
    return development, scores, points


def regression_metrics(labels, scores, weights):
    error = scores - labels
    count = weights.sum(axis=1)
    return {"rmse": np.sqrt(ratio(weights @ (error ** 2), count)), "mae": ratio(weights @ np.abs(error), count)}


def calculate_comparison(rows, scores, points, draws, seed):
    weights = np.vstack([np.ones(len(rows)), patient_weights(rows.subject_id.to_numpy(), draws, seed)])
    y = rows.burden_score.to_numpy(float)
    targets = {endpoint: rows[endpoint].to_numpy(bool) for endpoint in ENDPOINTS}
    baseline = regression_metrics(y, scores[BASELINE], weights)
    targets_by_rule = {rule: points[BASELINE][rule][ENDPOINTS[0]]["target"] for rule in RULES}
    free, based = {}, {rule: {} for rule in RULES}
    for name, values in scores.items():
        metrics = regression_metrics(y, values, weights)
        for key, error in list(metrics.items()):
            metrics[f"{key}_relative_pct"] = 100 * ratio(error - baseline[key], baseline[key])
        for endpoint, labels in targets.items():
            auc, ap = rank_metrics(labels, values, weights)
            metrics.update({f"{endpoint}_auroc": auc, f"{endpoint}_ap": ap})
        free[name] = {key: interval(v[0], v[1:]) for key, v in metrics.items()}
        for rule in RULES:
            results = {}
            for endpoint, labels in targets.items():
                point = points[name][rule][endpoint]
                threshold = point["threshold"]
                require(point["target"] == targets_by_rule[rule],
                        "OPERATING_POINT_MISMATCH", f"Operating-point target differs: {name}/{endpoint}/{rule}")
                require((threshold is None and rule == "specificity" and point["comparison"] == "no_alerts")
                        or (type(threshold) in (int, float) and np.isfinite(threshold)
                            and point["comparison"] == "score >= threshold"),
                        "OPERATING_POINT_MISMATCH", f"Invalid frozen threshold: {name}/{endpoint}/{rule}")
                estimates, counts = threshold_metrics(labels, fixed_alerts(labels, values, threshold), weights)
                calculated = {key: interval(v[0], v[1:]) for key, v in estimates.items()}
                require({key: int(v[0]) for key, v in counts.items()} == point["confusion_matrix"]
                        and {key: v["point"] for key, v in calculated.items()} == point["metrics"]
                        and calculated[rule]["point"] >= point["target"],
                        "OPERATING_POINT_MISMATCH", f"Frozen operating point disagrees with predictions: {name}/{endpoint}/{rule}")
                results[endpoint] = {"target": point["target"], "threshold": threshold,
                                     "comparison": point["comparison"], "metrics": calculated}
            based[rule][name] = results
    return free, based, targets_by_rule


def write_comparison(work, rows, free, based, targets, snapshot):
    write_json(work / "effective_configuration.json", snapshot)
    write_json(work / "summary.json", {"studies": len(rows), "patients": int(rows.subject_id.nunique()),
                                      "positive_studies": {k: int(rows[k].sum()) for k in ENDPOINTS},
                                      "reference": BASELINE, "confidence_level": 0.95,
                                      "bootstrap_unit": "subject_id", "thresholds_fixed_during_resampling": True})
    destination = work / "threshold_free"
    write_json(destination / "metrics.json", free)
    metrics = [("rmse_relative_pct", "RMSE difference (%)"), ("mae_relative_pct", "MAE difference (%)")]
    metrics.extend((f"{endpoint}_{name}", f"{endpoint} {name.upper()}")
                   for endpoint in ENDPOINTS for name in ("auroc", "ap"))
    interval_plot(free, metrics, destination / "metrics.svg", title="Threshold-free OOF metrics")
    for rule, result in based.items():
        target = targets[rule]
        destination = work / "threshold_based" / rule_folder(rule, target)
        write_json(destination / "metrics.json", result)
        selected = ("sensitivity", "specificity", "ppv" if rule == "sensitivity" else "npv")
        flattened = {key: {f"{ep}_{name}": value["metrics"][name]
                           for ep, value in endpoints.items() for name in selected}
                     for key, endpoints in result.items()}
        panels = [(f"{ep}_{name}", f"{ep} {name.upper()}") for ep in ENDPOINTS for name in selected]
        interval_plot(flattened, panels, destination / "metrics.svg", title=f"OOF: {rule} ≥ {target:.0%}")


def run_comparison(layout, key, *, draws=5000, seed=42):
    identifier(key)
    sources = discover(layout)
    release = sources[0][3]["details"]["data_release"]
    mismatched = [f"{model}/{candidate}" for model, candidate, _, manifest in sources
                  if manifest["details"]["data_release"] != release]
    require(not mismatched, "ASSESSMENT_DATA_MISMATCH", f"Data release differs: {', '.join(mismatched)}")
    package = load_package(ReleasedData.load(layout, release))
    rows, scores, points = load_predictions(package, sources)
    inputs = [{"manifest": manifest["self"], "generation": manifest["generation"]} for _, _, _, manifest in sources]
    snapshot = {"id": key, "data_release": release,
                "parameters": {"assessment_ids": [f"{model}/{candidate}" for model, candidate, _, _ in sources],
                               "bootstrap_draws": draws, "bootstrap_seed": seed}}
    store = experiment_store(layout, "assessment", "ehr", "comparison", key)
    ref = store.manifest_ref("comparison", "run")
    existing = completed_result(store, ref, snapshot, inputs)
    if existing:
        return existing
    with store.publish("comparison", "run", inputs=[item["manifest"] for item in inputs],
                       details={"data_release": release, "assessment_ids": snapshot["parameters"]["assessment_ids"]}) as work:
        free, based, targets = calculate_comparison(rows, scores, points, draws, seed)
        write_comparison(work, rows, free, based, targets, snapshot)
    return {"id": key, "status": "complete", "manifest": ref}
