"""Compare completed development assessments using a shared patient bootstrap."""

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
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData

BASELINE = "baseline_mean"
RULES = ("sensitivity", "specificity")


def discover(layout, modality, *, progress=None):
    root = layout.path("outputs", f"models/assessment/{modality}")
    sources = []
    paths = sorted(path for path in root.glob("*/*/manifest.json")
                   if not path.relative_to(root).parts[0].startswith(".")
                   and path.relative_to(root).parts[0] != "comparison")
    if progress:
        progress(f"Validating assessments: 0/{len(paths)}")
    for index, path in enumerate(paths, 1):
        model, key = path.relative_to(root).parts[:2]
        store = experiment_store(layout, "assessment", modality, model, key)
        ref = store.manifest_ref("assessment", "run")
        manifest = store.validate(ref)
        require(manifest["scope"] == store.scope, "INVALID_ASSESSMENT_SOURCE", f"Invalid assessment scope: {model}/{key}")
        sources.append((model, key, store, manifest))
        if progress:
            progress(f"Validating assessments: {index}/{len(paths)} ({model}/{key})")
    require(sources, "NO_ASSESSMENTS", "No completed development assessments were found")
    return sources


def load_predictions(release, samples, cv, sources, *, modality):
    columns = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]
    development = samples.iloc[cv["development_indices"]][columns].sort_values("sample_id").reset_index(drop=True)
    folds, means = {}, {}
    for fold in cv["folds"]:
        for sample in samples.iloc[fold["validation_indices"]].sample_id:
            folds[sample] = fold["fold"]
        means[fold["fold"]] = float(samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())
    fold_numbers = development.sample_id.map(folds)
    baseline = fold_numbers.map(means).to_numpy(float)
    scores, points = {BASELINE: baseline}, {}
    acceptance = release.acceptance
    for model, key, store, manifest in sources:
        name = f"{model}/{key}"
        layout = store.layout
        require(manifest["details"]["data_release"] == release.store.data.key,
                "ASSESSMENT_DATA_MISMATCH", f"Data release differs: {name}")
        train_store = experiment_store(layout, "train", modality, model, manifest["details"]["train_id"])
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


def calculate_comparison(rows, scores, points, draws, seed, *, progress=None, fixed_test=False):
    if progress:
        progress(f"Bootstrap ({draws:,} draws): 0/{len(scores)}")
    weights = np.vstack([np.ones(len(rows)), patient_weights(rows.subject_id.to_numpy(), draws, seed)])
    y = rows.burden_score.to_numpy(float)
    targets = {endpoint: rows[endpoint].to_numpy(bool) for endpoint in ENDPOINTS}
    baseline = regression_metrics(y, scores[BASELINE], weights)
    targets_by_rule = {rule: points[BASELINE][rule][ENDPOINTS[0]]["target"] for rule in RULES}
    free, based = {}, {rule: {} for rule in RULES}
    for index, (name, values) in enumerate(scores.items(), 1):
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
                matrix = {key: int(v[0]) for key, v in counts.items()}
                require(matrix == point["confusion_matrix"]
                        and {key: v["point"] for key, v in calculated.items()} == point["metrics"]
                        and (fixed_test or calculated[rule]["point"] >= point["target"]),
                        "OPERATING_POINT_MISMATCH", f"Frozen operating point disagrees with predictions: {name}/{endpoint}/{rule}")
                results[endpoint] = {"target": point["target"], "threshold": threshold,
                                     "comparison": point["comparison"], "metrics": calculated}
                if fixed_test:
                    results[endpoint]["confusion_matrix"] = matrix
            based[rule][name] = results
        if progress:
            progress(f"Bootstrap ({draws:,} draws): {index}/{len(scores)} ({name})")
    return free, based, targets_by_rule


def write_comparison(work, rows, free, based, targets, snapshot, *, progress=None, cohort="OOF"):
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
    plots = [(destination, free, metrics, f"Threshold-free {cohort}")]
    for rule, result in based.items():
        target = targets[rule]
        destination = work / "threshold_based" / rule_folder(rule, target)
        write_json(destination / "metrics.json", result)
        selected = ("sensitivity", "specificity", "ppv" if rule == "sensitivity" else "npv")
        flattened = {key: {f"{ep}_{name}": value["metrics"][name]
                           for ep, value in endpoints.items() for name in selected}
                     for key, endpoints in result.items()}
        panels = [(f"{ep}_{name}", f"{ep} {name.upper()}") for ep in ENDPOINTS for name in selected]
        plots.append((destination, flattened, panels, f"{cohort}: {rule} target {target:.0%}"))

    total = sum(len(metrics) for _, _, metrics, _ in plots)
    index = 0
    if progress:
        progress(f"Writing SVGs: 0/{total}")
    for destination, results, metrics, title in plots:
        for metric, label in metrics:
            interval_plot(results, metric, destination / f"{metric}.svg", title=f"{title} · {label}", xlabel=label)
            index += 1
            if progress:
                progress(f"Writing SVGs: {index}/{total} ({metric})")


def run_comparison(layout, key, *, modality, load_development, draws=5000, seed=42, progress=None):
    identifier(key)
    sources = discover(layout, modality, progress=progress)
    release = sources[0][3]["details"]["data_release"]
    mismatched = [f"{model}/{candidate}" for model, candidate, _, manifest in sources
                  if manifest["details"]["data_release"] != release]
    require(not mismatched, "ASSESSMENT_DATA_MISMATCH", f"Data release differs: {', '.join(mismatched)}")
    if progress:
        progress("Loading frozen development data and assessment predictions")
    frozen = ReleasedData.load(layout, release)
    samples, cv = load_development(frozen)
    rows, scores, points = load_predictions(frozen, samples, cv, sources, modality=modality)
    inputs = [{"manifest": manifest["self"], "generation": manifest["generation"]} for _, _, _, manifest in sources]
    snapshot = {"id": key, "data_release": release,
                "parameters": {"assessment_ids": [f"{model}/{candidate}" for model, candidate, _, _ in sources],
                               "bootstrap_draws": draws, "bootstrap_seed": seed}}
    store = experiment_store(layout, "assessment", modality, "comparison", key)
    ref = store.manifest_ref("comparison", "run")
    existing = completed_result(store, ref, snapshot, inputs)
    if existing:
        return existing

    def input_progress(stage, index):
        if progress:
            progress(f"{stage}: {index}/{len(inputs)}")

    input_progress("Binding assessments", 0)
    with store.publish("comparison", "run", inputs=[item["manifest"] for item in inputs],
                       input_progress=input_progress,
                       details={"data_release": release, "assessment_ids": snapshot["parameters"]["assessment_ids"]}) as work:
        free, based, targets = calculate_comparison(rows, scores, points, draws, seed, progress=progress)
        write_comparison(work, rows, free, based, targets, snapshot, progress=progress)
        input_progress("Rechecking assessments", 0)
    return {"id": key, "status": "complete", "manifest": ref}
