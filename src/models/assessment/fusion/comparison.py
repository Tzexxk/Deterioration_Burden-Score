"""Publish paired development comparisons within each fusion training protocol."""

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.bootstrap import interval, patient_weights
from src.models.assessment.comparison import BASELINE, RULES, calculate_comparison, load_predictions, regression_metrics, write_comparison
from src.models.assessment.evaluation import metric_pair
from src.models.assessment.fusion.learned_late.assessment import fixed_name
from src.models.assessment.operating_points import fixed_alerts
from src.models.ehr_data import load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData
from src.models.train.fusion.feature_fusion.training import output


def binding(source):
    manifest = source[3]
    return {"manifest": manifest["self"], "generation": manifest["generation"]}


def select_assessments(assessment_ids):
    chosen = {role: [] for role in ("ehr", "image", "weighted_late", "feature_fusion", "learned_late")}
    seen = set()
    for name in assessment_ids:
        require(isinstance(name, str) and len(name.split("/")) == 3,
                "INVALID_ASSESSMENT_SELECTION", "Use FAMILY/MODEL/ID for each assessment")
        family, model, key = map(identifier, name.split("/"))
        require(family in ("ehr", "image") and model != "comparison"
                or family == "fusion" and model in ("weighted_late", "feature_fusion", "learned_late"),
                "INVALID_ASSESSMENT_SELECTION", "Select EHR, image, F1, F2 or F3 assessments")
        require(name not in seen, "DUPLICATE_ASSESSMENT", f"Assessment selected twice: {name}")
        seen.add(name)
        chosen[family if family != "fusion" else model].append((family, model, key))
    require(chosen["weighted_late"] and all(len(chosen[k]) == 1 for k in ("ehr", "image", "feature_fusion", "learned_late")),
            "INVALID_FUSION_COMPARE", "Select one EHR, one image, one F2, one F3 and at least one F1 assessment")
    return {role: sorted(items) for role, items in chosen.items()}


def load_groups(layout, assessment_ids, progress=None):
    selected = select_assessments(assessment_ids)

    def load(family, model, key):
        store = experiment_store(layout, "assessment", family, model, key)
        manifest = store.validate(store.manifest_ref("assessment", "run"))
        require(manifest["scope"] == store.scope, "INVALID_ASSESSMENT_SOURCE", "Assessment scope differs")
        if progress:
            progress(f"Validated {family}/{model}/{key}")
        return model, key, store, manifest

    selected_sources = {role: [load(*item) for item in items] for role, items in selected.items()}
    sources = [item for items in selected_sources.values() for item in items]
    release = sources[0][3]["details"]["data_release"]
    require(all(item[3]["details"]["data_release"] == release for item in sources),
            "ASSESSMENT_DATA_MISMATCH", "Assessments must use the same frozen release")
    package = load_package(ReleasedData.load(layout, release))

    def single(modality, item):
        rows, values, points = load_predictions(package.release, package.samples, package.cv, [item], modality=modality)
        key = f"{item[0]}/{item[1]}"
        return rows, values, points, key

    for modality in ("ehr", "image"):
        item = selected_sources[modality][0]
        cohort, values, points, name = single(modality, item)
        if modality == "ehr":
            rows, historical, history_points = cohort, {BASELINE: values[BASELINE]}, {BASELINE: points[BASELINE]}
        require(rows.equals(cohort) and np.array_equal(historical[BASELINE], values[BASELINE])
                and history_points[BASELINE] == points[BASELINE], "INVALID_OOF_COVERAGE", "Base cohort or baseline differs")
        historical[modality], history_points[modality] = values[name], points[name]
    base_inputs = [binding(selected_sources[k][0]) for k in ("ehr", "image")]
    fold_lookup = {sample: f["fold"] for f in package.cv["folds"]
                   for sample in package.samples.iloc[f["validation_indices"]].sample_id}
    rows = rows.assign(fold=rows.sample_id.map(fold_lookup))
    candidates, margins = {}, None

    def published(item):
        _, _, store, manifest = item
        actual = pd.read_parquet(output(store, manifest, "oof_predictions.parquet")).sort_values("sample_id").reset_index(drop=True)
        require(all(np.array_equal(actual[k], rows[k]) for k in rows)
                and np.array_equal(actual.baseline_mean_score, historical[BASELINE]),
                "INVALID_OOF_COVERAGE", "Fusion rows differ from the frozen development cohort")
        return actual, read_json(output(store, manifest, "operating_points.json"))

    def add(scores, points, name, values, stored, point_key):
        require(name not in scores and np.isfinite(values).all(), "INVALID_OOF_PREDICTIONS", "Duplicate or non-finite scores")
        require({rule: stored[rule][BASELINE] for rule in RULES} == points[BASELINE],
                "OPERATING_POINT_MISMATCH", "Baseline operating points differ")
        scores[name], points[name] = values, {rule: stored[rule][point_key] for rule in RULES}

    for item in selected_sources["weighted_late"]:
        _, key, store, manifest = item
        require(manifest["inputs"] == base_inputs, "ASSESSMENT_DATA_MISMATCH", "F1 must bind the selected base assessments")
        actual, stored = published(item)
        params = read_json(output(store, manifest, "effective_configuration.json"))["parameters"]
        alpha = params["image_weight"]
        name, shrink = f"F1/{key}", f"EHR_shrink/{key}"
        require(np.array_equal(actual.ehr_predicted_score, historical["ehr"])
                and np.array_equal(actual.image_predicted_score, historical["image"])
                and np.array_equal(actual.predicted_score, (1-alpha)*historical["ehr"] + alpha*historical["image"])
                and np.array_equal(actual.ehr_shrink_score, (1-alpha)*historical["ehr"] + alpha*historical[BASELINE]),
                "INVALID_OOF_PREDICTIONS", "F1 scores do not replay from the selected base predictions")
        require(margins is None or margins == params["decision_rules"],
                "INVALID_FUSION_COMPARE", "F1 decision margins differ")
        margins = params["decision_rules"]
        add(historical, history_points, name, actual.predicted_score.to_numpy(float), stored, "model")
        add(historical, history_points, shrink, actual.ehr_shrink_score.to_numpy(float), stored, "ehr_shrink")
        candidates[name] = shrink

    def trained(model):
        item = selected_sources[model][0]
        manifest = item[3]
        cohort, _, points, _ = single("fusion", item)
        acceptance = package.release.acceptance
        train = read_json(layout.resolve(manifest["inputs"][0]["manifest"]))
        require(cohort.equals(rows.drop(columns="fold")) and train["inputs"] == [
            {"manifest": acceptance["self"], "generation": acceptance["generation"]}, *base_inputs],
            "ASSESSMENT_DATA_MISMATCH", "Fusion training must bind the same base sources and frozen release")
        return item, points[BASELINE]

    f2 = trained("feature_fusion")[0]
    actual, stored = published(f2)
    name = f"F2/{f2[1]}"
    add(historical, history_points, name, actual.predicted_score.to_numpy(float), stored, "model")
    candidates[name] = "ehr"

    f3, baseline_points = trained("learned_late")
    actual, stored = published(f3)
    _, _, store, manifest = f3
    params = read_json(output(store, manifest, "effective_configuration.json"))["parameters"]
    nested = {BASELINE: actual.baseline_mean_score.to_numpy(float)}
    nested_points = {BASELINE: baseline_points}
    require(np.array_equal(actual.predicted_score, actual.f3b) and np.array_equal(actual.baseline_mean, nested[BASELINE]),
            "INVALID_OOF_PREDICTIONS", "F3 primary or baseline score differs")
    for weight in params["fixed_image_weights"]:
        name = fixed_name(weight)
        require(np.array_equal(actual[name], (1-weight)*actual.ehr + weight*actual.image),
                "INVALID_OOF_PREDICTIONS", "Nested fixed-weight control differs")
        add(nested, nested_points, name, actual[name].to_numpy(float), stored, name)
    for name in ("ehr", "image", "ehr_bias", "f3a", "f3b"):
        add(nested, nested_points, name, actual[name].to_numpy(float), stored, "model" if name == "f3b" else name)
    return rows, {"historical": (historical, history_points, candidates),
                  "nested": (nested, nested_points, {"f3a": "ehr", "f3b": "ehr_bias"})}, sources, margins


def paired_errors(rows, scores, weights):
    y = rows.burden_score.to_numpy(float)
    output = {}
    for mode in ("raw", "postprocessed"):
        stats = {name: regression_metrics(y, v if mode == "raw" else np.clip(v, 0, 8), weights) for name, v in scores.items()}
        pairs = {}
        for name in scores:
            pairs[name] = {}
            for reference in scores:
                if name != reference:
                    delta = {m: stats[name][m] - stats[reference][m] for m in ("rmse", "mae")}
                    pairs[name][reference] = {m: interval(v[0], v[1:]) for m, v in delta.items()}
        output[mode] = pairs
    return output


def diagnostics(rows, scores, points, based, weights, candidates, margins, paired):
    labels = rows.burden_score.to_numpy(float)
    strata = {"zero": labels == 0, "high_burden": rows.high_burden.to_numpy(bool),
              **{f"fold_{int(n)}": rows.fold.to_numpy() == n for n in sorted(rows.fold.unique())}}
    subgroups = {}
    for key, mask in strata.items():
        selected = rows.loc[mask]
        subgroups[key] = {"studies": len(selected), "patients": int(selected.subject_id.nunique()),
                          "metrics": {k: metric_pair(selected, v[mask]) for k, v in scores.items()} if len(selected) else {}}
    for rule in RULES:
        for ep in ENDPOINTS:
            positive = rows[ep].to_numpy(bool)
            ehr_alerts = fixed_alerts(positive, scores["ehr"], points["ehr"][rule][ep]["threshold"])
            for name, values in scores.items():
                point = points[name][rule][ep]
                alerts = fixed_alerts(positive, values, point["threshold"])
                rates = (weights @ alerts) / weights.sum(axis=1)
                entry = based[rule][name][ep]
                entry["confusion_matrix"] = point["confusion_matrix"]
                entry["alert_rate"] = interval(rates[0], rates[1:])
                entry["corrected_ehr_misses"] = int((positive & ~ehr_alerts & alerts).sum())
                entry["new_misses_vs_ehr"] = int((positive & ehr_alerts & ~alerts).sum())
    decisions = {}
    for name, control in candidates.items():
        refs = paired["raw"][name]
        by_patient = pd.DataFrame({"n": 1, "sse": (scores[name]-labels)**2, "ehr_sse": (scores["ehr"]-labels)**2}).groupby(rows.subject_id).sum()
        total = by_patient.sum()
        loo = np.sqrt((total.sse-by_patient.sse)/(total.n-by_patient.n)) - np.sqrt((total.ehr_sse-by_patient.ehr_sse)/(total.n-by_patient.n))
        decisions[name] = {"ehr_point_margins_met": refs["ehr"]["rmse"]["point"] <= -margins["min_rmse_improvement_ehr"]
                            and refs["ehr"]["mae"]["point"] <= margins["max_mae_increase_ehr"],
                           "image_rmse_point_margin_met": refs["image"]["rmse"]["point"] <= -margins["min_rmse_improvement_image"],
                           "additional_control": control, "control_rmse_delta": refs[control]["rmse"],
                           "ehr_rmse_delta": refs["ehr"]["rmse"], "image_rmse_delta": refs["image"]["rmse"],
                           "leave_one_patient_out_ehr_rmse_range": [float(loo.min()), float(loo.max())],
                           "leave_one_patient_out_ehr_rmse_margin_met": int((loo <= -margins["min_rmse_improvement_ehr"]).sum())}
    return subgroups, decisions


def run_comparison(layout, key, *, assessment_ids, draws=5000, seed=42, progress=None):
    identifier(key)
    require(type(draws) is int and draws > 0 and type(seed) is int and seed >= 0,
            "INVALID_FUSION_COMPARE", "Invalid bootstrap settings")
    rows, groups, sources, margins = load_groups(layout, assessment_ids, progress)
    inputs = [binding(s) for s in sources]
    release = sources[0][3]["details"]["data_release"]
    snapshot = {"id": key, "data_release": release,
                "parameters": {"assessment_ids": sorted(assessment_ids), "bootstrap_draws": draws, "bootstrap_seed": seed}}
    store = experiment_store(layout, "assessment", "fusion", "comparison", key)
    ref = store.manifest_ref("comparison", "run")
    existing = completed_result(store, ref, snapshot, inputs)
    if existing:
        return existing
    with store.publish("comparison", "run", inputs=[x["manifest"] for x in inputs],
                       details={"data_release": release, "protocols": list(groups)}) as work:
        require(all(store.validate(x["manifest"])["generation"] == x["generation"] for x in inputs),
                "UPSTREAM_REBUILT", "An assessment changed after its scores were loaded")
        weights = np.vstack([np.ones(len(rows)), patient_weights(rows.subject_id.to_numpy(), draws, seed)])
        write_json(work / "effective_configuration.json", snapshot)
        write_json(work / "summary.json", {"studies": len(rows), "patients": int(rows.subject_id.nunique()),
            "decision_margins": margins, "protocols": {k: list(v[0]) for k, v in groups.items()},
            "prior_fixed_test_exposure": "Both source test results were viewed and used for source selection."})
        for name, (scores, points, candidates) in groups.items():
            if progress:
                progress(f"Comparing {name} protocol")
            folder = work / name
            free, based, targets = calculate_comparison(rows, scores, points, draws, seed, progress=progress)
            paired = paired_errors(rows, scores, weights)
            strata, decisions = diagnostics(rows, scores, points, based, weights, candidates, margins, paired)
            write_comparison(folder, rows, free, based, targets, snapshot, progress=progress, cohort=f"{name} OOF")
            write_json(folder / "paired_differences.json", paired)
            write_json(folder / "strata_and_folds.json", strata)
            write_json(folder / "decision_summary.json", decisions)
    return {"id": key, "status": "complete", "manifest": ref}
