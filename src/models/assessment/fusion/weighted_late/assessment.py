"""Fixed development-score fusion with explicit, generation-bound sources."""

import numpy as np

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.bootstrap import interval, patient_weights
from src.models.assessment.comparison import load_predictions, regression_metrics
from src.models.assessment.evaluation import metric_pair
from src.models.assessment.operating_points import operating_point, rule_folder
from src.models.assessment.plots import confusion_plot
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import (
    completed_result,
    experiment_store,
    read_batch,
    run_model_batch,
)
from src.models.image_data import load_image_cohort, load_image_package
from src.models.released_data import ReleasedData

MODEL = "weighted_late"


def validate_parameters(params):
    require(set(params) == {"ehr_source", "image_source", "image_weight", "sensitivity_target",
                            "specificity_target", "bootstrap_draws", "bootstrap_seed", "decision_rules"},
            "INVALID_FUSION_CONFIG", "Specify both assessment sources, weight, targets, bootstrap and decision rules")
    for modality in ("ehr", "image"):
        source = params[f"{modality}_source"]
        require(isinstance(source, dict) and set(source) == {"model", "assessment_id"},
                "INVALID_FUSION_CONFIG", "Each source requires model and assessment_id")
        for value in source.values():
            identifier(value)
    require(type(params["image_weight"]) in (int, float) and 0 <= params["image_weight"] <= 1,
            "INVALID_FUSION_CONFIG", "image_weight must be finite and between zero and one")
    for rule in ("sensitivity", "specificity"):
        require(type(params[f"{rule}_target"]) in (int, float) and 0 < params[f"{rule}_target"] < 1,
                "INVALID_FUSION_CONFIG", "Operating-point targets must be between zero and one")
    require(type(params["bootstrap_draws"]) is int and params["bootstrap_draws"] > 0
            and type(params["bootstrap_seed"]) is int and params["bootstrap_seed"] >= 0,
            "INVALID_FUSION_CONFIG", "Use positive bootstrap draws and a nonnegative seed")
    rules = params["decision_rules"]
    require(isinstance(rules, dict) and set(rules) == {
        "min_rmse_improvement_ehr", "max_mae_increase_ehr", "min_rmse_improvement_image"}
        and all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in rules.values()),
        "INVALID_FUSION_CONFIG", "Declare nonnegative, finite decision margins before running")


def load_sources(layout, release, samples, cv, params):
    scores, records, manifests = {}, {}, []
    reference = None
    for modality in ("ehr", "image"):
        choice = params[f"{modality}_source"]
        model, key = choice["model"], choice["assessment_id"]
        require(model != "comparison", "INVALID_FUSION_SOURCE", "Bind a basic assessment")
        store = experiment_store(layout, "assessment", modality, model, key)
        manifest = store.validate(store.manifest_ref("assessment", "run"))
        require(manifest["scope"] == store.scope, "INVALID_FUSION_SOURCE", "Assessment scope differs")
        rows, values, points = load_predictions(release, samples, cv, [(model, key, store, manifest)], modality=modality)
        baseline_points = points["baseline_mean"]
        require(all(points[f"{model}/{key}"][rule][endpoint]["target"] == params[f"{rule}_target"]
                    for rule in ("sensitivity", "specificity") for endpoint in ENDPOINTS),
                "OPERATING_POINT_MISMATCH", "Source targets must match the fusion targets")
        if reference is None:
            reference = rows
            scores["baseline_mean"] = values["baseline_mean"]
            reference_points = baseline_points
        else:
            require(rows.equals(reference) and np.array_equal(values["baseline_mean"], scores["baseline_mean"])
                    and baseline_points == reference_points,
                    "INVALID_OOF_COVERAGE", "The two development cohorts or baselines differ")
        scores[modality] = values[f"{model}/{key}"]
        train_store = experiment_store(layout, "train", modality, model, manifest["details"]["train_id"])
        train = read_json(layout.resolve(train_store.manifest_ref("train", "run")))
        snapshot = read_json(layout.resolve(train_store.output_ref(train, "effective_configuration.json")))
        require(snapshot["id"] == manifest["details"]["train_id"]
                and snapshot["data_release"] == release.store.data.key,
                "INVALID_FUSION_SOURCE", "Training snapshot identity differs")
        if modality == "image":
            package = load_image_package(release, snapshot["parameters"]["image_source"])
            require(snapshot["resolved_image_source"] == package.source
                    and snapshot["image_shape"] == list(package.shape) == train["details"]["image_shape"],
                    "INVALID_FUSION_SOURCE", "Image release, variant or dimensions differ from training")
        records[modality] = {"assessment": {"manifest": manifest["self"], "generation": manifest["generation"]},
                             "training": {"manifest": train["self"], "generation": train["generation"]},
                             "effective_training_configuration": snapshot}
        manifests.append(manifest)
    return reference, scores, records, manifests


def paired_differences(rows, scores, draws, seed):
    weights = np.vstack([np.ones(len(rows)), patient_weights(rows.subject_id.to_numpy(), draws, seed)])
    labels = rows.burden_score.to_numpy(float)
    result = {}
    for mode in ("raw", "postprocessed"):
        metrics = {key: regression_metrics(labels, values if mode == "raw" else np.clip(values, 0, 8), weights)
                   for key, values in scores.items()}
        result[mode] = {}
        for reference in ("ehr", "image", "baseline_mean", "ehr_shrink"):
            result[mode][reference] = {}
            for metric in ("rmse", "mae"):
                delta = metrics["model"][metric] - metrics[reference][metric]
                result[mode][reference][metric] = interval(delta[0], delta[1:])
    return {"difference": "fusion minus reference", "bootstrap_unit": "subject_id",
            "confidence_level": 0.95, "bootstrap_draws": draws, "bootstrap_seed": seed,
            "confidence_interval": "percentile", "results": result}


def publish_assessment(layout, config_path, experiment, release, samples, cv):
    params = experiment["parameters"]
    rows, scores, records, sources = load_sources(layout, release, samples, cv, params)
    store = experiment_store(layout, "assessment", "fusion", MODEL, experiment["id"])
    ref = store.manifest_ref("assessment", "run")
    inputs = [{"manifest": m["self"], "generation": m["generation"]} for m in sources]
    existing = completed_result(store, ref, experiment, inputs)
    if existing:
        return existing
    label = f"{MODEL}/{experiment['id']}"
    print(f"{label}: pairing development OOF and calculating fixed scores", flush=True)
    alpha = params["image_weight"]
    scores["model"] = (1 - alpha) * scores["ehr"] + alpha * scores["image"]
    scores["ehr_shrink"] = (1 - alpha) * scores["ehr"] + alpha * scores["baseline_mean"]
    folds, means = {}, {}
    for fold in cv["folds"]:
        folds.update(dict.fromkeys(samples.iloc[fold["validation_indices"]].sample_id, fold["fold"]))
        means[fold["fold"]] = float(samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())
    predictions = rows.assign(fold=rows.sample_id.map(folds), predicted_score=scores["model"],
                              postprocessed_score=np.clip(scores["model"], 0, 8),
                              baseline_mean_score=scores["baseline_mean"], ehr_predicted_score=scores["ehr"],
                              image_predicted_score=scores["image"], ehr_shrink_score=scores["ehr_shrink"])
    def metrics(mask):
        selected = predictions.loc[mask]
        return {"studies": len(selected), "patients": int(selected.subject_id.nunique()),
                **{MODEL if key == "model" else key: metric_pair(selected, values[mask])
                   for key, values in scores.items()}}
    pooled = metrics(np.ones(len(rows), dtype=bool))
    per_fold = [{"fold": n, "train_label_mean": means[n], **metrics(predictions.fold.to_numpy() == n)}
                for n in sorted(means)]
    with store.publish("assessment", "run", inputs=[m["self"] for m in sources],
                       configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "source_assessments": inputs,
                                "prediction_role": "oof_validation", "image_weight": alpha}) as work:
        require(all(store.validate(item["manifest"])["generation"] == item["generation"] for item in inputs),
                "UPSTREAM_REBUILT", "An assessment changed after its predictions were loaded")
        write_json(work / "effective_configuration.json", experiment)
        write_json(work / "sources.json", records)
        predictions.to_parquet(work / "oof_predictions.parquet", index=False)
        write_json(work / "metrics.json", {"prediction_role": "oof_validation", "folds": per_fold, "pooled_oof": pooled})
        points = {}
        for rule in ("sensitivity", "specificity"):
            target = params[f"{rule}_target"]
            points[rule] = {key: {ep: operating_point(rows[ep].to_numpy(bool), values, rule, target)
                                  for ep in ENDPOINTS} for key, values in scores.items()}
            plot = work / "threshold_based" / rule_folder(rule, target) / "confusion_matrices.svg"
            plot.parent.mkdir(parents=True)
            confusion_plot(points[rule], plot, title=f"{label}: {rule} >= {target:.0%}")
        write_json(work / "operating_points.json", points)
        print(f"{label}: paired patient bootstrap ({params['bootstrap_draws']:,} draws)", flush=True)
        differences = paired_differences(rows, scores, params["bootstrap_draws"], params["bootstrap_seed"])
        write_json(work / "paired_differences.json", differences)
        raw = differences["results"]["raw"]
        margins = params["decision_rules"]
        write_json(work / "decision_summary.json", {
            "role": "exploratory_development_review", "decision_rules": margins,
            "ehr_point_margins_met": raw["ehr"]["rmse"]["point"] <= -margins["min_rmse_improvement_ehr"]
                and raw["ehr"]["mae"]["point"] <= margins["max_mae_increase_ehr"],
            "image_rmse_point_margin_met": raw["image"]["rmse"]["point"] <= -margins["min_rmse_improvement_image"],
            "rmse_interval_upper_below_zero": {k: v["rmse"]["upper"] < 0 for k, v in raw.items()},
            "limitations": ["Development OOF participated in early stopping and candidate selection.",
                            "Both fixed-test results were viewed and used to select the fusion sources.",
                            "These margins are research criteria, not established clinical importance.",
                            "Review patient influence and both mandatory experiments before selection."]})
    return {"id": experiment["id"], "status": "complete", "manifest": ref}


def run_assessment(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment["parameters"])
    release = ReleasedData.load(layout, experiments[0]["data_release"])
    _, samples, cv = load_image_cohort(release)
    return run_model_batch(layout, "assessment", "fusion", MODEL, experiments,
                           lambda e: publish_assessment(layout, config_path, e, release, samples, cv))
