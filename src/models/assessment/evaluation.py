"""Shared model assessment publication and OOF validation."""

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import identifier, require
from src.models.assessment.operating_points import operating_point, rule_folder
from src.models.assessment.plots import confusion_plot
from src.models.ehr_metrics import ENDPOINTS, point_metrics
from src.models.experiments import completed_result, experiment_store


def validate_parameters(experiment):
    params = experiment["parameters"]
    require(set(params) == {"train_id", "sensitivity_target", "specificity_target", "plot_training_history"},
            "INVALID_ASSESSMENT_CONFIG", "Specify train_id, both operating-point targets and plot_training_history")
    identifier(params["train_id"])
    require(all(type(params[key]) in (int, float) and 0 < params[key] < 1
                for key in ("sensitivity_target", "specificity_target")),
            "INVALID_ASSESSMENT_CONFIG", "Sensitivity and specificity targets must be between zero and one")
    require(type(params["plot_training_history"]) is bool,
            "INVALID_ASSESSMENT_CONFIG", "plot_training_history must be a boolean")


def training_source(layout, release, train_id, model, *, modality):
    store = experiment_store(layout, "train", modality, model, train_id)
    ref = store.manifest_ref("train", "run")
    require(layout.resolve(ref).is_file(), "TRAINING_NOT_COMPLETE", "Assessment requires a published training ID")
    manifest = store.validate(ref)
    acceptance = release.acceptance
    require(manifest["scope"] == store.scope and manifest["inputs"] == [
        {"manifest": acceptance["self"], "generation": acceptance["generation"]}
    ], "ASSESSMENT_DATA_MISMATCH", "Training must bind the selected frozen dataset")
    return store, manifest


def metric_pair(rows, scores):
    return {"raw": point_metrics(rows, scores), "postprocessed": point_metrics(rows, np.clip(scores, 0, 8))}


def evaluate_folds(store, manifest, package, model, predict):
    frames, fold_metrics, histories = [], [], []
    columns = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", "phys_death", "high_burden"]
    for fold in package.cv["folds"]:
        raw, history = predict(store, manifest, package, fold)
        rows = package.samples.iloc[fold["validation_indices"]][columns].copy()
        mean = float(package.samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())
        baseline = np.full(len(rows), mean)
        rows = rows.assign(fold=fold["fold"], predicted_score=raw, postprocessed_score=np.clip(raw, 0, 8),
                           baseline_mean_score=baseline)
        fold_metrics.append({"fold": fold["fold"], "studies": len(rows), "patients": int(rows.subject_id.nunique()),
                             "train_label_mean": mean, model: metric_pair(rows, raw),
                             "baseline_mean": metric_pair(rows, baseline)})
        frames.append(rows)
        if history is not None:
            histories.append(history)
    predictions = pd.concat(frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    development = package.samples.iloc[package.cv["development_indices"]].sample_id.tolist()
    require(predictions.sample_id.is_unique and predictions.sample_id.tolist() == sorted(development),
            "INVALID_OOF_COVERAGE", "OOF predictions must cover the development set exactly once")
    raw, baseline = predictions.predicted_score.to_numpy(float), predictions.baseline_mean_score.to_numpy(float)
    pooled = {"studies": len(predictions), "patients": int(predictions.subject_id.nunique()),
              model: metric_pair(predictions, raw), "baseline_mean": metric_pair(predictions, baseline)}
    return predictions, {"prediction_role": "oof_validation", "folds": fold_metrics, "pooled_oof": pooled}, histories


def publish_assessment(layout, config_path, experiment, package, model, predict, plot_history, *, modality, source):
    params = experiment["parameters"]
    train_store, train = source
    upstream = {"manifest": train["self"], "generation": train["generation"]}
    key = experiment["id"]
    store = experiment_store(layout, "assessment", modality, model, key)
    ref = store.manifest_ref("assessment", "run")
    existing = completed_result(store, ref, experiment, [upstream])
    if existing:
        return existing
    with store.publish("assessment", "run", inputs=[train["self"]], configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "train_id": params["train_id"]}) as work:
        predictions, metrics, histories = evaluate_folds(train_store, train, package, model, predict)
        write_json(work / "effective_configuration.json", experiment)
        write_json(work / "metrics.json", metrics)
        predictions.to_parquet(work / "oof_predictions.parquet", index=False)
        points = {}
        for rule in ("sensitivity", "specificity"):
            target = params[f"{rule}_target"]
            scores = {key: predictions[column].to_numpy(float) for key, column in
                      (("model", "predicted_score"), ("baseline_mean", "baseline_mean_score"))}
            points[rule] = {key: {endpoint: operating_point(predictions[endpoint].to_numpy(bool), values, rule, target)
                                  for endpoint in ENDPOINTS} for key, values in scores.items()}
            plot = work / "threshold_based" / rule_folder(rule, target) / "confusion_matrices.svg"
            plot.parent.mkdir(parents=True)
            confusion_plot(points[rule], plot, title=f"{experiment['id']}: {rule} ≥ {target:.0%}")
        write_json(work / "operating_points.json", points)
        if params["plot_training_history"] and histories and plot_history is not None:
            plot_history(histories, work / "training_history.svg")
    return {"id": key, "status": "complete", "manifest": ref}
