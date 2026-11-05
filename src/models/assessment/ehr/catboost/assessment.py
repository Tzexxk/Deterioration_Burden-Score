"""Assess frozen CatBoost checkpoints on their own development validation folds."""

from importlib.metadata import version

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.operating_points import operating_point, rule_folder
from src.models.assessment.plots import confusion_plot
from src.models.ehr_data import checked_input_summary, load_package
from src.models.ehr_metrics import ENDPOINTS, point_metrics
from src.models.experiments import (
    completed_result,
    experiment_store,
    read_batch,
    run_model_batch,
)
from src.models.released_data import ReleasedData


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


def training_source(layout, package, train_id):
    store = experiment_store(layout, "train", "ehr", "catboost", train_id)
    ref = store.manifest_ref("train", "run")
    require(layout.resolve(ref).is_file(), "TRAINING_NOT_COMPLETE", "Assessment requires a published training ID")
    manifest = store.validate(ref)
    acceptance = package.release.acceptance
    require(manifest["scope"] == store.scope and manifest["inputs"] == [
        {"manifest": acceptance["self"], "generation": acceptance["generation"]}
    ], "ASSESSMENT_DATA_MISMATCH", "Training must bind the selected frozen dataset")
    return store, manifest


def checkpoint_prediction(store, manifest, package, fold):
    from catboost import CatBoostRegressor

    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    interface = read_json(output("interface.json"))
    train = package.samples.iloc[fold["train_indices"]]
    validation = package.samples.iloc[fold["validation_indices"]]
    categorical = package.schema["categorical_features"]
    features = list(package.features.columns)
    require(interface["fold"] == fold["fold"] and interface["catboost_version"] == version("catboost")
            and interface["features"] == features and interface["categorical_features"] == categorical
            and interface["train_sample_ids"] == train.sample_id.tolist()
            and interface["validation_sample_ids"] == validation.sample_id.tolist(),
            "INVALID_CHECKPOINT", "Checkpoint input, CatBoost version or fold identity differs")
    model = CatBoostRegressor()
    model.load_model(str(output("model.cbm")), format="cbm")
    require(model.feature_names_ == features and model.tree_count_ == interface["tree_count"]
            and model.get_cat_feature_indices() == [features.index(name) for name in categorical],
            "INVALID_CHECKPOINT", "Saved CatBoost model differs from its declared interface")
    history = read_json(output("history.json"))
    learn, valid = (np.asarray(history[part]["RMSE"], dtype=float) for part in ("learn", "validation"))
    best_round = interface["best_round"]
    require(learn.shape == valid.shape and learn.ndim == 1 and learn.size > 0
            and np.isfinite(learn).all() and np.isfinite(valid).all()
            and best_round == int(np.argmin(valid)) + 1 == interface["tree_count"],
            "INVALID_TRAINING_HISTORY", "RMSE history does not match the checkpoint's best round")
    raw = model.predict(package.features.iloc[fold["validation_indices"]])
    return raw, {"fold": fold["fold"], "best_round": best_round, "learn": learn, "validation": valid}


def metric_pair(rows, scores):
    return {"raw": point_metrics(rows, scores), "postprocessed": point_metrics(rows, np.clip(scores, 0, 8))}


def assess_folds(store, manifest, package):
    frames, fold_metrics, histories = [], [], []
    columns = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", "phys_death", "high_burden"]
    for fold in package.cv["folds"]:
        raw, history = checkpoint_prediction(store, manifest, package, fold)
        rows = package.samples.iloc[fold["validation_indices"]][columns].copy()
        mean = float(package.samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())
        baseline = np.full(len(rows), mean)
        rows = rows.assign(fold=fold["fold"], predicted_score=raw, postprocessed_score=np.clip(raw, 0, 8),
                           baseline_mean_score=baseline)
        fold_metrics.append({"fold": fold["fold"], "studies": len(rows), "patients": int(rows.subject_id.nunique()),
                             "train_label_mean": mean, "catboost": metric_pair(rows, raw),
                             "baseline_mean": metric_pair(rows, baseline)})
        frames.append(rows)
        histories.append(history)
    predictions = pd.concat(frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    development = package.samples.iloc[package.cv["development_indices"]].sample_id.tolist()
    require(predictions.sample_id.is_unique and predictions.sample_id.tolist() == sorted(development),
            "INVALID_OOF_COVERAGE", "OOF predictions must cover the development set exactly once")
    raw, baseline = predictions.predicted_score.to_numpy(float), predictions.baseline_mean_score.to_numpy(float)
    pooled = {"studies": len(predictions), "patients": int(predictions.subject_id.nunique()),
              "catboost": metric_pair(predictions, raw), "baseline_mean": metric_pair(predictions, baseline)}
    return predictions, {"prediction_role": "oof_validation", "folds": fold_metrics, "pooled_oof": pooled}, histories


def plot_history(histories, path):
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 8), layout="constrained")
    axes = figure.subplots(3, 2).ravel()
    for axis, history in zip(axes, histories):
        rounds = np.arange(1, len(history["learn"]) + 1)
        axis.plot(rounds, history["learn"], label="Train")
        axis.plot(rounds, history["validation"], label="Validation")
        axis.axvline(history["best_round"], color="gray", linestyle="--", label="Best round")
        axis.set(title=f"Fold {history['fold']}", xlabel="Training round", ylabel="RMSE")
        axis.legend()
    for axis in axes[len(histories):]:
        axis.set_visible(False)
    FigureCanvasSVG(figure).print_svg(path)


def publish_assessment(layout, config_path, experiment, package):
    params = experiment["parameters"]
    train_store, train = training_source(layout, package, params["train_id"])
    upstream = {"manifest": train["self"], "generation": train["generation"]}
    key = experiment["id"]
    store = experiment_store(layout, "assessment", "ehr", "catboost", key)
    ref = store.manifest_ref("assessment", "run")
    existing = completed_result(store, ref, experiment, [upstream])
    if existing:
        return existing
    with store.publish("assessment", "run", inputs=[train["self"]], configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "train_id": params["train_id"]}) as work:
        predictions, metrics, histories = assess_folds(train_store, train, package)
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
        if params["plot_training_history"]:
            plot_history(histories, work / "training_history.svg")
    return {"id": key, "status": "complete", "manifest": ref}


def run_assessment(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment)
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    checked_input_summary(package)
    return run_model_batch(layout, "assessment", "ehr", "catboost", experiments,
                           lambda experiment: publish_assessment(layout, config_path, experiment, package))
