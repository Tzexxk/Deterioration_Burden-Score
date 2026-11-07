"""Fit an outer-fold affine map from patient-disjoint inner predictions."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.train.image.resnet50.training import fit_fold, load_weights, validate_settings


def fit_affine(scores, labels):
    centered = scores - scores.mean()
    variance = float(centered @ centered)
    slope = float(centered @ (labels - labels.mean()) / variance) if variance > 0 else 0.0
    if not np.isfinite(slope) or slope <= 0:
        return {"intercept": 0.0, "slope": 1.0, "identity_fallback": True}
    return {"intercept": float(labels.mean() - slope * scores.mean()), "slope": slope, "identity_fallback": False}


def inner_predictions(package, outer, config, weights, *, candidate):
    outer_train = outer["train_indices"]
    allowed = set(outer_train)
    parts, splits = [], []
    with TemporaryDirectory() as temporary:
        for partition in package.cv["folds"]:
            if partition["fold"] == outer["fold"]:
                continue
            valid = [index for index in partition["validation_indices"] if index in allowed]
            held_out = set(valid)
            train = [index for index in outer_train if index not in held_out]
            train_rows, valid_rows = package.samples.iloc[train], package.samples.iloc[valid]
            require(train and valid and set(train_rows.subject_id).isdisjoint(valid_rows.subject_id),
                    "INVALID_CALIBRATION_SPLIT", "Inner training and prediction patients must be disjoint")
            inner_id = partition["fold"]
            inner = {"fold": inner_id, "train_indices": train, "validation_indices": valid}
            path = Path(temporary) / f"inner_{inner_id}"
            fit_fold(package, inner, config, weights, path, select_last=True,
                     label=f"{candidate} outer_fold={outer['fold']} inner_fold={inner_id}")
            replay = read_json(path / "interface.json")
            predictions = pd.read_parquet(path / "validation_predictions.parquet")
            require(replay["best_epoch"] == config["max_epochs"]
                    and replay["train_sample_ids"] == train_rows.sample_id.tolist()
                    and replay["validation_sample_ids"] == predictions.sample_id.tolist() == valid_rows.sample_id.tolist(),
                    "INVALID_CALIBRATION_PREDICTIONS", "Inner predictions must come from the fixed final checkpoint")
            splits.append(replay)
            parts.append(predictions.assign(inner_fold=inner_id, burden_score=valid_rows.burden_score.to_numpy()))
    oof = pd.concat(parts, ignore_index=True).sort_values("sample_id")
    require(oof.sample_id.tolist() == sorted(package.samples.iloc[outer_train].sample_id.tolist()),
            "INVALID_CALIBRATION_COVERAGE", "Inner predictions must cover the outer training set exactly once")
    return oof, splits


def calibrated_prediction(store, manifest, package, fold, *, predict, config, weights, folder, candidate):
    scores, history = predict(store, manifest, package, fold)
    work = folder / f"fold_{fold['fold']}"
    work.mkdir(parents=True)
    oof, splits = inner_predictions(package, fold, config, weights, candidate=candidate)
    affine = fit_affine(oof.predicted_score.to_numpy(float), oof.burden_score.to_numpy(float))
    write_json(work / "calibration.json", {**affine, "inner_epochs": config["max_epochs"], "inner_folds": splits})
    oof.to_parquet(work / "inner_oof_predictions.parquet", index=False)
    return affine["intercept"] + affine["slope"] * scores, history


def prepare_calibration(work, predict, *, layout, config, candidate, epochs):
    require(config.get("event_components") is True and "layer4_lr" in config
            and config["head_only_epochs"] < epochs <= config["max_epochs"],
            "INVALID_CALIBRATION_CONFIG", "Calibration requires an event model and a fixed fine-tuning schedule")
    inner_config = {**config, "max_epochs": epochs}
    inner_config.pop("early_stopping", None)
    validate_settings(inner_config)
    weights = load_weights(layout.resolve(config["weights"]), config)
    return partial(calibrated_prediction, predict=predict, config=inner_config, weights=weights,
                   folder=work / "calibration", candidate=candidate)
