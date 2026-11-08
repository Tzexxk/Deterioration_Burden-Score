"""Learn fusion coefficients inside each outer fold and evaluate held-out scores."""

from itertools import combinations

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.bootstrap import interval, patient_weights
from src.models.assessment.comparison import regression_metrics
from src.models.assessment.evaluation import metric_pair
from src.models.assessment.operating_points import operating_point, rule_folder
from src.models.assessment.plots import confusion_plot
from src.models.ehr_data import checked_input_summary, load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import (
    completed_result,
    experiment_store,
    read_batch,
    run_model_batch,
)
from src.models.released_data import ReleasedData
from src.models.test.ehr.catboost import model as ehr_model
from src.models.train.fusion.feature_fusion.training import output, resolve_sources
from src.models.train.fusion.learned_late.training import (
    IDENTITY,
    MODEL,
    TRAINING_MEDIAN,
    bindings,
    calibration_fold,
    fit_id,
    median_lengths,
    partitions,
    split_rows,
)
from src.models.train.fusion.learned_late.training import (
    validate_parameters as validate_training_parameters,
)


def fit_coefficients(ehr, image, labels):
    d, r = image - ehr, labels - ehr

    def weight(x, y):
        denom = x @ x
        return float(np.clip((x @ y) / denom, 0, 1)) if denom else 0.0

    # Subtract the first value before centering to make a constant difference exactly zero.
    shift = d - d[0]
    alpha = weight(shift - shift.mean(), r - r.mean())
    return {"f3a": {"alpha": weight(d, r), "bias": 0.0},
            "f3b": {"alpha": alpha, "bias": float(r.mean() - alpha * d.mean())},
            "ehr_bias": {"alpha": 0.0, "bias": float(r.mean())}}


def validate_parameters(params):
    require(set(params) == {"train_id", "fixed_image_weights", "sensitivity_target", "specificity_target",
                            "bootstrap_draws", "bootstrap_seed"}, "INVALID_FUSION_CONFIG", "Invalid F3 assessment fields")
    identifier(params["train_id"])
    weights = params["fixed_image_weights"]
    require(isinstance(weights, list) and weights and all(type(w) in (float, int) and 0 < w < 1 for w in weights)
            and len(weights) == len({fixed_name(w) for w in weights}),
            "INVALID_FUSION_CONFIG", "Fixed controls need distinct weights between zero and one")
    require(all(type(params[k]) in (float, int) and 0 < params[k] < 1 for k in ("sensitivity_target", "specificity_target"))
            and type(params["bootstrap_draws"]) is int and params["bootstrap_draws"] > 0
            and type(params["bootstrap_seed"]) is int and params["bootstrap_seed"] >= 0,
            "INVALID_FUSION_CONFIG", "Invalid operating point or bootstrap settings")


def fixed_name(weight):
    return f"fixed_alpha_{weight:.12g}".replace(".", "p")


def checked_lengths(store, manifest, package, folds, trained, prefix, params, meta):
    if params.get("round_policy") != TRAINING_MEDIAN:
        require(meta["selection"] == "fixed_training_lengths", "INVALID_NESTED_FIT", "Expected fixed training lengths")
        return {k: params[k] for k in ("ehr_rounds", "image_epochs")}
    calibration = []
    for held in trained:
        fold = calibration_fold(folds, trained, held)
        folder = f"{prefix}/calibration/fold_{held}"
        row = {"validation_fold": held}
        for modality, field in (("ehr", "best_round"), ("image", "best_epoch")):
            interface = read_json(output(store, manifest, f"{folder}/{modality}/interface.json"))
            history = read_json(output(store, manifest, f"{folder}/{modality}/history.json"))
            if modality == "ehr":
                errors = np.asarray(history["validation"]["RMSE"], dtype=float)
            else:
                epochs = history["epochs"]
                require([e["epoch"] for e in epochs] == list(range(1, len(epochs) + 1)),
                        "INVALID_NESTED_CALIBRATION", "Image calibration epochs are not contiguous")
                errors = np.asarray([e["validation"]["rmse"] for e in epochs], dtype=float)
            require(interface["fold"] == held and errors.ndim == 1 and errors.size > 0 and np.isfinite(errors).all()
                    and interface[field] == int(errors.argmin()) + 1
                    and interface["train_sample_ids"] == package.samples.iloc[fold["train_indices"]].sample_id.tolist()
                    and interface["validation_sample_ids"] == package.samples.iloc[fold["validation_indices"]].sample_id.tolist(),
                    "INVALID_NESTED_CALIBRATION", "Calibration history or sample scope differs from its training range")
            row[f"{modality}_{field}"] = interface[field]
        calibration.append(row)
    require(meta["selection"] == TRAINING_MEDIAN and meta["calibration"] == calibration,
            "INVALID_NESTED_CALIBRATION", "Stored best lengths differ from the training-scope histories")
    return median_lengths(calibration)


def read_base_fits(store, manifest, package, folds, params):
    fits = {}
    for size in (3, 4):
        for trained in combinations(sorted(folds), size):
            key = fit_id(trained)
            prefix = f"base_fits/{key}"
            _, validation, expected = split_rows(package, folds, trained)
            meta = read_json(output(store, manifest, f"{prefix}/fit.json"))
            lengths = checked_lengths(store, manifest, package, folds, trained, prefix, params, meta)
            require(all(meta.get(k) == v for k, v in expected.items())
                    and all(meta.get(k) == v for k, v in lengths.items()),
                    "INVALID_NESTED_FIT", "Base fit does not match its declared split and training lengths")
            output(store, manifest, f"{prefix}/model.pt")
            history = read_json(output(store, manifest, f"{prefix}/training_history.json"))["epochs"]
            require([e["epoch"] for e in history] == list(range(1, lengths["image_epochs"] + 1)),
                    "INVALID_NESTED_FIT", "Image refit history differs from the selected length")
            rows = pd.read_parquet(output(store, manifest, f"{prefix}/heldout_predictions.parquet"))
            require(all(np.array_equal(rows[c], package.samples.iloc[validation][c]) for c in IDENTITY)
                    and np.isfinite(rows[["ehr", "image"]].to_numpy(float)).all(),
                    "INVALID_NESTED_SCORES", "Base scores must match held-out identities and labels")
            checkpoint = output(store, manifest, f"{prefix}/model.cbm")
            replay = ehr_model.predict(package.features.iloc[validation], package.schema, lengths["ehr_rounds"], checkpoint.parent)
            require(np.array_equal(rows.ehr, replay), "INVALID_NESTED_SCORES", "EHR scores differ from the registered checkpoint")
            fits[key] = rows.set_index("sample_id", drop=False)
    return fits


def nested_predictions(package, folds, fits, controls, work):
    frames, details = [], []
    for outer, indices in sorted(folds.items()):
        other = sorted(set(folds) - {outer})
        pieces = []
        for inner in other:
            source = fit_id(tuple(n for n in other if n != inner))
            ids = package.samples.iloc[folds[inner]].sample_id
            pieces.append(fits[source].loc[ids].assign(inner_fold=inner, base_fit_id=source))
        inner_oof = pd.concat(pieces, ignore_index=True).sort_values("sample_id")
        expected = package.samples.iloc[sorted(i for n in other for i in folds[n])]
        require(inner_oof.sample_id.tolist() == sorted(expected.sample_id)
                and not set(inner_oof.subject_id) & set(package.samples.iloc[indices].subject_id),
                "INVALID_NESTED_FOLDS", "Inner predictions must cover only outer training patients")
        params = fit_coefficients(inner_oof.ehr.to_numpy(float), inner_oof.image.to_numpy(float),
                                  inner_oof.burden_score.to_numpy(float))
        outer_fit = fit_id(tuple(other))
        outer_rows = fits[outer_fit].loc[package.samples.iloc[indices].sample_id].reset_index(drop=True)
        outer_rows["fold"] = outer
        outer_rows["baseline_mean"] = float(expected.burden_score.to_numpy(float).mean())
        for name, pair in params.items():
            outer_rows[name] = (1 - pair["alpha"]) * outer_rows.ehr + pair["alpha"] * outer_rows.image + pair["bias"]
        for weight in controls:
            outer_rows[fixed_name(weight)] = (1 - weight) * outer_rows.ehr + weight * outer_rows.image
        frames.append(outer_rows)
        folder = work / f"fold_{outer}"
        folder.mkdir()
        inner_oof.to_parquet(folder / "inner_oof.parquet", index=False)
        info = {"fold": outer, "outer_base_fit_id": outer_fit, "inner_fits": sorted(inner_oof.base_fit_id.unique()),
                "inner_studies": len(inner_oof), "inner_patients": int(inner_oof.subject_id.nunique()), "coefficients": params}
        write_json(folder / "coefficients.json", info)
        details.append(info)
    rows = pd.concat(frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    require(rows.sample_id.tolist() == sorted(package.samples.iloc[package.cv["development_indices"]].sample_id),
            "INVALID_OOF_COVERAGE", "Nested predictions must cover development exactly once")
    return rows, details


def paired_differences(rows, scores, params):
    draws, seed = params["bootstrap_draws"], params["bootstrap_seed"]
    weights = np.vstack([np.ones(len(rows)), patient_weights(rows.subject_id.to_numpy(), draws, seed)])
    labels = rows.burden_score.to_numpy(float)
    results = {}
    for mode in ("raw", "postprocessed"):
        stats = {k: regression_metrics(labels, v if mode == "raw" else np.clip(v, 0, 8), weights) for k, v in scores.items()}
        results[mode] = {}
        for candidate in ("f3a", "f3b"):
            results[mode][candidate] = {}
            for reference in (name for name in scores if name != candidate):
                differences = results[mode][candidate][reference] = {}
                for metric in ("rmse", "mae"):
                    delta = stats[candidate][metric] - stats[reference][metric]
                    differences[metric] = interval(delta[0], delta[1:])
    return {"difference": "candidate minus matched reference", "bootstrap_unit": "subject_id",
            "confidence_level": 0.95, "bootstrap_draws": draws, "bootstrap_seed": seed, "results": results}


def write_assessment(work, rows, folds, experiment):
    params = experiment["parameters"]
    names = ["baseline_mean", "ehr", "image", "ehr_bias", "f3a", "f3b", *map(fixed_name, params["fixed_image_weights"])]
    scores = {name: rows[name].to_numpy(float) for name in names}
    predictions = rows.assign(predicted_score=scores["f3b"], postprocessed_score=np.clip(scores["f3b"], 0, 8),
                              baseline_mean_score=scores["baseline_mean"])
    predictions.to_parquet(work / "oof_predictions.parquet", index=False)
    write_json(work / "effective_configuration.json", experiment)
    write_json(work / "coefficients.json", {"folds": folds})

    def metrics(mask):
        subset = rows.loc[mask]
        return {"studies": len(subset), "patients": int(subset.subject_id.nunique()),
                **{MODEL if name == "f3b" else name: metric_pair(subset, values[mask]) for name, values in scores.items()}}

    write_json(work / "metrics.json", {"prediction_role": "nested_oof_validation", "primary_prediction": "f3b",
               "folds": [{"fold": int(n), **metrics(rows.fold.to_numpy() == n)} for n in sorted(rows.fold.unique())],
               "pooled_oof": metrics(np.ones(len(rows), dtype=bool))})
    points = {}
    for rule in ("sensitivity", "specificity"):
        target = params[f"{rule}_target"]
        points[rule] = {"model" if name == "f3b" else name: {
            ep: operating_point(rows[ep].to_numpy(bool), values, rule, target) for ep in ENDPOINTS} for name, values in scores.items()}
        plot = work / "threshold_based" / rule_folder(rule, target) / "confusion_matrices.svg"
        plot.parent.mkdir(parents=True)
        confusion_plot(points[rule], plot, title=f"{experiment['id']}: {rule} >= {target:.0%}")
    write_json(work / "operating_points.json", points)
    write_json(work / "paired_differences.json", paired_differences(rows, scores, params))


def publish_assessment(layout, config_path, experiment, package):
    params = experiment["parameters"]
    train = experiment_store(layout, "train", "fusion", MODEL, params["train_id"])
    manifest = train.validate(train.manifest_ref("train", "run"))
    config = read_json(output(train, manifest, "effective_configuration.json"))
    validate_training_parameters(config["parameters"])
    require(manifest["scope"] == train.scope and config["data_release"] == experiment["data_release"]
            and config["id"] == params["train_id"], "ASSESSMENT_DATA_MISMATCH", "F3 training identity differs")
    _, records, _ = resolve_sources(layout, package, config["parameters"])
    require(manifest["inputs"] == bindings(package, records) and read_json(output(train, manifest, "sources.json")) == records,
            "ASSESSMENT_DATA_MISMATCH", "F3 training must bind the exact base protocols and release")
    store = experiment_store(layout, "assessment", "fusion", MODEL, experiment["id"])
    ref = store.manifest_ref("assessment", "run")
    inputs = [{"manifest": manifest["self"], "generation": manifest["generation"]}]
    existing = completed_result(store, ref, experiment, inputs)
    if existing:
        return existing
    folds = partitions(package)
    fits = read_base_fits(train, manifest, package, folds, config["parameters"])
    with store.publish("assessment", "run", inputs=[manifest["self"]], configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "train_id": params["train_id"],
                                "prediction_role": "nested_oof_validation", "primary_prediction": "f3b"}) as work:
        require(store.validate(manifest["self"])["generation"] == manifest["generation"],
                "UPSTREAM_REBUILT", "F3 training changed after base scores were loaded")
        print(f"{MODEL}/{experiment['id']}: fitting fold-local coefficients and assessing matched controls", flush=True)
        rows, details = nested_predictions(package, folds, fits, params["fixed_image_weights"], work)
        write_assessment(work, rows, details, experiment)
    return {"id": experiment["id"], "status": "complete", "manifest": ref}


def run_assessment(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment["parameters"])
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    checked_input_summary(package)
    return run_model_batch(layout, "assessment", "fusion", MODEL, experiments,
                           lambda e: publish_assessment(layout, config_path, e, package))
