"""Fit the unique base-model splits required by nested five-fold fusion."""

from dataclasses import replace
from itertools import combinations
from statistics import median

import numpy as np

from src.common.manifest import write_json
from src.common.paths import identifier, require
from src.models.ehr_data import checked_input_summary, load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store, read_batch, run_model_batch
from src.models.released_data import ReleasedData
from src.models.test.ehr.catboost import model as ehr_model
from src.models.test.image.resnet50 import model as image_model
from src.models.train.ehr.catboost.training import fit_fold as calibrate_ehr
from src.models.train.fusion.feature_fusion.training import output, resolve_sources

MODEL = "learned_late"
TRAINING_MEDIAN = "training_median"
IDENTITY = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]


def fit_id(folds):
    return "train_" + "".join(str(fold) for fold in folds)


def partitions(package):
    folds = {f["fold"]: f["validation_indices"] for f in package.cv["folds"]}
    all_rows = sorted(index for indices in folds.values() for index in indices)
    require(set(folds) == set(range(1, 6)) and all_rows == sorted(package.cv["development_indices"])
            and len(all_rows) == len(set(all_rows)) and not set(all_rows) & set(package.cv["test_indices"]),
            "INVALID_NESTED_FOLDS", "Expected five disjoint development folds")
    return folds


def split_rows(package, folds, trained):
    scored = sorted(set(folds) - set(trained))
    train = sorted(index for n in trained for index in folds[n])
    validation = sorted(index for n in scored for index in folds[n])
    train_rows, val_rows = package.samples.iloc[train], package.samples.iloc[validation]
    require(not set(train_rows.subject_id) & set(val_rows.subject_id),
            "INVALID_NESTED_FOLDS", "Fit and scoring patients must be disjoint")
    metadata = {"train_folds": list(trained), "score_folds": scored,
                "train_sample_ids": train_rows.sample_id.tolist(), "score_sample_ids": val_rows.sample_id.tolist(),
                "train_subject_ids": sorted(set(map(int, train_rows.subject_id))),
                "score_subject_ids": sorted(set(map(int, val_rows.subject_id)))}
    return train, validation, metadata


def validate_parameters(params):
    sources = {"ehr_source", "image_source"}
    fixed = set(params) == sources | {"ehr_rounds", "image_epochs"}
    require(fixed or (set(params) == sources | {"round_policy"} and params["round_policy"] == TRAINING_MEDIAN),
            "INVALID_FUSION_CONFIG", "Specify source assessments and a training-median policy or fixed lengths")
    for family, model in (("ehr", "catboost"), ("image", "resnet50")):
        choice = params[f"{family}_source"]
        require(isinstance(choice, dict) and set(choice) == {"model", "assessment_id"} and choice["model"] == model,
                "INVALID_FUSION_CONFIG", "Use CatBoost and ResNet50 sources")
        identifier(choice["assessment_id"])
    require(not fixed or all(type(params[k]) is int and params[k] > 0 for k in ("ehr_rounds", "image_epochs")),
            "INVALID_FUSION_CONFIG", "Training lengths must be positive integers")


def bindings(package, records):
    acceptance = package.release.acceptance
    return [{"manifest": acceptance["self"], "generation": acceptance["generation"]},
            records["ehr"]["assessment"], records["image"]["assessment"]]


def calibration_fold(folds, trained, held):
    return {"fold": held, "train_indices": sorted(i for n in trained if n != held for i in folds[n]),
            "validation_indices": folds[held]}


def median_lengths(calibration):
    return {"ehr_rounds": int(median(row["ehr_best_round"] for row in calibration) + 0.5),
            "image_epochs": int(median(row["image_best_epoch"] for row in calibration) + 0.5)}


def calibrate_lengths(work, package, images, folds, trained, records, weights, label):
    ehr = records["ehr"]["effective_training_configuration"]
    image = records["image"]["effective_training_configuration"]["parameters"]
    results = []
    for held in trained:
        fold = calibration_fold(folds, trained, held)
        folder = work / "calibration" / f"fold_{held}"
        folder.mkdir(parents=True)
        print(f"{label}: calibrating lengths on fold {held} inside {fit_id(trained)}", flush=True)
        ehr_result = calibrate_ehr(package, ehr, fold, folder / "ehr")
        image_result = image_model.training.fit_fold(images, fold, image, weights, folder / "image",
                                                     label=f"{label} {fit_id(trained)} calibration={held}")
        results.append({"validation_fold": held, "ehr_best_round": ehr_result["best_round"],
                        "image_best_epoch": image_result["best_epoch"]})
        # Histories and interfaces are retained for assessment to reproduce the length selection.
        (folder / "ehr/model.cbm").unlink()
        (folder / "image/best.pt").unlink()
    return results


def fit_base(layout, work, package, images, folds, trained, records, lengths, label, calibration):
    train, validation, metadata = split_rows(package, folds, trained)
    ehr = records["ehr"]["effective_training_configuration"]["parameters"]
    image = records["image"]["effective_training_configuration"]["parameters"]
    x, y = package.features, package.samples.burden_score
    print(f"{label} {fit_id(trained)}: refitting {lengths['ehr_rounds']} CatBoost rounds and "
          f"{lengths['image_epochs']} image epochs on {len(train)} studies", flush=True)
    ehr_model.fit(x.iloc[train], y.iloc[train].to_numpy(float), package.schema, ehr, lengths["ehr_rounds"], work)
    ehr_scores = ehr_model.predict(x.iloc[validation], package.schema, lengths["ehr_rounds"], work)
    restricted = replace(images, cv={**images.cv, "development_indices": train})
    device = image_model.refit(layout, restricted, image, lengths["image_epochs"], work,
                               label=f"{label} {fit_id(trained)}")
    image_scores = image_model.predict(images, validation, image, work / "model.pt", device)
    require(all(np.shape(values) == (len(validation),) and np.isfinite(values).all()
                for values in (ehr_scores, image_scores)), "INVALID_NESTED_SCORES", "Expected finite held-out base scores")
    rows = package.samples.iloc[validation][IDENTITY].assign(ehr=ehr_scores, image=image_scores)
    rows.to_parquet(work / "heldout_predictions.parquet", index=False)
    write_json(work / "fit.json", {**metadata, **lengths, "image_device": device,
                                  "selection": TRAINING_MEDIAN if calibration else "fixed_training_lengths",
                                  "calibration": calibration})


def publish_training(layout, config_path, experiment, package):
    params = experiment["parameters"]
    sources, records, images = resolve_sources(layout, package, params)
    ehr = records["ehr"]["effective_training_configuration"]["parameters"]
    image = records["image"]["effective_training_configuration"]["parameters"]
    use_median = params.get("round_policy") == TRAINING_MEDIAN
    require(use_median or (params["ehr_rounds"] <= ehr["model"]["iterations"]
            and image["head_only_epochs"] < params["image_epochs"] <= image["max_epochs"]),
            "INVALID_FUSION_CONFIG", "Fixed lengths must be within source caps and include layer4 training")
    folds = partitions(package)
    splits = [trained for size in (3, 4) for trained in combinations(sorted(folds), size)]
    inputs = bindings(package, records)
    store = experiment_store(layout, "train", "fusion", MODEL, experiment["id"])
    ref = store.manifest_ref("train", "run")
    existing = completed_result(store, ref, experiment, inputs)
    if existing:
        return existing
    require(layout.resolve(image["weights"]).is_file(), "MISSING_PRETRAINED_WEIGHTS", "Image initialization weights are missing")
    image_manifest = sources["image"][1]
    checkpoint_bytes = max(item["bytes"] for item in image_manifest["outputs"] if item["name"].endswith("/best.pt"))
    with store.publish("train", "run", inputs=[x["manifest"] for x in inputs],
                       configs=[layout.ref("project", config_path)],
                       required_bytes=(len(splits) + int(use_median)) * checkpoint_bytes + 64 * 1024**2,
                       details={"data_release": experiment["data_release"], "base_fits_per_modality": len(splits)}) as work:
        require(all(store.validate(x["manifest"])["generation"] == x["generation"] for x in inputs),
                "UPSTREAM_REBUILT", "Source changed after configuration was loaded")
        weights = image_model.training.load_weights(layout.resolve(image["weights"]), image) if use_median else None
        for trained in splits:
            folder = work / "base_fits" / fit_id(trained)
            folder.mkdir(parents=True)
            label = f"{MODEL}/{experiment['id']}"
            calibration = calibrate_lengths(folder, package, images, folds, trained, records, weights, label) if use_median else []
            lengths = median_lengths(calibration) if use_median else {k: params[k] for k in ("ehr_rounds", "image_epochs")}
            fit_base(layout, folder, package, images, folds, trained, records, lengths, label, calibration)
        write_json(work / "effective_configuration.json", experiment)
        write_json(work / "sources.json", records)
        write_json(work / "training_summary.json", {"id": experiment["id"], "base_fit_ids": [fit_id(x) for x in splits],
                   "inner_fits_per_modality": 10, "outer_fits_per_modality": 5, "test_consumed_for_fitting": False,
                   "calibration_fits_per_modality": sum(map(len, splits)) if use_median else 0,
                   "prior_fixed_test_exposure": "Both source test results were viewed and used for source selection."})
    return {"id": experiment["id"], "status": "complete", "manifest": ref}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment["parameters"])
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    checked_input_summary(package)
    return run_model_batch(layout, "train", "fusion", MODEL, experiments,
                           lambda e: publish_training(layout, config_path, e, package))
