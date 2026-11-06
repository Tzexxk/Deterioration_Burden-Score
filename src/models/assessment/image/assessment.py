"""Recompute development predictions from published image checkpoints."""

from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.paths import read_json, require
from src.models.assessment.evaluation import (
    publish_assessment,
    training_source,
    validate_parameters,
)
from src.models.assessment.plots import plot_history
from src.models.experiments import read_batch, run_model_batch
from src.models.image_data import load_image_package
from src.models.released_data import ReleasedData
from src.models.train.image.training import (
    FoldImages,
    configure_runtime,
    image_runtime_versions,
    run_epoch,
    seed_worker,
)


def output(store, manifest, name):
    return store.layout.resolve(store.output_ref(manifest, name))


def checkpoint_prediction(store, manifest, package, fold, *, config, candidate, build_model, transform):
    n = fold["fold"]
    train, valid = fold["train_indices"], fold["validation_indices"]
    prefix = f"fold_{n}"
    interface = read_json(output(store, manifest, f"{prefix}/interface.json"))
    require(interface["fold"] == n and interface["seed"] == config["seed"]
            and interface["train_sample_ids"] == package.samples.iloc[train].sample_id.tolist()
            and interface["validation_sample_ids"] == package.samples.iloc[valid].sample_id.tolist()
            and interface["device"] in ("cpu", "cuda:0"),
            "INVALID_IMAGE_CHECKPOINT", "Image checkpoint fold, seed, device or sample IDs differ")
    history = read_json(output(store, manifest, f"{prefix}/history.json"))["epochs"]
    learn, validation = ([row[role]["rmse"] for row in history] for role in ("train", "validation"))
    best = interface["best_epoch"]
    require(bool(history) and [row["epoch"] for row in history] == list(range(1, len(history) + 1))
            and np.isfinite(learn).all() and np.isfinite(validation).all()
            and best == int(np.argmin(validation)) + 1,
            "INVALID_TRAINING_HISTORY", "RMSE history differs from the image best checkpoint")

    print(f"{candidate} fold={n}: reloading epoch {best} and scoring validation images", flush=True)
    device = configure_runtime(interface["device"].split(":")[0], config["seed"])
    checkpoint = torch.load(output(store, manifest, f"{prefix}/best.pt"), map_location="cpu", weights_only=True)
    model = build_model(config)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    loader = DataLoader(FoldImages(package, valid, augment=False, transform=transform), batch_size=config["batch_size"],
                        num_workers=config["num_workers"], worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(config["seed"]))
    metrics, ids, scores = run_epoch(model, loader, device)
    saved = pd.read_parquet(output(store, manifest, f"{prefix}/validation_predictions.parquet"))
    require(ids == interface["validation_sample_ids"] == saved.sample_id.tolist()
            and np.array_equal(scores, saved.predicted_score.to_numpy(float))
            and metrics == history[best - 1]["validation"],
            "IMAGE_PREDICTIONS_MISMATCH", "Reloaded checkpoint disagrees with published validation predictions")
    return np.asarray(scores), {"fold": n, "best_round": best, "learn": learn, "validation": validation}


def assess_model(layout, config_path, selected, model, build_model, input_transform,
                 runtime_versions=image_runtime_versions):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment)
    release = ReleasedData.load(layout, experiments[0]["data_release"])

    def assess(experiment):
        train_id = experiment["parameters"]["train_id"]
        source = training_source(layout, release, train_id, model, modality="image")
        store, manifest = source
        snapshot = read_json(output(store, manifest, "effective_configuration.json"))
        require(snapshot["id"] == train_id and snapshot["data_release"] == experiment["data_release"]
                and snapshot["runtime_versions"] == runtime_versions(snapshot["parameters"]),
                "INVALID_IMAGE_TRAINING_SNAPSHOT", "Training ID, release or image runtime differs")
        config = snapshot["parameters"]
        package = load_image_package(release, config["image_source"])
        require(snapshot["resolved_image_source"] == package.source
                and snapshot["image_shape"] == list(package.shape) == manifest["details"]["image_shape"],
                "ASSESSMENT_DATA_MISMATCH", "Image source or dimensions differ from training")
        candidate = f"{model}/{experiment['id']}"
        predict = partial(checkpoint_prediction, config=config, candidate=candidate, build_model=build_model,
                          transform=input_transform(config))
        plot = partial(plot_history, round_label="Epoch")
        return publish_assessment(layout, config_path, experiment, package, model, predict, plot,
                                  modality="image", source=source)

    return run_model_batch(layout, "assessment", "image", model, experiments, assess)
