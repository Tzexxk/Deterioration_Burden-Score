"""Refit and replay the selected ResNet50 burden models."""

from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.manifest import write_json
from src.common.paths import require
from src.models.train.image.resnet50 import training
from src.models.train.image.training import (
    FoldImages,
    configure_runtime,
    frozen_head_optimizer,
    run_epoch,
    seed_worker,
)


def loader(package, indices, config, *, shuffle=False):
    return DataLoader(FoldImages(package, indices, augment=False, transform=training.input_transform(config)),
                      batch_size=config["batch_size"], shuffle=shuffle, num_workers=config["num_workers"],
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"]))


def score(package, indices, config, model, device):
    _, ids, values = run_epoch(model, loader(package, indices, config), device)
    require(ids == package.samples.iloc[indices].sample_id.tolist(), "INVALID_IMAGE_COVERAGE",
            "Refit checkpoint predictions must follow frozen sample order")
    return np.asarray(values, dtype=float)


def refit(layout, package, config, rounds, work, *, label):
    supported = {"image_source", "weights", "device", "num_workers", "batch_size", "max_epochs", "head_lr",
                 "weight_decay", "seed", "pretraining", "head_only_epochs", "layer4_lr", "event_components",
                 "early_stopping"}
    require(set(config) <= supported, "UNSUPPORTED_IMAGE_REFIT", "Refit supports linear and event ResNet50 candidates")
    training.validate_settings(config)
    indices = package.cv["development_indices"]
    device = configure_runtime(config["device"], config["seed"])
    weights = training.load_weights(layout.resolve(config["weights"]), config)
    model = training.build_model(config, weights)
    warmup = config.get("head_only_epochs")
    optimizer = (training.layer4_optimizer(model, config, device) if warmup is not None
                 else frozen_head_optimizer(model, model.fc, config, device))
    step = None
    if config.get("event_components"):
        targets = training.event_targets(package.samples.iloc[indices])
        step = partial(training.event_step, targets_by_id=targets)
    images = loader(package, indices, config, shuffle=True)
    history = []
    for epoch in range(1, rounds + 1):
        if warmup is not None:
            training.set_stage(model, epoch, warmup)
        metrics, _, _ = run_epoch(model, images, device, optimizer, training_step=step)
        history.append({"epoch": epoch, "stage": 2 if warmup is not None and epoch > warmup else 1,
                        "train": metrics})
        print(f"{label} epoch={epoch}/{rounds} train RMSE={metrics['rmse']:.6f}", flush=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, work / "model.pt")
    write_json(work / "training_history.json", {"epochs": history})
    expected = score(package, indices, config, model, device)
    restored = predict(package, indices, config, work / "model.pt", str(device))
    require(np.array_equal(expected, restored), "INVALID_CHECKPOINT", "Refit checkpoint must replay development scores")
    return str(device)


def predict(package, indices, config, checkpoint, device_name):
    require(device_name in ("cpu", "cuda:0"), "INVALID_CHECKPOINT", "Unknown checkpoint device")
    device = configure_runtime(device_name.split(":")[0], config["seed"])
    model = training.build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return score(package, indices, config, model.to(device), device)
