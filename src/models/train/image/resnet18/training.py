"""Train and publish a fresh ResNet18 for each development fold."""

import math

import torch
from torch import nn
from torchvision.models import resnet18

from src.common.paths import require
from src.models.train.image.training import configure_runtime, fit_epochs, train_model


def build_model(weights=None):
    model = resnet18(weights=None)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.float()


def set_stage(model, epoch, head_only_epochs):
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    model.fc.train().requires_grad_(True)
    if epoch > head_only_epochs:
        model.layer4.train().requires_grad_(True)


def fit_fold(package, fold, config, weights, folder, *, label):
    seed = config["seed"]
    device = configure_runtime(config["device"], seed)
    model = build_model(weights).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.fc.parameters(), "lr": config["head_lr"]},
         {"params": model.layer4.parameters(), "lr": config["layer4_lr"]}],
        weight_decay=config["weight_decay"], betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
    return fit_epochs(package, fold, config, model, optimizer, folder, label=label, device=device,
                      augment=config.get("augment", True), schedule=set_stage,
                      head_only_epochs=config["head_only_epochs"])


def validate_settings(config):
    integers = {"seed", "num_workers", "batch_size", "head_only_epochs", "max_epochs", "patience"}
    rates = {"head_lr", "layer4_lr", "weight_decay"}
    require(set(config) - {"augment"} == integers | rates | {"image_source", "weights", "device"}
            and type(config.get("augment", True)) is bool
            and config["device"] in ("auto", "cpu", "cuda")
            and all(type(config[k]) is int and config[k] >= 0 for k in integers)
            and config["batch_size"] > 0 and config["patience"] > 0 and 0 <= config["seed"] < 2**32
            and config["max_epochs"] > 0 and config["head_only_epochs"] <= config["max_epochs"]
            and all(type(config[k]) in (int, float) and math.isfinite(config[k]) and config[k] >= 0 for k in rates)
            and config["head_lr"] > 0 and config["layer4_lr"] > 0,
            "INVALID_IMAGE_CONFIG", "Invalid ResNet18 training settings")


def load_weights(path, _config):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(weights, dict) and all(isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all())
                                              for t in weights.values()),
            "INVALID_PRETRAINED_WEIGHTS", "Expected a finite ResNet18 backbone state dictionary")
    return weights


def run_training(layout, config_path, selected=None):
    return train_model(layout, config_path, selected, "resnet18", validate_settings, fit_fold, load_weights)
