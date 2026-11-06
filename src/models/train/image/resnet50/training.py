"""Fit a linear regression head on frozen ResNet50 features."""

import torch
from torch import nn
from torchvision.models import resnet50

from src.common.paths import require
from src.models.train.image.training import (
    configure_runtime,
    fit_frozen_head,
    imagenet_input,
    train_model,
    validate_frozen_settings,
    validate_pretrained_state,
)

MODEL = "resnet50"
IMAGENET = "IMAGENET1K_V1"
XRV = "resnet50-res512-all"


def xray_input(gray):
    return gray * 2048 - 1024


def input_transform(config):
    return xray_input if config["pretraining"] == XRV else imagenet_input


def build_model(config, weights=None):
    model = resnet50(weights=None)
    head = nn.Linear(model.fc.in_features, 1)
    if config["pretraining"] == XRV:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights is not None:
            model.fc = nn.Linear(model.fc.in_features, 18)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    model.fc = head
    return model.float()


def load_weights(path, _config):
    return validate_pretrained_state(torch.load(path, map_location="cpu", weights_only=True))


def validate_settings(config):
    require(config.get("pretraining") in (IMAGENET, XRV), "INVALID_IMAGE_CONFIG",
            "Select an ImageNet-1K or TorchXRayVision ResNet50 checkpoint")
    validate_frozen_settings({key: value for key, value in config.items() if key != "pretraining"})


def fit_fold(package, fold, config, weights, folder, *, label):
    device = configure_runtime(config["device"], config["seed"])
    model = build_model(config, weights)
    return fit_frozen_head(package, fold, config, model, model.fc, folder, label=label, device=device,
                           transform=input_transform(config))


def run_training(layout, config_path, selected=None):
    return train_model(layout, config_path, selected, MODEL, validate_settings, fit_fold, load_weights)
