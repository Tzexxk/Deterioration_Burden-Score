"""Fit a linear regression head on a frozen ImageNet-1K Swin-B."""

import torch
from torch import nn
from torchvision.models import swin_b

from src.models.train.image.training import (
    configure_runtime,
    fit_frozen_head,
    train_model,
    validate_frozen_settings,
    validate_pretrained_state,
)

MODEL = "swin_b"


def build_model(_config, weights=None):
    model = swin_b(weights=None)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    model.head = nn.Linear(model.head.in_features, 1)
    return model.float()


def load_weights(path, _config):
    return validate_pretrained_state(torch.load(path, map_location="cpu", weights_only=True))


def fit_fold(package, fold, config, weights, folder, *, label):
    device = configure_runtime(config["device"], config["seed"])
    model = build_model(config, weights)
    return fit_frozen_head(package, fold, config, model, model.head, folder, label=label, device=device)


def run_training(layout, config_path, selected=None):
    return train_model(layout, config_path, selected, MODEL, validate_frozen_settings, fit_fold, load_weights)
