"""Fit a linear regression head on a frozen DenseNet-121 backbone."""

import math
import re
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn
from torchvision.models import densenet121

from src.common.paths import PipelineError, require
from src.models.train.image.training import (
    configure_runtime,
    fit_epochs,
    image_runtime_versions,
    imagenet_input,
    train_model,
)

MODEL = "densenet121"
CHEXPERT = "densenet121-res224-chex"
IMAGENET = "IMAGENET1K_V1"
CHEXPERT_FILE = "chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
LEGACY_DENSE_KEY = re.compile(r"(denselayer\d+\.(?:norm|relu|conv))\.([12])\.(weight|bias|running_mean|running_var)$")


def chexpert_input(gray):
    return gray * 2048 - 1024


def input_transform(config):
    return chexpert_input if config["pretraining"] == CHEXPERT else imagenet_input


def build_model(config):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    if config["pretraining"] == CHEXPERT:
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model.float()


def runtime_versions(config):
    versions = image_runtime_versions(config)
    if config["pretraining"] == CHEXPERT:
        try:
            versions["torchxrayvision"] = version("torchxrayvision")
        except PackageNotFoundError as error:
            raise PipelineError("TORCHXRAYVISION_MISSING", "Install torchxrayvision for CheXpert weights") from error
    return versions


def load_weights(path, config):
    if config["pretraining"] == CHEXPERT:
        require(path.name == CHEXPERT_FILE, "INVALID_PRETRAINED_WEIGHTS",
                "Use the official local TorchXRayVision CheXpert checkpoint filename")
        from torchxrayvision.models import DenseNet

        # The official legacy file contains a model object. Only load the trusted release.
        state = DenseNet(weights=CHEXPERT, cache_dir=str(path.parent)).state_dict()
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    require(isinstance(state, dict) and all(isinstance(name, str) for name in state),
            "INVALID_PRETRAINED_WEIGHTS", "Expected a DenseNet-121 state dictionary")
    backbone = {
        LEGACY_DENSE_KEY.sub(r"\g<1>\g<2>.\g<3>", name.removeprefix("features.")): tensor
        for name, tensor in state.items() if name.startswith("features.")
    }
    require(bool(backbone) and all(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all())
                                   for tensor in backbone.values()),
            "INVALID_PRETRAINED_WEIGHTS", "Expected finite DenseNet-121 backbone tensors")
    return backbone


def validate_settings(config):
    integers = {"seed", "num_workers", "batch_size", "max_epochs"}
    rates = {"head_lr", "weight_decay"}
    require(set(config) == integers | rates | {"pretraining", "image_source", "weights", "device"}
            and config["pretraining"] in (CHEXPERT, IMAGENET)
            and config["device"] in ("auto", "cpu", "cuda")
            and all(type(config[key]) is int and config[key] >= 0 for key in integers)
            and config["batch_size"] > 0 and config["max_epochs"] > 0 and config["seed"] < 2**32
            and all(type(config[key]) in (int, float) and math.isfinite(config[key]) and config[key] >= 0
                    for key in rates)
            and config["head_lr"] > 0,
            "INVALID_IMAGE_CONFIG", "Invalid DenseNet-121 training settings")


def fit_fold(package, fold, config, weights, folder, *, label):
    device = configure_runtime(config["device"], config["seed"])
    model = build_model(config)
    model.features.load_state_dict(weights, strict=True)
    model.to(device).eval().requires_grad_(False)
    model.classifier.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config["head_lr"],
                                  weight_decay=config["weight_decay"], foreach=False, fused=False)
    return fit_epochs(package, fold, config, model, optimizer, folder, label=label, device=device,
                      transform=input_transform(config))


def run_training(layout, config_path, selected=None):
    return train_model(layout, config_path, selected, MODEL, validate_settings, fit_fold, load_weights,
                       runtime_versions)
