"""Fit a linear head on features from locally loaded timm ImageNet models."""

from collections import OrderedDict
from functools import partial
from importlib.metadata import PackageNotFoundError, version

from torch import nn

from src.common.paths import PipelineError
from src.models.train.image.training import (
    configure_runtime,
    fit_frozen_head,
    image_runtime_versions,
    train_model,
    validate_frozen_settings,
    validate_pretrained_state,
)


def build_timm_model(architecture, _config, weights=None, **model_options):
    from timm import create_model

    encoder = create_model(architecture, pretrained=False, **model_options)
    if weights is not None:
        encoder.load_state_dict(weights, strict=True)
    encoder.reset_classifier(0)
    return nn.Sequential(OrderedDict([
        ("encoder", encoder),
        ("head", nn.Linear(encoder.num_features, 1)),
    ])).float()


def runtime_versions(config):
    versions = image_runtime_versions(config)
    try:
        versions.update({name: version(name) for name in ("timm", "safetensors")})
    except PackageNotFoundError as error:
        raise PipelineError("TIMM_RUNTIME_MISSING", "Install timm and safetensors for this image model") from error
    return versions


def load_weights(path, _config):
    from safetensors.torch import load_file

    return validate_pretrained_state(load_file(str(path), device="cpu"))


def fit_fold(package, fold, config, weights, folder, *, label, build_model):
    device = configure_runtime(config["device"], config["seed"])
    model = build_model(config, weights)
    return fit_frozen_head(package, fold, config, model, model.head, folder, label=label, device=device)


def train_timm_model(layout, config_path, selected, model, build_model):
    fit = partial(fit_fold, build_model=build_model)
    return train_model(layout, config_path, selected, model, validate_frozen_settings, fit, load_weights,
                       runtime_versions)
