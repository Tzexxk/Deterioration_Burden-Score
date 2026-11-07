"""Fit burden-score heads on frozen or fine-tuned ResNet50 features."""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from src.cohort.burden_io import COMPONENTS, score
from src.common.paths import require
from src.models.train.image.training import (
    configure_runtime,
    fit_epochs,
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


class OrdinalHead(nn.Module):
    def __init__(self, features, thresholds):
        super().__init__()
        self.projection = nn.Linear(features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(2.0))
        self.cutpoint_gaps = nn.Parameter(torch.full((len(thresholds) - 1,), math.log(math.expm1(1))))
        self.register_buffer("thresholds", torch.tensor(thresholds), persistent=False)
        self.register_buffer("score_steps", torch.diff(self.thresholds, prepend=self.thresholds.new_zeros(1)),
                             persistent=False)

    def forward(self, features):
        cutpoints = [self.first_cutpoint]
        for gap in F.softplus(self.cutpoint_gaps).unbind():
            cutpoints.append(cutpoints[-1] + gap)
        return self.projection(features) - torch.stack(cutpoints)

    def score(self, logits):
        return (logits.sigmoid() * self.score_steps).sum(dim=1, keepdim=True)


class EventHead(nn.Linear):
    def __init__(self, features):
        super().__init__(features, 4)
        self.register_buffer("score_weights", torch.tensor([1., 2., 2., 3.]), persistent=False)

    def score(self, logits):
        return (logits.sigmoid() * self.score_weights).sum(dim=1, keepdim=True)


class BurdenHeadResNet(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.fc = head

    def logits(self, image):
        return self.fc(self.backbone(image))

    def forward(self, image):
        return self.fc.score(self.logits(image))


def ordinal_step(model, image, labels, _batch_ids):
    logits = model.logits(image)
    targets = (labels >= model.fc.thresholds).to(logits.dtype)
    return model.fc.score(logits), F.binary_cross_entropy_with_logits(logits, targets)


def event_targets(rows):
    targets = {}
    for sample_id, *flags in rows[["sample_id", *COMPONENTS]].itertuples(index=False, name=None):
        components = score(flags)
        targets[sample_id] = [components[name] for name in ("T", "R", "C", "D")]
    return targets


def event_step(model, image, _labels, batch_ids, *, targets_by_id):
    logits = model.logits(image)
    targets = logits.new_tensor([targets_by_id[sample_id] for sample_id in batch_ids])
    return model.fc.score(logits), F.binary_cross_entropy_with_logits(logits, targets)


class MixedGlobalPool(nn.Module):
    def __init__(self, max_weight):
        super().__init__()
        self.max_weight = max_weight

    def forward(self, features):
        average = F.adaptive_avg_pool2d(features, 1)
        maximum = features.amax(dim=(-2, -1), keepdim=True)
        return (1 - self.max_weight) * average + self.max_weight * maximum


def build_model(config, weights=None):
    model = resnet50(weights=None)
    features = model.fc.in_features
    if "ordinal_thresholds" in config or "event_components" in config:
        head = nn.Identity()
    elif "mlp_hidden_size" in config:
        hidden = config["mlp_hidden_size"]
        head = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    else:
        head = nn.Linear(features, 1)
    if config["pretraining"] == XRV:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights is not None:
            model.fc = nn.Linear(model.fc.in_features, 18)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    if "max_pool_weight" in config:
        model.avgpool = MixedGlobalPool(config["max_pool_weight"])
    model.fc = head
    if "ordinal_thresholds" in config:
        model = BurdenHeadResNet(model, OrdinalHead(features, config["ordinal_thresholds"]))
    elif "event_components" in config:
        model = BurdenHeadResNet(model, EventHead(features))
    return model.float()


def load_weights(path, _config):
    return validate_pretrained_state(torch.load(path, map_location="cpu", weights_only=True))


def validate_settings(config):
    require(config.get("pretraining") in (IMAGENET, XRV), "INVALID_IMAGE_CONFIG",
            "Select an ImageNet-1K or TorchXRayVision ResNet50 checkpoint")
    variants = {"high_score_weight", "lds", "head_only_epochs", "layer4_lr", "mlp_hidden_size", "ordinal_thresholds",
                "mean_output_init", "event_components"}
    optional = variants | {"pretraining", "early_stopping", "update_layer4_bn", "max_pool_weight", "train_affine"}
    validate_frozen_settings({key: value for key, value in config.items() if key not in optional})
    if "max_pool_weight" in config:
        weight = config["max_pool_weight"]
        require(type(weight) in (int, float) and 0 < weight < 1,
                "INVALID_IMAGE_CONFIG", "Mixed pooling requires a max weight between zero and one")
    if "train_affine" in config:
        affine = config["train_affine"]
        require(isinstance(affine, dict)
                and set(affine) == {"probability", "degrees", "translate_fraction"}
                and all(type(value) in (int, float) for value in affine.values())
                and 0 < affine["probability"] <= 1 and 0 < affine["degrees"] <= 180
                and 0 < affine["translate_fraction"] < 1,
                "INVALID_IMAGE_CONFIG", "Weak affine augmentation requires valid probability and transform bounds")
    if "early_stopping" in config:
        rule = config["early_stopping"]
        require(isinstance(rule, dict) and set(rule) == {"min_epochs", "patience"}
                and all(type(value) is int and value > 0 for value in rule.values())
                and rule["min_epochs"] + rule["patience"] <= config["max_epochs"] and "layer4_lr" not in config,
                "INVALID_IMAGE_CONFIG", "Early stopping requires a frozen encoder and a valid epoch budget")
    if "update_layer4_bn" in config:
        require(config["update_layer4_bn"] is True and "layer4_lr" in config,
                "INVALID_IMAGE_CONFIG", "Layer4 BatchNorm updates require layer4 fine-tuning")
    for option in ("mean_output_init", "event_components"):
        if option in config:
            require(config[option] is True and (config.keys() & variants) == {option},
                    "INVALID_IMAGE_CONFIG", f"{option} must be true and used alone with the frozen baseline")
    if "mlp_hidden_size" in config:
        hidden = config["mlp_hidden_size"]
        require(type(hidden) is int and hidden > 0, "INVALID_IMAGE_CONFIG", "MLP hidden size must be positive")
    if "ordinal_thresholds" in config:
        thresholds = config["ordinal_thresholds"]
        require(isinstance(thresholds, list) and all(type(value) is int for value in thresholds)
                and thresholds == sorted(set(thresholds)) and thresholds[:1] == [1] and thresholds[-1:] == [8]
                and not config.keys() & {"mlp_hidden_size", "high_score_weight", "lds", "head_only_epochs", "layer4_lr"},
                "INVALID_IMAGE_CONFIG", "Ordinal thresholds must increase from 1 to 8 and use a frozen encoder")
    if "high_score_weight" in config:
        weighting = config["high_score_weight"]
        require(isinstance(weighting, dict) and set(weighting) == {"threshold", "weight"}
                and all(type(value) in (int, float) and math.isfinite(value) for value in weighting.values())
                and 0 < weighting["threshold"] <= 8 and weighting["weight"] > 1,
                "INVALID_IMAGE_CONFIG", "Expected a high-score threshold in (0, 8] and weight above 1")
    if "lds" in config:
        lds = config["lds"]
        require("high_score_weight" not in config
                and isinstance(lds, dict) and set(lds) == {"bandwidth", "power", "max_ratio"}
                and all(type(value) in (int, float) and math.isfinite(value) for value in lds.values())
                and lds["bandwidth"] > 0 and lds["power"] > 0 and lds["max_ratio"] > 1,
                "INVALID_IMAGE_CONFIG", "LDS requires positive bandwidth and power, a ratio above 1, and no other weighting")
    if "head_only_epochs" in config or "layer4_lr" in config:
        warmup, rate = config.get("head_only_epochs"), config.get("layer4_lr")
        require(type(warmup) is int and 0 < warmup < config["max_epochs"]
                and type(rate) in (int, float) and math.isfinite(rate) and rate > 0,
                "INVALID_IMAGE_CONFIG", "Layer4 fine-tuning requires head-only epochs and a positive learning rate")


def weighted_mse(error, labels, *, threshold, weight):
    return (error.square() * torch.where(labels >= threshold, weight, 1.0)).mean()


def lds_weights(labels, *, bandwidth, power, max_ratio):
    labels = torch.as_tensor(labels, dtype=torch.float64)
    require(labels.numel() > 0 and bool(((labels >= 0) & (labels <= 8) & (labels == labels.floor())).all()),
            "INVALID_IMAGE_LABELS", "LDS expects integer training scores from 0 to 8")
    counts = torch.bincount(labels.long(), minlength=9).to(labels.dtype)
    scores = torch.arange(9, dtype=labels.dtype)
    distance = (scores[:, None] - scores) / bandwidth
    density = (torch.exp(-0.5 * distance.square()) @ counts).clamp_min(torch.finfo(labels.dtype).tiny)
    ratio = (density[0] / density).pow(power).clamp(1, max_ratio)
    return (ratio / (ratio @ counts / counts.sum())).float()


def lds_mse(error, labels, *, weights):
    return (error.square() * weights[labels.long()]).mean()


def set_stage(model, epoch, head_only_epochs, *, update_layer4_bn=False):
    fine_tuning = epoch > head_only_epochs
    model.layer4.requires_grad_(fine_tuning)
    if update_layer4_bn:
        for module in model.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(fine_tuning)


def fit_fold(package, fold, config, weights, folder, *, label):
    device = configure_runtime(config["device"], config["seed"])
    model = build_model(config, weights)
    transform = input_transform(config)
    training_loss = partial(weighted_mse, **config["high_score_weight"]) if "high_score_weight" in config else None
    training_step = ordinal_step if "ordinal_thresholds" in config else None
    if "mean_output_init" in config:
        train_mean = package.samples.iloc[fold["train_indices"]].burden_score.mean()
        nn.init.zeros_(model.fc.weight)
        nn.init.constant_(model.fc.bias, float(train_mean))
    if "event_components" in config:
        rows = package.samples.iloc[fold["train_indices"]]
        training_step = partial(event_step, targets_by_id=event_targets(rows))
    if "lds" in config:
        labels = package.samples.iloc[fold["train_indices"]].burden_score.to_numpy(copy=True)
        weights_by_score = lds_weights(labels, **config["lds"]).to(device)
        training_loss = partial(lds_mse, weights=weights_by_score)
    if "layer4_lr" in config:
        model.to(device).eval().requires_grad_(False)
        model.fc.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [{"params": model.fc.parameters(), "lr": config["head_lr"]},
             {"params": model.layer4.parameters(), "lr": config["layer4_lr"]}],
            weight_decay=config["weight_decay"], foreach=False, fused=False)
        return fit_epochs(package, fold, {**config, "patience": config["max_epochs"]}, model, optimizer, folder,
                          label=label, device=device, transform=transform,
                          schedule=partial(set_stage, update_layer4_bn=config.get("update_layer4_bn", False)),
                          head_only_epochs=config["head_only_epochs"], training_loss=training_loss)
    return fit_frozen_head(package, fold, config, model, model.fc, folder, label=label, device=device,
                           transform=transform, training_loss=training_loss, training_step=training_step)


def run_training(layout, config_path, selected=None):
    return train_model(layout, config_path, selected, MODEL, validate_settings, fit_fold, load_weights)
