"""Recompute development predictions from the published ResNet18 checkpoints."""

from src.models.assessment.image.assessment import assess_model
from src.models.train.image.resnet18.training import build_model
from src.models.train.image.training import imagenet_input


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, "resnet18", lambda _config: build_model(),
                        lambda _config: imagenet_input)
