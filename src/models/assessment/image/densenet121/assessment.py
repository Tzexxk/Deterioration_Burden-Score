"""Assess published DenseNet-121 development checkpoints."""

from src.models.assessment.image.assessment import assess_model
from src.models.train.image.densenet121.training import (
    MODEL,
    build_model,
    input_transform,
    runtime_versions,
)


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, MODEL, build_model, input_transform, runtime_versions)
