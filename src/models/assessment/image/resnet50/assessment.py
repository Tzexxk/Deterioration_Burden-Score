"""Assess published ResNet50 development checkpoints."""

from src.models.assessment.image.assessment import assess_model
from src.models.assessment.image.resnet50.calibration import prepare_calibration
from src.models.train.image.resnet50.training import MODEL, build_model, input_transform


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, MODEL, build_model, input_transform,
                        prepare_calibration=prepare_calibration)
