"""Assess published DeiT-B/16 development checkpoints."""

from src.models.assessment.image.assessment import assess_model
from src.models.train.image.deit_base_patch16_224.training import MODEL, build_model
from src.models.train.image.timm_frozen import runtime_versions
from src.models.train.image.training import imagenet_input


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, MODEL, build_model, lambda _config: imagenet_input, runtime_versions)
