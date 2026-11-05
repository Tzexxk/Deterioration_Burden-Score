"""Assess saved Kernel Ridge checkpoints on the development validation folds."""

from src.models.assessment.ehr.evaluation import assess_model
from src.models.train.ehr.kernel_ridge.training import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, "kernel_ridge", checkpoint_prediction)
