"""Assess saved SVR checkpoints on the development validation folds."""

from src.models.assessment.ehr.evaluation import assess_model
from src.models.train.ehr.svr.training import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, "svr", checkpoint_prediction)
