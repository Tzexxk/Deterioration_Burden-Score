"""Assess EHR XGBoost checkpoints and plot fold training histories."""

from src.models.assessment.ehr.evaluation import assess_model
from src.models.assessment.plots import plot_history
from src.models.train.ehr.xgboost.training import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    plot = plot_history
    return assess_model(layout, config_path, selected, "xgboost", checkpoint_prediction, plot)
