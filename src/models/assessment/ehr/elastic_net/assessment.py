"""Assess saved Elastic Net coefficients on the development validation folds."""

from src.models.assessment.ehr.evaluation import assess_model
from src.models.ehr_linear import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, "elastic_net", checkpoint_prediction)
