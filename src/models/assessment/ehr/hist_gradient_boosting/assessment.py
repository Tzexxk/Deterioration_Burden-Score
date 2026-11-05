"""Assess EHR HistGradientBoosting checkpoints on their validation folds."""

from functools import partial

from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.assessment.ehr.evaluation import assess_model
from src.models.train.ehr.sklearn_training import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    predict = partial(checkpoint_prediction, estimator=HistGradientBoostingRegressor)
    return assess_model(layout, config_path, selected, "hist_gradient_boosting", predict)
