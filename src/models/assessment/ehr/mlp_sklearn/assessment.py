"""Assess EHR scikit-learn MLP checkpoints on their validation folds."""

from functools import partial

from sklearn.neural_network import MLPRegressor

from src.models.assessment.ehr.evaluation import assess_model
from src.models.train.ehr.sklearn_training import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    predict = partial(checkpoint_prediction, estimator=MLPRegressor)
    return assess_model(layout, config_path, selected, "mlp_sklearn", predict)
