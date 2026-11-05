"""Train the configured EHR HistGradientBoosting candidates."""

from functools import partial

from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.train.ehr import sklearn_training

fit_fold = partial(sklearn_training.fit_fold, estimator=HistGradientBoostingRegressor, family="hist")
run_training = partial(sklearn_training.run_training, model="hist_gradient_boosting", family="hist", estimator=HistGradientBoostingRegressor)
