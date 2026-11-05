"""Train the configured EHR Random Forest candidates."""

from functools import partial

from sklearn.ensemble import RandomForestRegressor

from src.models.train.ehr import sklearn_training

fit_fold = partial(sklearn_training.fit_fold, estimator=RandomForestRegressor, family="forest")
run_training = partial(sklearn_training.run_training, model="random_forest", family="forest", estimator=RandomForestRegressor)
