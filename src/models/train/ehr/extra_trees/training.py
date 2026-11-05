"""Train the configured EHR Extra Trees candidates."""

from functools import partial

from sklearn.ensemble import ExtraTreesRegressor

from src.models.train.ehr import sklearn_training

fit_fold = partial(sklearn_training.fit_fold, estimator=ExtraTreesRegressor, family="extra")
run_training = partial(sklearn_training.run_training, model="extra_trees", family="extra", estimator=ExtraTreesRegressor)
