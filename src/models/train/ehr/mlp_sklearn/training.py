"""Train the configured EHR scikit-learn MLP candidates."""

from functools import partial

from sklearn.neural_network import MLPRegressor

from src.models.train.ehr import sklearn_training

fit_fold = partial(sklearn_training.fit_fold, estimator=MLPRegressor, family="mlp")
run_training = partial(sklearn_training.run_training, model="mlp_sklearn", family="mlp", estimator=MLPRegressor)
