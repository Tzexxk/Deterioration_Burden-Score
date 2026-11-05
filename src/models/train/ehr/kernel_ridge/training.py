"""Fit RBF Kernel Ridge candidates on the five development folds."""

from importlib.metadata import version

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_fold import fold_interface, read_fold_interface
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.experiments import read_batch
from src.models.train.ehr.training import train_model

FIXED_PARAMETERS = {"coef0": 1, "degree": 3, "gamma": 1.0, "kernel": "rbf", "kernel_params": None}


def validate_experiment(experiment):
    params = experiment["parameters"]
    require(set(params) == {"candidate"} and isinstance(params["candidate"], dict),
            "INVALID_KERNEL_RIDGE_CONFIG", "A Kernel Ridge candidate object is required")
    candidate = params["candidate"]
    require(set(candidate) == {"name", "family", "params"} and candidate["family"] == "kernel"
            and isinstance(candidate["params"], dict), "INVALID_KERNEL_RIDGE_CONFIG", "Invalid Kernel Ridge candidate")
    settings = candidate["params"]
    require(set(settings) == FIXED_PARAMETERS.keys() | {"alpha"}
            and all(settings[name] == value for name, value in FIXED_PARAMETERS.items())
            and type(settings["alpha"]) in (int, float) and np.isfinite(settings["alpha"]) and settings["alpha"] > 0,
            "INVALID_KERNEL_RIDGE_CONFIG", "Use the RBF protocol with positive finite alpha and configured gamma 1.0")


def predict_checkpoint(checkpoint, training_features, validation_features):
    train, validation = (transform_features(features, checkpoint["preprocessing"])
                         for features in (training_features, validation_features))
    coefficients = np.asarray(checkpoint["dual_coefficients"], dtype=float)
    gamma, offset = checkpoint["gamma"], checkpoint["target_mean"]
    require(coefficients.shape == (len(train),) and np.isfinite(coefficients).all()
            and gamma == 1 / train.shape[1] and np.isfinite(offset),
            "INVALID_CHECKPOINT", "Invalid Kernel Ridge coefficients, gamma or target mean")
    with threadpool_limits(limits=1):
        predicted = rbf_kernel(validation, train, gamma=gamma) @ coefficients + offset
    require(np.isfinite(predicted).all(), "INVALID_CHECKPOINT", "Checkpoint predictions must be finite")
    return predicted


def fit_fold(package, experiment, fold, folder):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y = package.samples.iloc[train].burden_score.to_numpy(float)
    offset = float(y.mean())
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        values = transform_features(x_train, state)
        # The historical protocol scales gamma by the fold's transformed feature count.
        settings = {**experiment["parameters"]["candidate"]["params"], "gamma": 1 / values.shape[1]}
        model = KernelRidge(**settings).fit(values, y - offset)
        expected = model.predict(transform_features(x_validation, state)) + offset
    folder.mkdir()
    path = folder / "model.json"
    write_json(path, {"preprocessing": state, "dual_coefficients": model.dual_coef_.tolist(),
                      "gamma": settings["gamma"], "target_mean": offset})
    np.testing.assert_allclose(predict_checkpoint(read_json(path), x_train, x_validation), expected, rtol=0, atol=1e-12)
    write_json(folder / "interface.json", fold_interface(package, fold, settings))
    return {"fold": fold["fold"], "fit_method": "single_fit"}


def checkpoint_prediction(store, manifest, package, fold):
    interface = read_fold_interface(store, manifest, package, fold)
    ref = store.output_ref(manifest, f"fold_{fold['fold']}/model.json")
    checkpoint = read_json(store.layout.resolve(ref))
    train, validation = fold["train_indices"], fold["validation_indices"]
    require(checkpoint["gamma"] == interface["parameters"]["gamma"]
            and checkpoint["target_mean"] == package.samples.iloc[train].burden_score.to_numpy(float).mean(),
            "INVALID_CHECKPOINT", "Kernel Ridge gamma or target mean differs from its training fold")
    return predict_checkpoint(checkpoint, package.features.iloc[train], package.features.iloc[validation]), None


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("scikit-learn") == "1.9.1", "INVALID_SKLEARN_VERSION", "Use scikit-learn 1.9.1 for this protocol")
    return train_model(layout, config_path, experiments, "kernel_ridge", fit_fold)
