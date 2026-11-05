"""Train and reload CPU XGBoost candidates with fold validation early stopping."""

from importlib.metadata import version

import numpy as np
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_fold import fold_interface, read_fold_interface
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.experiments import read_batch
from src.models.train.ehr.training import candidate_settings, train_model


def predict(model, values, rounds):
    with threadpool_limits(limits=1):
        scores = model.predict(values, iteration_range=(0, rounds)).astype(float)
    require(scores.shape == (len(values),) and np.isfinite(scores).all(),
            "INVALID_CHECKPOINT", "XGBoost predictions must be finite and match the fold")
    return scores


def fit_fold(package, experiment, fold, folder):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y, y_validation = (package.samples.iloc[indices].burden_score.to_numpy(float) for indices in (train, validation))
    settings = candidate_settings(experiment, "xgb")
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        train_values, values = (transform_features(x, state) for x in (x_train, x_validation))
        model = XGBRegressor(**{**settings, "missing": np.nan})
        model.fit(train_values, y, eval_set=[(train_values, y), (values, y_validation)], verbose=False)
    rounds = int(model.best_iteration) + 1
    expected = predict(model, values, rounds)
    folder.mkdir()
    path = folder / "model.ubj"
    model.save_model(path)
    loaded = XGBRegressor(n_jobs=1)
    loaded.load_model(path)
    np.testing.assert_allclose(predict(loaded, values, rounds), expected, rtol=0, atol=1e-12)
    write_json(folder / "preprocessing.json", state)
    write_json(folder / "history.json", model.evals_result())
    write_json(folder / "interface.json", {**fold_interface(package, fold, settings),
               "xgboost_version": version("xgboost"), "best_round": rounds})
    return {"fold": fold["fold"], "fit_method": "validation_early_stopping", "best_round": rounds}


def checkpoint_prediction(store, manifest, package, fold):
    interface = read_fold_interface(store, manifest, package, fold)
    require(interface["xgboost_version"] == version("xgboost"), "INVALID_CHECKPOINT", "XGBoost version differs")

    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    model = XGBRegressor(n_jobs=1)
    model.load_model(output("model.ubj"))
    rounds = interface["best_round"]
    values = transform_features(package.features.iloc[fold["validation_indices"]], read_json(output("preprocessing.json")))
    require(model.n_features_in_ == values.shape[1] and rounds == int(model.best_iteration) + 1,
            "INVALID_CHECKPOINT", "XGBoost model width or best round differs from its interface")
    history = read_json(output("history.json"))
    learn, valid = (np.asarray(history[part]["rmse"], dtype=float) for part in ("validation_0", "validation_1"))
    require(learn.shape == valid.shape and learn.ndim == 1 and learn.size > 0
            and np.isfinite(learn).all() and np.isfinite(valid).all()
            and rounds == int(np.argmin(valid)) + 1 and len(valid) == model.get_booster().num_boosted_rounds(),
            "INVALID_TRAINING_HISTORY", "RMSE history differs from the XGBoost checkpoint")
    return predict(model, values, rounds), {"fold": fold["fold"], "best_round": rounds,
                                            "learn": learn, "validation": valid}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    required = {"objective": "reg:squarederror", "eval_metric": "rmse", "missing": "NaN", "device": "cpu", "n_jobs": 1}
    known = XGBRegressor().get_params().keys()
    for experiment in experiments:
        settings = candidate_settings(experiment, "xgb")
        require(set(settings) <= known and all(settings.get(key) == value for key, value in required.items())
                and type(settings.get("early_stopping_rounds")) is int and settings["early_stopping_rounds"] > 0
                and settings.get("callbacks") is None,
                "INVALID_XGBOOST_CONFIG", "Use CPU XGBoost with RMSE validation and positive early stopping")
    require(version("xgboost") == "3.2.0" and version("scikit-learn") == "1.9.1",
            "INVALID_MODEL_RUNTIME", "Use XGBoost 3.2.0 and scikit-learn 1.9.1")
    return train_model(layout, config_path, experiments, "xgboost", fit_fold)
