"""Validate the CatBoost v1 protocol carried over from the original EHR workflow."""

import math

from src.common.paths import identifier
from src.common.paths import require as require_field


def require(condition, message):
    require_field(condition, "INVALID_EHR_PROTOCOL", message)


def _integer(value, name, minimum=1):
    require(type(value) is int and value >= minimum, f"Invalid integer: {name}")


def validate_config(cfg):
    expected_keys = {
        "config_version", "protocol_id", "experiment_status", "data_release",
        "seed", "model", "candidates", "preprocessing", "selection",
        "postprocessing", "final_model", "evaluation",
    }
    require(set(cfg) == expected_keys, "Unknown or missing configuration fields")
    require(cfg["config_version"] == "ehr_modeling_v1", "Unsupported config version")
    identifier(cfg["protocol_id"])
    require(cfg["experiment_status"] == "independent_rebuild_after_prior_test_exposure",
            "Prior test exposure must remain disclosed")
    _integer(cfg["seed"], "seed", 0)
    model = cfg["model"]
    fixed_model = {"family": "CatBoostRegressor", "task_type": "CPU",
                   "loss_function": "RMSE", "eval_metric": "RMSE",
                   "use_best_model": True, "allow_writing_files": False, "verbose": False}
    require(set(model) == set(fixed_model) | {
        "iterations", "learning_rate", "early_stopping_rounds", "thread_count"},
        "Unsupported model parameters")
    require(all(model[k] == v for k, v in fixed_model.items()), "Model policy changed")
    for key in ("iterations", "early_stopping_rounds", "thread_count"):
        _integer(model[key], key)
    require(isinstance(model["learning_rate"], (int, float)) and
            math.isfinite(model["learning_rate"]) and 0 < model["learning_rate"] <= 1,
            "Invalid learning rate")
    candidates = cfg["candidates"]
    require(isinstance(candidates, list) and bool(candidates), "No candidates")
    names = []
    for c in candidates:
        require(set(c) == {"name", "depth", "l2_leaf_reg"}, "Unsupported candidate")
        names.append(identifier(c["name"]))
        _integer(c["depth"], "depth")
        require(c["depth"] <= 16, "CatBoost depth exceeds 16")
        require(type(c["l2_leaf_reg"]) in (int, float) and
                math.isfinite(c["l2_leaf_reg"]) and c["l2_leaf_reg"] >= 0, "Invalid L2")
    require(len(names) == len(set(names)), "Duplicate candidate names")
    policies = {
        "preprocessing": {"feature_order": "schema_numeric_then_categorical",
                          "numeric_missing": "catboost_native_nan",
                          "categorical_missing": "__MISSING__",
                          "unknown_categories": "catboost_native_string_hashing"},
        "selection": {"prediction_column": "predicted_score",
                      "primary_metric": "pooled_oof_rmse",
                      "tie_breakers": ["pooled_oof_mae", "depth_ascending",
                                       "l2_leaf_reg_descending"]},
        "postprocessing": {"raw_column": "predicted_score",
                           "processed_column": "postprocessed_score",
                           "clip_range": [0.0, 8.0], "role": "secondary_reporting_only"},
        "final_model": {"scheme": "single_model_refit_on_all_development",
                        "round_rule": "median_best_rounds_half_up",
                        "use_best_model": False, "early_stopping_rounds": None},
    }
    for name, expected in policies.items():
        require(cfg[name] == expected, f"Unsupported {name} policy")
    ev = cfg["evaluation"]
    fixed_ev = {
        "endpoints": ["phys_death", "high_burden"],
        "auprc_method": "average_precision_score",
        "threshold_prediction_column": "predicted_score",
        "threshold_rule": "highest_finite_oof_threshold_meeting_sensitivity",
        "bootstrap_unit": "subject_id", "confidence_interval": "percentile",
        "single_class_bootstrap": "skip_auroc_and_ap_record_valid_and_skipped",
    }
    require(set(ev) == set(fixed_ev) | {
        "sensitivity_target", "bootstrap_draws", "bootstrap_seed", "confidence_level"},
        "Unsupported evaluation parameters")
    require(all(ev[k] == v for k, v in fixed_ev.items()), "Evaluation policy changed")
    for key in ("sensitivity_target", "confidence_level"):
        require(type(ev[key]) in (int, float) and 0 < ev[key] < 1, f"Invalid {key}")
    _integer(ev["bootstrap_draws"], "bootstrap_draws")
    _integer(ev["bootstrap_seed"], "bootstrap_seed", 0)
    return cfg



def validate_training_config(experiment):
    parameters = dict(experiment["parameters"])
    candidate = parameters.pop("candidate", None)
    require(isinstance(candidate, dict), "A CatBoost candidate object is required")
    return validate_config({**parameters, "data_release": experiment["data_release"], "candidates": [candidate]})
