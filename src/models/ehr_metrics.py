"""Regression and endpoint metrics shared by EHR assessment and testing."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.acceptance_checks import require

ENDPOINTS = ("phys_death", "high_burden")


def point_metrics(rows: pd.DataFrame, predicted, weights=None) -> dict:
    y = rows.burden_score.to_numpy(dtype=float)
    p = np.asarray(predicted, dtype=float)
    require(p.shape == y.shape and len(p) > 0 and np.isfinite(p).all(),
            "Invalid/empty continuous predictions")
    error = p - y
    result = {"mae": float(np.average(np.abs(error), weights=weights)),
              "rmse": float(np.sqrt(np.average(error ** 2, weights=weights)))}
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    for endpoint in ENDPOINTS:
        z = rows[endpoint].to_numpy(dtype=int)
        both = w[z == 1].sum() > 0 and w[z == 0].sum() > 0
        result[endpoint + "_auroc"] = (
            float(roc_auc_score(z, p, sample_weight=weights)) if both else None)
        result[endpoint + "_ap"] = (
            float(average_precision_score(z, p, sample_weight=weights)) if both else None)
    return result
