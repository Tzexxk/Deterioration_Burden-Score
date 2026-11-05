"""Select OOF operating points and measure them on a fixed cohort."""

import numpy as np
from sklearn.metrics import roc_curve

from src.common.paths import require
from src.models.assessment.bootstrap import threshold_metrics


def rule_folder(rule, target):
    return f"{rule}_{100 * target:.12g}".replace(".", "p")


def choose_threshold(labels, scores, rule, target):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    feasible = (tpr >= target) if rule == "sensitivity" else (1 - fpr >= target)
    values = thresholds[feasible & np.isfinite(thresholds)]
    if not len(values):
        require(rule == "specificity", "THRESHOLD_UNAVAILABLE", "No threshold meets the sensitivity target")
        return None
    return float(values.max() if rule == "sensitivity" else values.min())


def fixed_alerts(labels, scores, threshold):
    return scores >= threshold if threshold is not None else np.zeros_like(labels)


def operating_point(labels, scores, rule, target):
    threshold = choose_threshold(labels, scores, rule, target)
    alerts = fixed_alerts(labels, scores, threshold)
    metrics, counts = threshold_metrics(labels, alerts, np.ones((1, len(labels))))
    return {"target": target, "threshold": threshold, "comparison": "score >= threshold" if threshold is not None else "no_alerts",
            "confusion_matrix": {k: int(v[0]) for k, v in counts.items()},
            "metrics": {k: float(v[0]) if np.isfinite(v[0]) else None for k, v in metrics.items()}}
