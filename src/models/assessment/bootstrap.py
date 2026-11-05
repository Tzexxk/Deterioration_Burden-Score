"""Paired patient resampling and percentile intervals for study-level predictions."""

import numpy as np


def patient_weights(patient_ids, draws, seed):
    patients, groups = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(len(patients), size=len(patients)),
                                  minlength=len(patients)) for _ in range(draws)])
    return counts[:, groups].astype(float)


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan),
                     where=np.asarray(denominator) > 0)


def rank_metrics(labels, scores, weights):
    # Group equal scores before accumulating mass so ties match sklearn's ROC AUC and AP.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    mass = weights[:, order]
    positive = np.add.reduceat(mass * labels[order], starts, axis=1)
    negative = np.add.reduceat(mass * (~labels[order]), starts, axis=1)
    tp, fp = positive.cumsum(axis=1), negative.cumsum(axis=1)
    n_pos, n_neg = tp[:, -1], fp[:, -1]
    roc = ratio((positive * (n_neg[:, None] - fp + negative / 2)).sum(axis=1), n_pos * n_neg)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = ratio((positive * precision).sum(axis=1), n_pos)
    ap[n_neg == 0] = np.nan
    return roc, ap


def threshold_metrics(labels, alerts, weights):
    tp = weights @ (labels & alerts)
    fp = weights @ (~labels & alerts)
    fn = weights @ (labels & ~alerts)
    tn = weights @ (~labels & ~alerts)
    return ({"sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
             "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn)},
            {"tp": tp, "fp": fp, "fn": fn, "tn": tn})


def interval(point, draws):
    valid = draws[np.isfinite(draws)]
    lower, upper = np.percentile(valid, [2.5, 97.5]) if valid.size else (None, None)
    return {"point": float(point) if np.isfinite(point) else None,
            "lower": float(lower) if lower is not None else None,
            "upper": float(upper) if upper is not None else None,
            "requested_draws": len(draws), "valid_draws": len(valid), "skipped_draws": len(draws) - len(valid)}
