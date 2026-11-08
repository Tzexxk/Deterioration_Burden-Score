"""Shared fixed-test output and development threshold freezing."""

import numpy as np

from src.common.manifest import write_json
from src.models.assessment.comparison import BASELINE, RULES
from src.models.assessment.evaluation import metric_pair
from src.models.assessment.operating_points import (
    measure_operating_point,
    operating_point,
    rule_folder,
)
from src.models.assessment.plots import confusion_plot
from src.models.ehr_metrics import ENDPOINTS

SAMPLE_COLUMNS = ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]
POINT_FIELDS = ("target", "threshold", "comparison")


def output(store, manifest, name):
    return store.layout.resolve(store.output_ref(manifest, name))


def frozen_thresholds(development, mean, points):
    baseline = np.full(len(development), mean)
    thresholds = {rule: {"model": {}, BASELINE: {}} for rule in RULES}
    for rule in RULES:
        for endpoint in ENDPOINTS:
            point = points[rule]["model"][endpoint]
            thresholds[rule]["model"][endpoint] = {field: point[field] for field in POINT_FIELDS}
            reference = operating_point(development[endpoint].to_numpy(bool), baseline, rule, point["target"])
            thresholds[rule][BASELINE][endpoint] = {field: reference[field] for field in POINT_FIELDS}
    return thresholds


def write_fixed_evaluation(work, config, rows, raw, mean, thresholds):
    scores = {"model": raw, BASELINE: np.full(len(rows), mean)}
    predictions = rows.assign(predicted_score=raw, postprocessed_score=np.clip(raw, 0, 8),
                              baseline_mean_score=scores[BASELINE])
    predictions.to_parquet(work / "test_predictions.parquet", index=False)
    write_json(work / "effective_configuration.json", config)
    write_json(work / "summary.json", {"prediction_role": "fixed_test", "studies": len(rows),
               "patients": int(rows.subject_id.nunique()),
               "positive_studies": {name: int(rows[name].sum()) for name in ENDPOINTS}, "baseline_mean": mean})
    write_json(work / "metrics.json", {name: metric_pair(rows, values) for name, values in scores.items()})
    points = {}
    for rule in RULES:
        values = {name: {ep: measure_operating_point(rows[ep].to_numpy(bool), scores[name], point)
                         for ep, point in endpoints.items()} for name, endpoints in thresholds[rule].items()}
        target = values["model"][ENDPOINTS[0]]["target"]
        folder = work / "threshold_based" / rule_folder(rule, target)
        folder.mkdir(parents=True)
        confusion_plot(values, folder / "confusion_matrices.svg",
                       title=f"Fixed test · development {rule} target {target:.0%}")
        points[rule] = values
    write_json(work / "operating_points.json", points)
