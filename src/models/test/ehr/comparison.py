"""Compare completed EHR fixed-test evaluations using shared patient draws."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.common.paths import identifier, read_json, require
from src.models.assessment.ehr.comparison import (
    BASELINE,
    RULES,
    calculate_comparison,
    write_comparison,
)
from src.models.assessment.ehr.evaluation import metric_pair
from src.models.ehr_data import checked_input_summary, load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData
from src.models.test.ehr.workflow import SAMPLE_COLUMNS, output


def discover(layout, progress):
    root = layout.path("outputs", "models/test/ehr")
    paths = sorted(root.glob("*/*/steps/test/evaluate/manifest.json"))
    require(paths, "NO_TEST_EVALUATIONS", "Run fixed-test evaluate for each selected candidate first")
    if progress:
        progress(f"Validating test evaluations: 0/{len(paths)}")
    sources = []
    for index, path in enumerate(paths, 1):
        model, key = path.relative_to(root).parts[:2]
        store = experiment_store(layout, "test", "ehr", model, key)
        manifest = store.validate(store.manifest_ref("test", "evaluate"))
        require(manifest["scope"] == store.scope and (manifest["stage"], manifest["phase"]) == ("test", "evaluate"),
                "INVALID_TEST_EVALUATION", "Expected a completed fixed-test evaluation")
        sources.append((f"{model}/{key}", store, manifest))
        if progress:
            progress(f"Validating test evaluations: {index}/{len(paths)} ({model}/{key})")
    return sources


def load_results(package, sources):
    rows = package.samples.iloc[package.cv["test_indices"]][SAMPLE_COLUMNS].reset_index(drop=True)
    development = package.samples.iloc[package.cv["development_indices"]]
    mean = float(development.burden_score.to_numpy(float).mean())
    scores, points = {BASELINE: np.full(len(rows), mean)}, {}
    for name, store, manifest in sources:
        refit_ref = store.manifest_ref("test", "refit")
        refit = read_json(store.layout.resolve(refit_ref))
        require(refit["scope"] == store.scope
                and manifest["inputs"] == [{"manifest": refit_ref, "generation": refit["generation"]}],
                "INVALID_TEST_EVALUATION", f"Fixed-test evaluation must bind its refit: {name}")
        frozen = read_json(output(store, refit, "thresholds.json"))
        actual = pd.read_parquet(output(store, manifest, "test_predictions.parquet"))
        require(actual.sample_id.is_unique and all(np.array_equal(actual[key], rows[key]) for key in SAMPLE_COLUMNS)
                and np.array_equal(actual.baseline_mean_score, scores[BASELINE]),
                "INVALID_TEST_COVERAGE", f"Fixed-test rows or development mean baseline differ: {name}")
        values = actual.predicted_score.to_numpy(float)
        require(np.isfinite(values).all() and np.array_equal(actual.postprocessed_score, np.clip(values, 0, 8)),
                "INVALID_TEST_PREDICTIONS", f"Fixed-test scores or clipping differ: {name}")
        scores[name] = values
        metrics = read_json(output(store, manifest, "metrics.json"))
        require(metrics == {"model": metric_pair(rows, values), BASELINE: metric_pair(rows, scores[BASELINE])},
                "INVALID_TEST_METRICS", f"Published test metrics disagree with predictions: {name}")
        stored = read_json(output(store, manifest, "operating_points.json"))
        for rule in RULES:
            for group in ("model", BASELINE):
                for ep in ENDPOINTS:
                    point = stored[rule][group][ep]
                    require({field: point[field] for field in ("target", "threshold", "comparison")} == frozen[rule][group][ep],
                            "OPERATING_POINT_MISMATCH", f"Test thresholds differ from the frozen refit: {name}")
        baseline_points = {rule: stored[rule][BASELINE] for rule in RULES}
        points.setdefault(BASELINE, baseline_points)
        require(points[BASELINE] == baseline_points, "OPERATING_POINT_MISMATCH", f"Test baseline differs: {name}")
        points[name] = {rule: stored[rule]["model"] for rule in RULES}
    return rows, scores, points


def run_comparison(layout, key, *, draws=5000, seed=42, progress=None):
    identifier(key)
    sources = discover(layout, progress)
    release = sources[0][2]["details"]["data_release"]
    require(all(manifest["details"]["data_release"] == release for _, _, manifest in sources),
            "TEST_DATA_MISMATCH", "Fixed-test comparisons require a common frozen data release")
    if progress:
        progress("Loading fixed-test predictions and frozen operating points")
    package = load_package(ReleasedData.load(layout, release))
    checked_input_summary(package)
    rows, scores, points = load_results(package, sources)
    inputs = [{"manifest": manifest["self"], "generation": manifest["generation"]} for _, _, manifest in sources]
    ids = [name for name, _, _ in sources]
    snapshot = {"id": key, "data_release": release,
                "parameters": {"evaluation_ids": ids, "bootstrap_draws": draws, "bootstrap_seed": seed}}
    store = experiment_store(layout, "test", "ehr", "comparison", key)
    ref = store.manifest_ref("comparison", "run")
    existing = completed_result(store, ref, snapshot, inputs)
    if existing:
        return existing

    def input_progress(stage, index):
        if progress:
            progress(f"{stage.replace('assessments', 'test evaluations')}: {index}/{len(inputs)}")

    input_progress("Binding assessments", 0)
    with store.publish("comparison", "run", inputs=[item["manifest"] for item in inputs],
                       input_progress=input_progress, details={"data_release": release, "evaluation_ids": ids}) as work:
        with threadpool_limits(limits=1):
            free, based, targets = calculate_comparison(rows, scores, points, draws, seed, progress=progress, fixed_test=True)
        write_comparison(work, rows, free, based, targets, snapshot, progress=progress, cohort="Fixed test")
        input_progress("Rechecking assessments", 0)
    return {"id": key, "status": "complete", "manifest": ref}
