"""Compare selected fixed-test fusion candidates and their controls."""

import numpy as np
from threadpoolctl import threadpool_limits

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.bootstrap import interval, patient_weights
from src.models.assessment.comparison import BASELINE, RULES, calculate_comparison, write_comparison
from src.models.assessment.fusion.comparison import paired_errors
from src.models.assessment.operating_points import fixed_alerts, measure_operating_point
from src.models.ehr_data import load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import completed_result, experiment_store
from src.models.released_data import ReleasedData
from src.models.test.comparison import load_results
from src.models.test.evaluation import output
from src.models.test.fusion.workflow import BASE_MODELS, CANDIDATES, TEST_HISTORY, binding, checked_manifest, fusion_scores


def select_evaluations(layout, evaluation_ids):
    expected = {(family, model): family for family, model in BASE_MODELS.items()}
    expected.update({("fusion", model): name for model, name in CANDIDATES.items()})
    selected = {}
    for entry in evaluation_ids:
        parts = entry.split("/")
        require(len(parts) == 3 and tuple(parts[:2]) in expected,
                "INVALID_TEST_EVALUATION", f"Unknown evaluation: {entry}")
        family, model, key = parts
        name = expected[family, model]
        require(name not in selected, "INVALID_TEST_EVALUATION", f"Multiple evaluations selected for {family}/{model}")
        store = experiment_store(layout, "test", family, model, key)
        manifest = store.validate(store.manifest_ref("test", "evaluate"))
        require(manifest["scope"] == store.scope and (manifest["stage"], manifest["phase"]) == ("test", "evaluate"),
                "INVALID_TEST_EVALUATION", f"Expected a completed fixed-test evaluation: {entry}")
        selected[name] = (name, store, manifest)
    require(selected.keys() == set(expected.values()), "INVALID_TEST_EVALUATION",
            "Select one evaluation each for CatBoost, ResNet50, F1 and F3b")
    return selected


def load_comparison_scores(layout, release, selected):
    package = load_package(ReleasedData.load(layout, release))
    base_sources = [selected[family] for family in BASE_MODELS]
    base_inputs = [binding(manifest) for _, _, manifest in base_sources]
    rows, scores, points = load_results(package, base_sources)
    _, candidates, candidate_points = load_results(
        package, [selected[name] for name in CANDIDATES.values()], additional_inputs=base_inputs)
    require(candidate_points[BASELINE] == points[BASELINE], "OPERATING_POINT_MISMATCH",
            "Fusion and base evaluations must use the same baseline")
    for name in CANDIDATES.values():
        _, store, _ = selected[name]
        refit = checked_manifest(store, "test", "refit", release)
        summary = read_json(output(store, refit, "fit_summary.json"))
        require(summary["candidate"] == name
                and summary["base_refits"] == {family: manifest["inputs"][0] for family, _, manifest in base_sources},
                "INVALID_TEST_EVALUATION", f"Frozen fusion sources differ from the selected base evaluations: {name}")
        if name == "f1":
            decision_margins = summary["decision_margins"]
        replay = fusion_scores(summary, scores)
        require(np.array_equal(replay[name], candidates[name]), "INVALID_TEST_PREDICTIONS",
                f"Published predictions differ from the frozen fusion formula: {name}")
        scores.update(replay)
        points[name] = candidate_points[name]
        thresholds = read_json(output(store, refit, "thresholds.json"))
        for control in summary["mechanism_controls"]:
            points[control] = {
                rule: {ep: measure_operating_point(rows[ep].to_numpy(bool), scores[control], thresholds[rule][control][ep])
                       for ep in ENDPOINTS} for rule in RULES
            }
    return rows, scores, points, decision_margins


def run_comparison(layout, key, *, evaluation_ids, draws=5000, seed=42, progress=None):
    identifier(key)
    require(type(draws) is int and draws > 0 and type(seed) is int and seed >= 0,
            "INVALID_BOOTSTRAP_SETTINGS", "Use positive draws and a nonnegative seed")
    if progress:
        progress("Validating selected fixed-test evaluations")
    selected = select_evaluations(layout, evaluation_ids)
    release = selected["ehr"][2]["details"]["data_release"]
    require(all(manifest["details"]["data_release"] == release for _, _, manifest in selected.values()),
            "TEST_DATA_MISMATCH", "Fixed-test comparisons require a common frozen data release")
    inputs = [binding(manifest) for _, _, manifest in selected.values()]
    snapshot = {"id": key, "data_release": release,
                "parameters": {"evaluation_ids": list(evaluation_ids), "bootstrap_draws": draws, "bootstrap_seed": seed}}
    store = experiment_store(layout, "test", "fusion", "comparison", key)
    ref = store.manifest_ref("comparison", "run")
    existing = completed_result(store, ref, snapshot, inputs)
    if existing:
        return existing

    if progress:
        progress("Checking fixed-test scores against the frozen fusion rules and thresholds")
    rows, scores, points, decision_margins = load_comparison_scores(layout, release, selected)
    details = {"data_release": release, "evaluation_ids": list(evaluation_ids), "prior_fixed_test_exposure": TEST_HISTORY}
    with store.publish("comparison", "run", inputs=[item["manifest"] for item in inputs], details=details) as work:
        require(all(binding(read_json(layout.resolve(item["manifest"]))) == item for item in inputs),
                "UPSTREAM_REBUILT", "An evaluation changed while preparing the comparison")
        with threadpool_limits(limits=1):
            free, based, targets = calculate_comparison(rows, scores, points, draws, seed,
                                                         progress=progress, fixed_test=True)
            weights = np.vstack([np.ones(len(rows)), patient_weights(rows.subject_id.to_numpy(), draws, seed)])
            for groups in based.values():
                for name, endpoints in groups.items():
                    for ep, entry in endpoints.items():
                        alerts = fixed_alerts(rows[ep].to_numpy(bool), scores[name], entry["threshold"])
                        rate = (weights @ alerts) / weights.sum(axis=1)
                        entry["alert_rate"] = interval(rate[0], rate[1:])
            write_json(work / "paired_differences.json", paired_errors(rows, scores, weights))
        write_comparison(work, rows, free, based, targets, snapshot, progress=progress, cohort="Exploratory fixed test")
        report = read_json(work / "summary.json")
        report.update(prior_fixed_test_exposure=TEST_HISTORY, candidates=list(CANDIDATES.values()),
                      mechanism_controls=["ehr_shrink", "f3a", "ehr_bias"], decision_margins=decision_margins)
        write_json(work / "summary.json", report)
    return {"id": key, "status": "complete", "manifest": ref}
