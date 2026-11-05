"""Deterministic, generation-bound patient-level train/validation/test allocation."""

import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import freeze_cohort
from src.cohort.burden_io import COMPONENTS, open_store
from src.cohort.burden_support import AuditWorkspace, write_csv
from src.cohort.burden_validation import _reject
from src.common.manifest import ManifestStore, write_json
from src.common.paths import read_json, require

SPLITS = ("train", "validation", "test")

ALGORITHM = "patient_stratified_pcg64_v1"

SCHEMA = pa.schema(
    [
        ("subject_id", pa.int64()),
        ("split", pa.string()),
        ("seed", pa.int64()),
        ("split_version", pa.string()),
        ("cohort_version", pa.string()),
        ("study_count", pa.int64()),
        ("max_burden_score", pa.float32()),
        ("ever_phys_death", pa.uint8()),
        ("ever_high_burden", pa.uint8()),
    ]
)

TABLE_NAME = "patient_split.parquet"
CV_ALGORITHM = "patient_grouped_study_balanced_cv_v1"
CV_INDICES = "cross_validation_indices.json"
SPLIT_METHODS = {"holdout": ALGORITHM, "five-fold": CV_ALGORITHM}

def apportion(
    n: int, weights: list[int], minimum: list[int], capacity: list[int]
) -> list[int]:
    """Bounded integer largest-deficit rounding; ties follow SPLITS order."""
    if (
        type(n) is not int
        or n < 0
        or any(len(v) != 3 for v in (weights, minimum, capacity))
        or any(type(x) is not int for v in (weights, minimum, capacity) for x in v)
        or any(w <= 0 for w in weights)
        or sum(weights) != 100
        or any(lo < 0 or cap < lo for lo, cap in zip(minimum, capacity))
        or sum(minimum) > n
        or sum(capacity) < n
    ):
        raise ValueError("Patient allocation constraints cannot be satisfied")
    allocation = [
        min(cap, max(lo, n * w // 100))
        for w, lo, cap in zip(weights, minimum, capacity)
    ]
    while sum(allocation) < n:
        group = min(
            (i for i in range(3) if allocation[i] < capacity[i]),
            key=lambda i: (100 * allocation[i] - n * weights[i], i),
        )
        allocation[group] += 1
    while sum(allocation) > n:
        group = min(
            (i for i in range(3) if allocation[i] > minimum[i]),
            key=lambda i: (n * weights[i] - 100 * allocation[i], i),
        )
        allocation[group] -= 1
    return allocation

def aggregate_patients(cohort: pa.Table, version: str) -> list[dict]:
    required = {
        "sample_id": pa.string(),
        "subject_id": pa.int64(),
        "hadm_id": pa.int64(),
        "study_id": pa.int64(),
        "burden_score": pa.float32(),
        "phys_death": pa.uint8(),
        "high_burden": pa.uint8(),
        "cohort_version": pa.string(),
        **dict.fromkeys(COMPONENTS, pa.uint8()),
    }
    if len(set(cohort.column_names)) != len(cohort.column_names) or any(
        k not in cohort.column_names
        or cohort.schema.field(k).type != v
        or cohort[k].null_count
        for k, v in required.items()
    ):
        raise ValueError("Missing, null or invalid common-cohort fields")
    patients, samples, studies = {}, set(), set()
    for row in cohort.to_pylist():
        key = (row["subject_id"], row["hadm_id"], row["study_id"])
        if (
            any(v <= 0 for v in key)
            or row["sample_id"] != f"p{key[0]}_h{key[1]}_s{key[2]}"
            or row["sample_id"] in samples
            or key in studies
            or row["cohort_version"] != version
        ):
            raise ValueError("Invalid, duplicated or mixed-version study identity")
        flags = [row[c] for c in COMPONENTS]
        if any(f not in (0, 1) for f in flags):
            raise ValueError("Invalid event flag in common cohort")
        t, r, c, d = max(flags[:2]), max(flags[2:4]), flags[4], flags[5]
        score = t + 2 * r + 2 * c + 3 * d
        if (
            row["burden_score"] != score
            or row["phys_death"] != int(r == 1 or c == 1 or d == 1)
            or row["high_burden"] != int(score >= 3)
        ):
            raise ValueError("Invalid common-cohort score or derived endpoint")
        samples.add(row["sample_id"])
        studies.add(key)
        patient = patients.setdefault(
            key[0],
            {
                "subject_id": key[0],
                "study_count": 0,
                "max_burden_score": 0.0,
                "ever_phys_death": 0,
                "ever_high_burden": 0,
            },
        )
        patient["study_count"] += 1
        patient["max_burden_score"] = max(patient["max_burden_score"], score)
        for flag in ("phys_death", "high_burden"):
            patient["ever_" + flag] = max(patient["ever_" + flag], row[flag])
    if not patients:
        raise ValueError("The common cohort is empty")
    return [patients[key] for key in sorted(patients)]

def allocate_patients(cohort: pa.Table, settings: dict) -> tuple[pa.Table, dict]:
    patients = aggregate_patients(cohort, settings["cohort_version"])
    n = len(patients)
    positives = sum(p["ever_phys_death"] for p in patients)
    high = sum(p["ever_high_burden"] for p in patients)
    if high < 3:
        raise ValueError(
            "At least three high-burden patients are required; do not change the seed or split patients"
        )
    total = apportion(n, settings["weights"], [1] * 3, [n] * 3)
    phys = apportion(positives, settings["weights"], [1] * 3, total)
    high_counts = apportion(high, settings["weights"], [1] * 3, phys)
    strata = (
        ([p for p in patients if p["ever_high_burden"]], high_counts),
        (
            [p for p in patients if p["ever_phys_death"] and not p["ever_high_burden"]],
            [a - b for a, b in zip(phys, high_counts)],
        ),
        (
            [p for p in patients if not p["ever_phys_death"]],
            [a - b for a, b in zip(total, phys)],
        ),
    )
    rng = np.random.Generator(np.random.PCG64(settings["seed"]))
    assignments = []
    for layer, quotas in strata:
        order = rng.permutation(len(layer))
        start = 0
        for split, count in zip(SPLITS, quotas):
            for position in order[start : start + count]:
                assignments.append(
                    {
                        **layer[int(position)],
                        "split": split,
                        "seed": settings["seed"],
                        "split_version": settings["split_version"],
                        "cohort_version": settings["cohort_version"],
                    }
                )
            start += count
        if start != len(layer):
            raise ValueError("Stratum allocation did not cover all patients")
    table = pa.Table.from_pylist(
        sorted(assignments, key=lambda p: p["subject_id"]), schema=SCHEMA
    )
    return table, {
        "total": total,
        "ever_phys_death": phys,
        "ever_high_burden": high_counts,
    }

def select_fold(patients: pa.Table, bundle: dict, fold: int) -> pa.Table:
    """Return a patient split view without copying images or EHR features."""
    if type(fold) is not int or not 1 <= fold <= 5:
        raise ValueError("fold must be an integer from 1 to 5")
    memberships = {r["subject_id"]: r["fold_id"] for r in bundle["master"]}
    rows = patients.to_pylist()
    if len(rows) != len(memberships) or {r["subject_id"] for r in rows} != set(memberships):
        raise ValueError("Fold view patient coverage mismatch")
    for row in rows:
        group = memberships[row["subject_id"]]
        row["split"] = "test" if group == 0 else "validation" if group == fold else "train"
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def allocate_grouped_cv(cohort: pa.Table, settings: dict) -> tuple[pa.Table, dict]:
    """Allocate whole patients; optimize study counts before label balance.

    Identical patient label-count vectors share integer quota variables. Patients
    are never split: seed42 orders the actual people within each vector group.
    """
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    patients = aggregate_patients(cohort, settings["cohort_version"])
    rows = sorted(cohort.to_pylist(), key=lambda r: r["sample_id"])
    by_patient = {p["subject_id"]: [] for p in patients}
    for row in rows:
        by_patient[row["subject_id"]].append(row)
    scores = list(range(9))
    names = ["studies", *[f"score_{s}" for s in scores], "phys_death", "high_burden"]
    def partition(people, fractions):
        grouped = {}
        for patient in people:
            values = by_patient[patient["subject_id"]]
            vector = (len(values), *[sum(r["burden_score"] == s for r in values) for s in scores],
                      *[sum(r[k] for r in values) for k in ("phys_death", "high_burden")])
            grouped.setdefault(vector, []).append(patient["subject_id"])
        vectors = sorted(grouped)
        features = np.asarray(vectors, dtype=float)
        capacity = np.asarray([len(grouped[v]) for v in vectors])
        bins, metrics, groups = len(fractions), len(names), len(vectors)
        fractions = np.asarray(fractions, dtype=float)
        totals = capacity @ features
        variables = groups * bins + metrics * bins
        matrix_rows, matrix_cols, matrix_values, lower, upper = [], [], [], [], []

        def constraint(coefficients, lo, hi):
            row = len(lower)
            for col, value in coefficients.items():
                if value:
                    matrix_rows.append(row)
                    matrix_cols.append(col)
                    matrix_values.append(value)
            lower.append(lo)
            upper.append(hi)

        for g in range(groups):
            constraint({g * bins + b: 1 for b in range(bins)}, capacity[g], capacity[g])
        for b in range(bins):
            for metric in range(metrics):
                coefficients = {g * bins + b: features[g, metric] for g in range(groups)}
                slack = groups * bins + metric * bins + b
                target = totals[metric] * fractions[b]
                constraint({**coefficients, slack: -1}, -np.inf, target)
                constraint({**coefficients, slack: 1}, target, np.inf)
            for metric in (metrics - 2, metrics - 1):
                minimum_groups = 5 if bins == 2 and b == 1 else 1
                constraint({g * bins + b: int(features[g, metric] > 0)
                            for g in range(groups)}, minimum_groups, np.inf)
                constraint({g * bins + b: int(features[g, 0] - features[g, metric] > 0)
                            for g in range(groups)}, minimum_groups, np.inf)
        bounds = Bounds(np.zeros(variables), np.r_[np.repeat(capacity, bins),
                                                  np.full(metrics * bins, np.inf)])
        integrality = np.r_[np.ones(groups * bins), np.zeros(metrics * bins)]

        def solve(objective):
            matrix = coo_matrix((matrix_values, (matrix_rows, matrix_cols)),
                                shape=(len(lower), variables)).tocsc()
            result = milp(objective, integrality=integrality, bounds=bounds,
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={"time_limit": 120.0, "mip_rel_gap": 0.0})
            if not result.success:
                raise ValueError("Grouped CV allocation failed; preserve patient groups and policy: "
                                 + result.message)
            return result

        objective = np.zeros(variables)
        objective[groups * bins:groups * bins + bins] = 1
        first = solve(objective)
        constraint({groups * bins + b: 1 for b in range(bins)}, -np.inf, first.fun + 1e-6)
        rng = np.random.Generator(np.random.PCG64(settings["seed"]))
        objective[:groups * bins] = rng.uniform(0, 1e-8, groups * bins)
        for metric in range(1, metrics):
            objective[groups * bins + metric * bins:groups * bins + (metric + 1) * bins] = (
                (5 if metric >= metrics - 2 else 1) / max(totals[metric], 1)
            )
        result = solve(objective)
        quotas_by_vector = np.rint(result.x[:groups * bins]).astype(int).reshape(groups, bins)
        if np.max(np.abs(quotas_by_vector.ravel() - result.x[:groups * bins])) > 1e-5:
            raise ValueError("Noninteger grouped CV assignment")
        memberships = {}
        for vector, quotas in zip(vectors, quotas_by_vector):
            people = np.asarray(sorted(grouped[vector]))
            people = people[rng.permutation(len(people))]
            start = 0
            for b, count in enumerate(quotas):
                for subject in people[start:start + count]:
                    memberships[int(subject)] = b
                start += count
            if start != len(people):
                raise ValueError("Grouped CV patient coverage failed")
        return memberships

    # Freeze the holdout membership before optimizing development folds.
    holdout = partition(patients, [0.15, 0.85])
    development_patients = [p for p in patients if holdout[p["subject_id"]] == 1]
    development_folds = partition(development_patients, [0.2] * 5)
    memberships = {sid: 0 for sid, group in holdout.items() if group == 0}
    memberships.update({sid: fold + 1 for sid, fold in development_folds.items()})
    fractions = np.asarray([0.15] + [0.17] * 5)
    master = [{"study_index": i, "sample_id": r["sample_id"],
               "subject_id": r["subject_id"], "hadm_id": r["hadm_id"],
               "burden_score": int(r["burden_score"]), "phys_death": r["phys_death"],
               "high_burden": r["high_burden"], "fold_id": memberships[r["subject_id"]]}
              for i, r in enumerate(rows)]
    test = [r["study_index"] for r in master if r["fold_id"] == 0]
    folds = [{"fold": fold,
              "train_indices": [r["study_index"] for r in master if r["fold_id"] not in (0, fold)],
              "validation_indices": [r["study_index"] for r in master if r["fold_id"] == fold]}
             for fold in range(1, 6)]
    bundle = dict(index_base=0, ordering="sample_id ascending", test_fold_id=0,
                  allocation_order="holdout_then_development_folds",
                  primary_view_fold=1, seed=settings["seed"], n_folds=5,
                  algorithm_version=CV_ALGORITHM, solver_runtime=scipy.__version__,
                  cohort_version=settings["cohort_version"], split_version=settings["split_version"],
                  target_study_fractions=fractions.tolist(), master=master,
                  test_indices=test, folds=folds,
                  development_indices=[r["study_index"] for r in master if r["fold_id"] != 0],
                  preprocessing="Fit all learned preprocessing only on each fold training indices; "
                  "refit on development only for the final single model. Test remains untouched.")
    def describe(indices):
        part = [master[i] for i in indices]
        return dict(studies=len(part), patients=len({r["subject_id"] for r in part}),
                    burden_scores={str(s): sum(r["burden_score"] == s for r in part)
                                   for s in range(9)},
                    outcomes={k: dict(positive=sum(r[k] for r in part),
                                      negative=sum(1 - r[k] for r in part),
                                      positive_patients=len({r["subject_id"] for r in part if r[k]}))
                              for k in ("phys_death", "high_burden")})
    bundle["distribution"] = {
        "test": describe(test), "development": describe(bundle["development_indices"]),
        "folds": [{"fold": f["fold"], "train": describe(f["train_indices"]),
                   "validation": describe(f["validation_indices"])} for f in folds],
    }
    bundle["score_group_support"] = {
        str(s): len({r["subject_id"] for r in master if r["burden_score"] == s})
        for s in scores
    }
    assignments = [{**p, "split": "test" if memberships[p["subject_id"]] == 0 else
                    "validation" if memberships[p["subject_id"]] == 1 else "train",
                    "seed": settings["seed"], "split_version": settings["split_version"],
                    "cohort_version": settings["cohort_version"]} for p in patients]
    table = pa.Table.from_pylist(assignments, schema=SCHEMA)
    quotas = {name: [sum(1 if name == "total" else p[name] for p in assignments
                         if p["split"] == part) for part in SPLITS]
              for name in ("total", "ever_phys_death", "ever_high_burden")}
    quotas["cross_validation"] = bundle
    validate_cv_indices(cohort, bundle, table)
    return table, quotas


def validate_cv_indices(cohort: pa.Table, bundle: dict, patients: pa.Table | None = None) -> None:
    """Check index identities and all five group boundaries independently of the solver."""
    rows = sorted(cohort.to_pylist(), key=lambda r: r["sample_id"])
    master = bundle["master"]
    if (bundle["index_base"] != 0 or bundle["n_folds"] != 5 or
            bundle["primary_view_fold"] != 1 or bundle["test_fold_id"] != 0 or
            bundle["ordering"] != "sample_id ascending" or bundle["seed"] != 42 or
            bundle["algorithm_version"] != CV_ALGORITHM or
            bundle["allocation_order"] != "holdout_then_development_folds" or
            {r["cohort_version"] for r in rows} != {bundle["cohort_version"]} or
            len(master) != len(rows) or [r["fold"] for r in bundle["folds"]] != list(range(1, 6))):
        raise ValueError("Invalid cross-validation index contract")
    membership = {}
    for i, (row, actual) in enumerate(zip(rows, master)):
        expected = {k: row[k] for k in ("sample_id", "subject_id", "hadm_id", "burden_score",
                                        "phys_death", "high_burden")}
        if (type(actual["study_index"]) is not int or actual["study_index"] != i
                or any(actual[k] != v for k, v in expected.items())
                or type(actual["fold_id"]) is not int or not 0 <= actual["fold_id"] <= 5):
            raise ValueError("CV master index identity changed")
        subject = actual["subject_id"]
        if subject in membership and membership[subject] != actual["fold_id"]:
            raise ValueError("A patient crosses CV groups")
        membership[subject] = actual["fold_id"]
    def indices(values):
        if (any(type(i) is not int or not 0 <= i < len(rows) for i in values)
                or values != sorted(set(values))):
            raise ValueError("Invalid or duplicate CV index")
        return set(values)
    test = indices(bundle["test_indices"])
    if test != {r["study_index"] for r in master if r["fold_id"] == 0}:
        raise ValueError("Test index membership changed")
    development = set(range(len(rows))) - test
    if indices(bundle["development_indices"]) != development:
        raise ValueError("CV development membership changed")
    validation_seen = set()
    for fold in bundle["folds"]:
        train, validation = indices(fold["train_indices"]), indices(fold["validation_indices"])
        if (validation != {r["study_index"] for r in master if r["fold_id"] == fold["fold"]}
                or train != development - validation or validation_seen & validation):
            raise ValueError("CV train/validation coverage or complement changed")
        validation_seen |= validation
        for part in (train, validation, test):
            for outcome in ("phys_death", "high_burden"):
                if {rows[i][outcome] for i in part} != {0, 1}:
                    raise ValueError("Both binary outcome classes are required in every CV set")
    if validation_seen != development:
        raise ValueError("Every development study must validate exactly once")
    if patients is not None:
        for p in patients.to_pylist():
            if p["split_version"] != bundle["split_version"]:
                raise ValueError("CV split version changed")
            expected = "test" if membership[p["subject_id"]] == 0 else (
                "validation" if membership[p["subject_id"]] == 1 else "train")
            if p["split"] != expected:
                raise ValueError("Primary split view disagrees with CV fold1")


def check_assignments(
    cohort: pa.Table, patients: pa.Table, settings: dict, quotas: dict
) -> dict:
    if patients.schema != SCHEMA or any(patients[k].null_count for k in SCHEMA.names):
        raise ValueError("Invalid patient split schema or null fields")
    summaries = {
        r["subject_id"]: r
        for r in aggregate_patients(cohort, settings["cohort_version"])
    }
    records = patients.to_pylist()
    by_id = {r["subject_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(summaries):
        raise ValueError("Patient split coverage is missing, duplicated or extra")
    for row in records:
        if (
            row["split"] not in SPLITS
            or row["seed"] != settings["seed"]
            or row["split_version"] != settings["split_version"]
            or row["cohort_version"] != settings["cohort_version"]
            or any(row[k] != value for k, value in summaries[row["subject_id"]].items())
        ):
            raise ValueError("Invalid patient assignment or aggregation provenance")
    actual = {
        name: [
            sum(
                1 if name == "total" else r[name]
                for r in records
                if r["split"] == split
            )
            for split in SPLITS
        ]
        for name in quotas
    }
    if (
        actual != quotas
        or any(v <= 0 for v in actual["total"])
        or any(v <= 0 for v in actual["ever_high_burden"])
    ):
        raise ValueError("Patient quotas or high-burden coverage failed")
    return {
        "patient_coverage_complete": True,
        "patient_groups_disjoint": True,
        "each_study_has_one_split": True,
        "groups_nonempty": True,
        "quotas_match": True,
        "high_burden_in_each_split": True,
        "quotas": quotas,
    }

def split_reports(
    cohort: pa.Table, patients: pa.Table, checks: dict
) -> tuple[list[dict], dict]:
    by_id = {p["subject_id"]: p for p in patients.to_pylist()}
    rows = cohort.to_pylist()
    summary, evaluability = [], {}
    for split in SPLITS:
        group = [r for r in rows if by_id[r["subject_id"]]["split"] == split]
        people = [p for p in by_id.values() if p["split"] == split]
        categories = [("total", "all", group)]
        categories += [
            (
                "study_score",
                str(score),
                [r for r in group if r["burden_score"] == score],
            )
            for score in range(9)
        ]
        categories += [
            (
                "patient_max_score",
                str(score),
                [
                    r
                    for r in group
                    if by_id[r["subject_id"]]["max_burden_score"] == score
                ],
            )
            for score in range(9)
        ]
        categories += [
            ("endpoint", flag, [r for r in group if r[flag] == 1])
            for flag in (*COMPONENTS, "phys_death", "high_burden", "early_discharge")
        ]
        summary.extend(
            {
                "split": split,
                "category": cat,
                "value": value,
                **freeze_cohort._counts(selected),
            }
            for cat, value, selected in categories
        )
        evaluability[split] = {}
        for flag in (*COMPONENTS, "phys_death", "high_burden"):
            positive = [r for r in group if r[flag] == 1]
            positive_patients = len({r["subject_id"] for r in positive})
            evaluability[split][flag] = {
                "positive_studies": len(positive),
                "negative_studies": len(group) - len(positive),
                "positive_patients": positive_patients,
                "negative_patients": len(people) - positive_patients,
                "discrimination_evaluable": 0 < len(positive) < len(group),
                "patient_classes_available": 0 < positive_patients < len(people),
                "reason": "both_classes_present"
                if 0 < len(positive) < len(group)
                else "no_positive_studies"
                if not positive
                else "no_negative_studies",
            }
    return summary, {
        **checks,
        "cohort_counts": freeze_cohort._counts(rows),
        "endpoint_evaluability": evaluability,
    }

def check_source_leakage(connection, patients: pa.Table) -> None:
    connection.register("patient_split", patients)
    _reject(
        connection,
        "SELECT count(*) FROM (SELECT e.component,e.source_record_key FROM evidence e JOIN common c USING(sample_id) JOIN patient_split p ON p.subject_id=e.subject_id GROUP BY e.component,e.source_record_key HAVING count(DISTINCT p.split)>1)",
        "A stable component/source record crosses patient splits",
    )

def validate_settings(settings):
    expected={"algorithm_version":ALGORITHM,"weights":[70,15,15],"seed":42,
              "primary_stratum":"ever_phys_death","require_high_burden_in_each_split":True,"confirmed":True}
    if settings.get("algorithm_version") == CV_ALGORITHM:
        expected.update(algorithm_version=CV_ALGORITHM, weights=[68, 17, 15],
                        primary_stratum="phys_death_and_high_burden", test_fraction=0.15,
                        n_folds=5, primary_fold=1, index_base=0)
    require(all(type(settings.get(k)) is type(v) and settings[k]==v for k,v in expected.items())
            and all(type(v) is int for v in settings["weights"]),
            "INVALID_SPLIT_CONFIG","Configuration must explicitly confirm the supported patient allocation rules")
    require(isinstance(settings.get("cohort_version"),str) and re.fullmatch(r"v[1-9][0-9]*_r[1-9][0-9]*",settings["cohort_version"])
            and settings.get("split_version")==settings["cohort_version"],
            "INVALID_SPLIT_CONFIG","Split identity must match the current data run")
    require(all(isinstance(settings.get(k),str) and settings[k].strip() for k in ("reviewer","reviewed_on","rationale")),
            "INCOMPLETE_REVIEW","Record approval of the allocation policy before splitting")
    date.fromisoformat(settings["reviewed_on"])
    return settings


def load_settings(context,path="cohort/patient_split.json"):
    settings=context.read(path)
    require(not {"cohort_version","split_version"}&set(settings),"INVALID_SPLIT_CONFIG",
            "Use the current v/r directory; do not declare separate cohort or split versions")
    return validate_settings({**settings,"cohort_version":context.key,"split_version":context.key})


def load_frozen_cohort(data_context,store):
    prior=store.validate(store.manifest_ref("step08","freeze"))
    frozen=read_json(data_context.layout.resolve(store.output_ref(prior,"cohort_freeze.json")))
    require(frozen.get("data_run")==frozen.get("cohort_version")==data_context.key
            and frozen.get("horizon_hours")==72 and frozen.get("common_cohort_frozen") is True
            and frozen.get("cohort_checks_passed") is True
            and prior["details"].get("cohort_checks_passed") is True,
            "COHORT_NOT_FROZEN","A checked, frozen 72-hour common cohort is required")
    cohort=pq.read_table(data_context.layout.resolve(store.output_ref(prior,"cohort.parquet")))
    cohort_audit=pq.read_table(data_context.layout.resolve(store.output_ref(prior,"cohort_audit.parquet")))
    require(freeze_cohort._counts(cohort.to_pylist())==frozen["summary"]["counts"]==prior["details"]["counts"],
            "COHORT_COUNT_MISMATCH","Frozen cohort counts differ")
    context=freeze_cohort.verify_inputs(data_context,store)
    context.update(cohort=cohort,cohort_audit=cohort_audit)
    return context


def create_patient_split(data_context,split_path="cohort/patient_split.json", *, split_method="holdout"):
    started=datetime.now(timezone.utc)
    settings=load_settings(data_context,split_path)
    require(SPLIT_METHODS.get(split_method) == settings["algorithm_version"],
            "SPLIT_METHOD_MISMATCH", "Requested split method must match the selected configuration")
    cross_validation = split_method == "five-fold"
    fold_paths = [f"fold_{fold}/{TABLE_NAME}" for fold in range(1, 6)] if cross_validation else []
    data_files = (*fold_paths, CV_INDICES) if cross_validation else (TABLE_NAME,)
    store=ManifestStore(data_context.layout,data=data_context)
    prior_ref=store.manifest_ref("step08","freeze")
    details={"horizon_hours":72,"algorithm_version":settings["algorithm_version"],
             "split_method":split_method}
    with store.publish("step09","split",inputs=[prior_ref],configs=[data_context.config_ref(split_path)],
                       data_files=data_files,details=details,required_bytes=128*1024*1024) as work:
        context=load_frozen_cohort(data_context,store)
        run=AuditWorkspace(work,data_context.layout)
        with tempfile.TemporaryDirectory(dir=work,prefix=".split_") as scratch:
            limits=context["clinical_snapshot"]["ehr"]["burden_v2"]
            with open_store(Path(scratch),limits["memory_limit"],max_temp_directory_size=limits["max_temp_directory_size"]) as connection:
                expected,expected_audit=freeze_cohort.validate_cohort(connection,context)
                require(context["cohort"].equals(expected) and context["cohort_audit"].equals(expected_audit),
                        "COHORT_MISMATCH","Common cohort does not reproduce the frozen labels and reviewed QC")
                cohort = context["cohort"]
                if cross_validation:
                    patients, quotas = allocate_grouped_cv(cohort, settings)
                    indices = quotas.pop("cross_validation")
                    for fold, name in enumerate(fold_paths, 1):
                        view = select_fold(patients, indices, fold)
                        check_source_leakage(connection, view)
                        path = work / name
                        path.parent.mkdir()
                        pq.write_table(view, path, compression="zstd")
                    write_json(work / CV_INDICES, indices)
                else:
                    patients, quotas = allocate_patients(cohort, settings)
                    check_source_leakage(connection, patients)
                    pq.write_table(patients, work / TABLE_NAME, compression="zstd")
                checks = check_assignments(cohort, patients, settings, quotas)
                checks["source_records_do_not_cross_splits"] = True
                if cross_validation:
                    checks["cross_validation_distribution"] = indices["distribution"]
                summary, checks = split_reports(cohort, patients, checks)
        write_csv(work/"split_summary.csv",["split","category","value","studies","admissions","patients"],summary)
        write_json(work/"split_checks.json",checks)
        run.checks.write_report(work/"csv_tail_checks.csv")
        prior=store.validate(prior_ref)
        record={"data_run":data_context.key,"cohort_version":data_context.key,"split_version":data_context.key,
                "horizon_hours":72,"patient_split_frozen":True,"split_checks_passed":True,
                "algorithm_version":settings["algorithm_version"],"settings":settings,"checks":checks,
                "split_method":split_method,
                "source_cohort":{"manifest":prior_ref,"generation":prior["generation"]},
                "csv_tail_checks":run.csv_tail_summary(),"source_csv_tail_checks":context["source05"]["csv_tail_checks"],
                "review_csv_tail_checks":context["source07"]["csv_tail_checks"],
                "runtime_versions":{"numpy":np.__version__,"pyarrow":pa.__version__},
                "started_utc":started.isoformat(),"finished_utc":datetime.now(timezone.utc).isoformat()}
        write_json(work/"split_freeze.json",record)
        write_json(work/"run_metadata.json",record)
        text=("# Patient-level split\n\n"
              f"Data run: {data_context.key}. Seed: {settings['seed']}. Weights: {settings['weights']}.\n\n"
              "Patients are grouped across all studies and admissions. Fixed strata and PCG64 allocation "
              "preserve patient and stable-source isolation; each split contains high-burden patients. "
              "See split_summary.csv and split_checks.json for counts and endpoint evaluability. "
              "Missing endpoint classes are reported, not repaired by selecting another seed. "
              "Fit imputation, scaling, feature selection and thresholds on training data only.\n")
        if cross_validation:
            text = ("# Patient-grouped five-fold cross-validation\n\n"
                    "The fixed test set targets 15% of studies. Each development fold is held out "
                    "once, with disjoint patients and both binary outcome classes in every partition. "
                    "The split summary describes fold 1; the index bundle describes all five folds. "
                    "Fit learned preprocessing using each fold's training partition only.\n")
        (work/"report.md").write_text(text,encoding="utf-8",newline="\n")
        details.update(patient_split_frozen=True,checks_passed=True,quotas=quotas,
                       counts={row["split"]:{k:row[k] for k in ("studies","admissions","patients")}
                               for row in summary if row["category"]=="total"})
    return 0
