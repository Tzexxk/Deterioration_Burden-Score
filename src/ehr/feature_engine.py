"""Reviewed laboratory aggregation and known-at-imaging static features."""

import json
import math
import re

import pyarrow as pa

from src.cohort.burden_io import clinical_time
from src.ehr import lab_audit as lab
from src.ehr import lab_review as review

APPROVED_FEATURES = {
    "50861": ("alt", "IU/L"),
    "50868": ("anion_gap", "mEq/L"),
    "50878": ("ast", "IU/L"),
    "50882": ("bicarbonate", "mEq/L"),
    "50885": ("total_bilirubin", "mg/dL"),
    "50893": ("total_calcium", "mg/dL"),
    "50902": ("chloride", "mEq/L"),
    "50912": ("creatinine", "mg/dL"),
    "50931": ("glucose", "mg/dL"),
    "50960": ("magnesium", "mg/dL"),
    "50970": ("phosphate", "mg/dL"),
    "50971": ("potassium", "mEq/L"),
    "50983": ("sodium", "mEq/L"),
    "51006": ("bun", "mg/dL"),
    "51221": ("hematocrit", "%"),
    "51222": ("hemoglobin", "g/dL"),
    "51237": ("inr", "1"),
    "51248": ("mch", "pg"),
    "51249": ("mchc", "g/dL"),
    "51250": ("mcv", "fL"),
    "51265": ("platelet_count", "K/uL"),
    "51275": ("ptt", "sec"),
    "51277": ("rdw", "%"),
    "51279": ("rbc", "m/uL"),
    "51301": ("wbc", "K/uL"),
}


AGGREGATIONS = [
    "last_24h",
    "min_24h",
    "max_24h",
    "mean_24h",
    "count_24h",
    "hours_since_last",
]


STATIC = [
    "age_at_t0",
    "sex",
    "admission_type",
    "admission_location",
    "careunit_at_t0",
    "current_service",
    "hours_from_admission",
]


CATEGORICAL = [
    "sex",
    "admission_type",
    "admission_location",
    "careunit_at_t0",
    "current_service",
]


SOURCE_COLUMNS = {
    "patients": "subject_id gender anchor_age anchor_year",
    "admissions": "subject_id hadm_id admittime admission_type admission_location",
    "services": "subject_id hadm_id transfertime curr_service",
}


ISSUE_COLUMNS = ["sample_id", "source", "field", "reason", "evidence"]


MISSING = "__MISSING__"


def validate_dictionary(table: pa.Table, version: str) -> list[dict]:
    rows = table.to_pylist()
    if not rows or any(r["dictionary_version"] != version for r in rows):
        raise ValueError("Dictionary version mismatch")
    retained = [r for r in rows if r["keep"]]
    expected = {int(k): v for k, v in APPROVED_FEATURES.items()}
    if {r["itemid"] for r in retained} != set(expected):
        raise ValueError("Unapproved laboratory item scope")
    for row in retained:
        name, unit = expected[row["itemid"]]
        allowed_units = (
            {"%", "g/dL"}
            if row["itemid"] == 51249
            else {""}
            if row["itemid"] == 51237
            else {unit}
        )
        if (
            row["feature_name"] != name
            or row["canonical_unit"] != unit
            or row["source_unit"] not in allowed_units
            or row["missing_unit_rule"]
            != ("explicit_item_unit" if row["itemid"] == 51237 else "")
            or json.loads(row["specimen_rule"]) != {"kind": "item_identity"}
            or review.duplicate_rule(row["duplicate_rule"])
            != {
                "kind": "latest_available",
                "conflicting_id": "error",
                "missing_specimen": "error",
            }
            or review.numeric_rule(row["numeric_result_rule"])
            != {
                "kind": "valuenum_only",
                "comparator": "exclude",
                "text_disagreement": "error",
            }
            or row["valid_min"] is not None
            or row["valid_max"] is not None
            or row["conversion_multiplier"] != 1
            or row["conversion_offset"] != 0
        ):
            raise ValueError("Dictionary rules differ from approved Step 10B scope")
    branches = [(r["itemid"], r["source_unit"]) for r in retained]
    if len(branches) != len(set(branches)) or {
        unit for item, unit in branches if item == 51249
    } != {"%", "g/dL"}:
        raise ValueError("Incomplete or overlapping approved unit branches")
    return sorted(retained, key=lambda r: (r["itemid"], r["source_unit"]))


def clean_labs(connection, rules: list[dict]) -> None:
    """Select the latest available revision before rejecting nonnumeric results."""
    review.validate_retained_conflicts(connection, rules)
    pieces, params = [], []
    for row in rules:
        pieces.append(
            "SELECT a.*,?::VARCHAR AS feature_name,?::VARCHAR AS canonical_unit,"
            "numeric_value*?+? AS converted_value FROM associations a WHERE itemid=? AND valueuom=? "
            "AND availability_reason='available' AND ownership_reason='owned'"
        )
        params.extend(
            [
                row["feature_name"],
                row["canonical_unit"],
                row["conversion_multiplier"],
                row["conversion_offset"],
                str(row["itemid"]),
                row["source_unit"],
            ]
        )
    connection.execute("CREATE TABLE selected AS " + " UNION ALL ".join(pieces), params)
    connection.execute(
        "CREATE TABLE lab_trace AS SELECT *, CASE WHEN revision_rank<>1 THEN 'superseded_revision' "
        "WHEN regexp_matches(value,'[<>≤≥]') THEN 'comparator' "
        "WHEN numeric_value IS NULL THEN 'no_finite_valuenum' ELSE 'retained' END AS disposition "
        "FROM (SELECT *,row_number() OVER(PARTITION BY sample_id,patient_id,assigned_hadm_id,itemid,specimen_id,chart_at "
        "ORDER BY stored_at DESC,try_cast(labevent_id AS BIGINT) DESC,lab_record_key DESC) AS revision_rank FROM selected)"
    )
    connection.execute(
        "CREATE TABLE cleaned AS SELECT * FROM lab_trace WHERE disposition='retained'"
    )
    if connection.execute(
        "SELECT count(*) FROM cleaned WHERE NOT isfinite(converted_value)"
    ).fetchone()[0]:
        raise ValueError("Non-finite laboratory conversion")


def schema_for(rules: list[dict], version: str) -> dict:
    units = {r["feature_name"]: r["canonical_unit"] for r in rules}
    names = sorted(units)
    numeric = [
        f"{name}_{aggregation}" for name in names for aggregation in AGGREGATIONS
    ] + ["age_at_t0", "hours_from_admission"]
    features = {}
    for name in names:
        ids = sorted({r["itemid"] for r in rules if r["feature_name"] == name})
        for aggregation in AGGREGATIONS:
            column = f"{name}_{aggregation}"
            features[column] = {
                "dtype": "float32",
                "unit": "count"
                if aggregation == "count_24h"
                else "hour"
                if aggregation == "hours_since_last"
                else units[name],
                "source": "hosp/labevents.csv",
                "itemids": ids,
                "aggregation": aggregation,
                "description": f"Reviewed {name}: {aggregation}; available in the inclusive pre-imaging 24-hour window.",
            }
    for name in STATIC:
        features[name] = {
            "dtype": "string" if name in CATEGORICAL else "float32",
            "unit": "year"
            if name == "age_at_t0"
            else "hour"
            if name == "hours_from_admission"
            else None,
            "source": "patients"
            if name in {"age_at_t0", "sex"}
            else "services"
            if name == "current_service"
            else "frozen cohort evidence"
            if name == "careunit_at_t0"
            else "admissions",
            "aggregation": "latest transfertime <= t0"
            if name == "current_service"
            else "static or t0 context",
            "description": name.replace("_", " "),
        }
    return {
        "schema_version": "1.0",
        "feature_set_version": version,
        "key_column": "sample_id",
        "lookback_hours": 24,
        "numeric_features": numeric,
        "categorical_features": CATEGORICAL,
        "categorical_missing_value": MISSING,
        "features": features,
        "limitations": [
            "Deidentified anchor ages cannot recover true ages above 89.",
            "Service availability uses transfertime; no independent service storetime is supplied.",
        ],
        "preprocessing_applied": False,
    }


def float32_table(rows: list[dict], schema: dict) -> pa.Table:
    columns = {"sample_id": pa.array([r["sample_id"] for r in rows], type=pa.string())}
    for name in schema["numeric_features"]:
        values = [float("nan") if r.get(name) is None else float(r[name]) for r in rows]
        if any(math.isinf(v) for v in values):
            raise ValueError(f"Non-finite input feature: {name}")
        array = pa.array(values, type=pa.float64()).cast(pa.float32())
        if any(math.isinf(v) for v in array.to_pylist()):
            raise ValueError(f"float32 overflow: {name}")
        columns[name] = array
    for name in schema["categorical_features"]:
        columns[name] = pa.array(
            [r.get(name) or MISSING for r in rows], type=pa.string()
        )
    return pa.table(columns)


def aggregate_labs(connection, cohort: pa.Table, rules: list[dict]) -> list[dict]:
    names = sorted({r["feature_name"] for r in rules})
    rows = {
        r["sample_id"]: {
            "sample_id": r["sample_id"],
            **{
                f"{n}_{a}": 0 if a == "count_24h" else None
                for n in names
                for a in AGGREGATIONS
            },
        }
        for r in cohort.to_pylist()
    }
    for result in lab.query_rows(
        connection,
        "SELECT sample_id,feature_name,"
        "first(converted_value ORDER BY chart_at DESC,stored_at DESC,try_cast(labevent_id AS BIGINT) DESC,lab_record_key DESC) AS last_24h,"
        "min(converted_value) AS min_24h,max(converted_value) AS max_24h,avg(converted_value) AS mean_24h,count(*) AS count_24h,"
        "epoch(max(t0)-max(chart_at))/3600.0 AS hours_since_last FROM cleaned GROUP BY sample_id,feature_name",
    ):
        rows[result["sample_id"]].update(
            {f"{result['feature_name']}_{a}": result[a] for a in AGGREGATIONS}
        )
    return [rows[k] for k in sorted(rows)]


def static_features(
    connection, cohort: pa.Table, evidence: pa.Table
) -> tuple[dict, list[dict]]:
    """Only cohort-filtered source subsets and known-at-t0 evidence enter Python."""
    sources = {
        name: lab.query_rows(connection, f"SELECT DISTINCT * FROM static_{name}")
        for name in SOURCE_COLUMNS
    }
    by_patient, by_admission, by_services = {}, {}, {}

    def number(value):
        return int(value) if value and re.fullmatch(r"[1-9][0-9]*", value) else None

    for row in sources["patients"]:
        by_patient.setdefault(number(row["subject_id"]), []).append(row)
    for row in sources["admissions"]:
        by_admission.setdefault(number(row["hadm_id"]), []).append(row)
    for row in sources["services"]:
        by_services.setdefault(number(row["hadm_id"]), []).append(row)
    known = {r["sample_id"]: r for r in evidence.to_pylist()}
    output, issues = {}, []
    for name, records in sources.items():
        for raw in records:
            if number(raw["subject_id"]) is None or (
                name != "patients" and number(raw["hadm_id"]) is None
            ):
                issues.append(
                    dict(
                        sample_id="",
                        source=name,
                        field="identity",
                        reason="invalid_source_identity",
                        evidence=json.dumps(raw, sort_keys=True),
                    )
                )
    for study in cohort.to_pylist():
        key, patient, admission, t0 = (
            study[k] for k in ["sample_id", "subject_id", "hadm_id", "t0"]
        )
        row = dict.fromkeys(STATIC)
        output[key] = row

        def issue(source, field, reason, raw):
            issues.append(
                dict(
                    sample_id=key,
                    source=source,
                    field=field,
                    reason=reason,
                    evidence=json.dumps(raw, default=str, sort_keys=True),
                )
            )

        patients = by_patient.get(patient, [])
        admissions = by_admission.get(admission, [])
        baseline = known.get(key)
        if (
            baseline is None
            or baseline["subject_id"] != patient
            or baseline["hadm_id"] != admission
            or baseline["t0"] != t0
        ):
            issue(
                "cohort", "identity", "missing_or_conflicting_frozen_evidence", baseline
            )
            continue
        row["age_at_t0"], row["careunit_at_t0"] = (
            baseline.get("age_at_t0"),
            baseline.get("careunit_at_t0"),
        )
        if len(patients) > 1:
            issue("patients", "identity", "conflicting_rows", patients)
        elif patients:
            p = patients[0]
            row["sex"] = (p["gender"] or "").strip() or None
            if row["sex"] not in {None, "F", "M"}:
                issue("patients", "sex", "unsupported_nonempty_value", p)
            raw_age, raw_year = p["anchor_age"], p["anchor_year"]
            if raw_age and raw_year:
                try:
                    age = int(raw_age) + t0.year - int(raw_year)
                    if age < 0 or age != row["age_at_t0"]:
                        raise ValueError("age mismatch")
                except (ValueError, TypeError):
                    issue("patients", "age_at_t0", "invalid_or_conflicting_age", p)
            elif any(v and not re.fullmatch(r"[0-9]+", v) for v in [raw_age, raw_year]):
                issue("patients", "age_at_t0", "invalid_nonempty_age_field", p)
        if len(admissions) > 1:
            issue("admissions", "identity", "conflicting_rows", admissions)
        elif admissions:
            a = admissions[0]
            if number(a["subject_id"]) != patient:
                issue("admissions", "identity", "patient_mismatch", a)
            admitted = clinical_time(a["admittime"])
            if (
                admitted is None
                or admitted != baseline.get("admittime")
                or admitted > t0
            ):
                issue("admissions", "admittime", "invalid_or_conflicting_admittime", a)
            else:
                row["hours_from_admission"] = (t0 - admitted).total_seconds() / 3600
            for field in ["admission_type", "admission_location"]:
                row[field] = (a[field] or "").strip() or None
        services = []
        for s in by_services.get(admission, []):
            time = clinical_time(s["transfertime"])
            if number(s["subject_id"]) != patient or time is None:
                issue(
                    "services", "identity_or_time", "invalid_or_conflicting_service", s
                )
            elif time <= t0:
                services.append((time, (s["curr_service"] or "").strip() or None))
        if services:
            latest = max(time for time, _ in services)
            values = {value for time, value in services if time == latest}
            if len(values) != 1:
                issue(
                    "services",
                    "current_service",
                    "conflicting_latest_service",
                    sorted(values, key=str),
                )
            else:
                row["current_service"] = next(iter(values))
    return output, issues


def validate_wide(table: pa.Table, cohort: pa.Table, schema: dict) -> None:
    numeric, categorical = schema["numeric_features"], schema["categorical_features"]
    names = ["sample_id", *numeric, *categorical]
    keys = table["sample_id"].to_pylist()
    if (
        len(names) != len(set(names))
        or table.column_names != names
        or set(schema["features"]) != set(names[1:])
        or len(keys) != len(set(keys))
        or set(keys) != set(cohort["sample_id"].to_pylist())
    ):
        raise ValueError("Wide table keys or whitelist mismatch")
    for name in numeric:
        if (
            table[name].type != pa.float32()
            or table[name].null_count
            or any(math.isinf(v) for v in table[name].to_pylist())
        ):
            raise ValueError(f"Invalid numeric feature: {name}")
    for name in categorical:
        if (
            table[name].type != pa.string()
            or table[name].null_count
            or any(not v for v in table[name].to_pylist())
        ):
            raise ValueError(f"Invalid categorical feature: {name}")


def feature_reports(
    connection, table: pa.Table, cohort: pa.Table, patients: pa.Table, schema: dict,
    *, cv_indices: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    split_by_patient = {r["subject_id"]: r["split"] for r in patients.to_pylist()}
    subject = {r["sample_id"]: r["subject_id"] for r in cohort.to_pylist()}
    rows = table.to_pylist()
    groups = {
        group: [
            r
            for r in rows
            if group == "all" or split_by_patient[subject[r["sample_id"]]] == group
        ]
        for group in ["all", "train", "validation", "test"]
    }
    if cv_indices is not None:
        by_sample = {row["sample_id"]: row for row in rows}
        master = cv_indices["master"]
        for fold in cv_indices["folds"]:
            for part in ("train", "validation"):
                groups[f"fold_{fold['fold']}_{part}"] = [
                    by_sample[master[i]["sample_id"]] for i in fold[f"{part}_indices"]
                ]
    summary, missingness = [], []
    raw = {
        (r["sample_id"], r["feature_name"])
        for r in lab.query_rows(
            connection,
            "SELECT DISTINCT sample_id,feature_name FROM selected WHERE numeric_value IS NOT NULL",
        )
    }
    for group, subset in groups.items():
        patient_ids = {subject[r["sample_id"]] for r in subset}
        for name in schema["numeric_features"] + schema["categorical_features"]:
            values = [
                r[name]
                for r in subset
                if r[name] != MISSING
                and not (isinstance(r[name], float) and math.isnan(r[name]))
            ]
            numeric = name in schema["numeric_features"]
            summary.append(
                dict(
                    split=group,
                    feature=name,
                    studies=len(subset),
                    missing=len(subset) - len(values),
                    missing_rate=(len(subset) - len(values)) / len(subset)
                    if subset
                    else None,
                    minimum=min(values) if numeric and values else None,
                    maximum=max(values) if numeric and values else None,
                    distinct_values=len(set(values)),
                    all_missing=not values,
                    constant=len(set(values)) == 1,
                )
            )
        for name in sorted(
            {
                r["feature_name"]
                for r in lab.query_rows(
                    connection, "SELECT DISTINCT feature_name FROM selected"
                )
            }
            | {
                n.removesuffix("_last_24h")
                for n in schema["numeric_features"]
                if n.endswith("_last_24h")
            }
        ):
            before = {r["sample_id"] for r in subset if (r["sample_id"], name) in raw}
            after = {
                r["sample_id"] for r in subset if not math.isnan(r[f"{name}_last_24h"])
            }
            rate = 1 - len(after) / len(subset) if subset else None
            raw_rate = 1 - len(before) / len(subset) if subset else None

            def coverage_band(observed):
                # Integer comparisons preserve the approved inclusive thresholds.
                missing, total = len(subset) - observed, len(subset)
                return (
                    None
                    if not total
                    else 0
                    if 100 * missing < 30 * total
                    else 1
                    if 100 * missing <= 50 * total
                    else 2
                    if 100 * missing <= 80 * total
                    else 3
                )

            missingness.append(
                dict(
                    split=group,
                    feature=name,
                    studies=len(subset),
                    patients=len(patient_ids),
                    before_studies=len(before),
                    after_studies=len(after),
                    before_patients=len({subject[k] for k in before}),
                    after_patients=len({subject[k] for k in after}),
                    before_missing_rate=raw_rate,
                    after_missing_rate=rate,
                    crossed_band=coverage_band(len(before))
                    != coverage_band(len(after)),
                    all_missing=not after,
                )
            )
    return summary, missingness
