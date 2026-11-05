"""Independent sample, laboratory and static reconstruction; no exporter logic."""

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EVENTS = (
    "new_icu_ccu_transfer",
    "new_stepdown_transfer",
    "new_invasive_ventilation",
    "new_severe_hypoxemia",
    "new_iv_vaso_inotrope_infusion",
    "in_hospital_death",
)


SPLITS = ("train", "validation", "test")


AGGREGATIONS = (
    "last_24h",
    "min_24h",
    "max_24h",
    "mean_24h",
    "count_24h",
    "hours_since_last",
)


CATEGORIES = [
    "sex",
    "admission_type",
    "admission_location",
    "careunit_at_t0",
    "current_service",
]


SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        *[(k, pa.int64()) for k in ("subject_id", "hadm_id", "study_id")],
        *[(k, pa.string()) for k in ("dicom_id", "split", "image_relpath")],
        ("burden_score", pa.float32()),
        *[(k, pa.uint8()) for k in (*EVENTS, "phys_death", "high_burden")],
        ("t0", pa.timestamp("us")),
        ("observation_end", pa.timestamp("us")),
        ("observed_hours", pa.float32()),
        ("early_discharge", pa.uint8()),
    ]
)


CHECKS = (
    "provenance",
    "samples",
    "ehr_contract",
    "laboratory_reconstruction",
    "static_reconstruction",
    "images",
    "metadata_summary",
    "zip_relocation",
    "source_stability",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    if isinstance(value, datetime):
        return value if value.tzinfo is None else None
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d{1,6})?", value
    ):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def integer(value):
    try:
        return int(value) if re.fullmatch(r"[0-9]+", str(value)) else None
    except (ValueError, TypeError):
        return None


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def indexed(table, key):
    rows = table.to_pylist()
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows) and None not in result, f"Duplicate or null {key}")
    return result


def validate_samples(samples, cohort, split, settings):
    require(
        samples.schema == SAMPLE_SCHEMA
        and samples.num_rows > 0
        and all(not c.null_count for c in samples.columns),
        "Invalid sample types or nulls",
    )
    rows = indexed(samples, "sample_id")
    original = indexed(cohort, "sample_id")
    assignments = indexed(split, "subject_id")
    require(set(rows) == set(original), "Common cohort sample keys changed")
    require(
        len({r["study_id"] for r in rows.values()}) == len(rows)
        and len({r["dicom_id"] for r in rows.values()}) == len(rows),
        "Study or image key is not unique",
    )
    require(
        set(assignments) == {r["subject_id"] for r in rows.values()},
        "Patient coverage changed",
    )
    for key, row in rows.items():
        source = original[key]
        ids = [row[k] for k in ("subject_id", "hadm_id", "study_id")]
        require(
            all(v > 0 for v in ids) and key == f"p{ids[0]}_h{ids[1]}_s{ids[2]}",
            "Invalid sample identity",
        )
        require(
            row["split"] in SPLITS and row["split"] == assignments[ids[0]]["split"],
            "Patient split changed",
        )
        expected_path = f"images/p{str(ids[0])[:2]}/p{ids[0]}/s{ids[2]}.npy"
        require(
            row["image_relpath"] == expected_path == source["npy_relpath"],
            "Image path identity changed",
        )
        for name in SAMPLE_SCHEMA.names:
            if name not in {"split", "image_relpath"}:
                require(
                    row[name] == source[name],
                    f"Frozen sample value changed: {key}/{name}",
                )
        require(
            all(
                row[k] in (0, 1)
                for k in (*EVENTS, "phys_death", "high_burden", "early_discharge")
            ),
            "Nonbinary label",
        )
        transfer = max(row[EVENTS[0]], row[EVENTS[1]])
        respiratory = max(row[EVENTS[2]], row[EVENTS[3]])
        circulatory, death = row[EVENTS[4]], row[EVENTS[5]]
        score = transfer + 2 * respiratory + 2 * circulatory + 3 * death
        require(
            row["burden_score"] == score
            and row["phys_death"] == int(bool(respiratory or circulatory or death))
            and row["high_burden"] == int(score >= 3),
            "Target formula mismatch",
        )
        end = min(row["t0"] + timedelta(hours=72), source["dischtime"])
        require(
            source["admittime"] <= row["t0"] < end and row["observation_end"] == end,
            "Observation cutoff mismatch",
        )
        hours = np.float32((end - row["t0"]).total_seconds() / 3600).item()
        require(
            row["observed_hours"] == hours
            and row["early_discharge"]
            == int(
                source["dischtime"] < row["t0"] + timedelta(hours=72)
                and str(source.get("hospital_expire_flag")) == "0"
                and "death_time_conflict" not in source.get("exclusion_reason", "")
            ),
            "Observed duration or discharge mismatch",
        )
        for event in EVENTS:
            time = source[event + "_first_event_time"]
            require(
                not row[event] or time is not None and row["t0"] < time <= end,
                "Positive event outside outcome window",
            )
    counts = {
        "studies": len(rows),
        "admissions": len({r["hadm_id"] for r in rows.values()}),
        "patients": len(assignments),
    }
    require(counts == settings["expected_counts"], "Frozen cohort counts changed")
    for group in SPLITS:
        part = [r for r in rows.values() if r["split"] == group]
        require(
            len(part) == settings["expected_split_studies"][group]
            and len({r["subject_id"] for r in part})
            == settings["expected_split_patients"][group],
            "Frozen split counts changed",
        )
    return counts


def retained_rules(dictionary):
    rules = [r for r in dictionary.to_pylist() if r["keep"]]
    require(bool(rules), "No retained dictionary rules")
    seen = set()
    for r in rules:
        key = (str(r["itemid"]), r["source_unit"])
        require(key not in seen, "Overlapping dictionary branches")
        seen.add(key)
        require(
            json.loads(r["specimen_rule"]) == {"kind": "item_identity"},
            "Unsupported specimen selector",
        )
        require(
            json.loads(r["duplicate_rule"])
            == {
                "kind": "latest_available",
                "conflicting_id": "error",
                "missing_specimen": "error",
            },
            "Unsupported duplicate selector",
        )
        require(
            json.loads(r["numeric_result_rule"])
            == {
                "kind": "valuenum_only",
                "comparator": "exclude",
                "text_disagreement": "error",
            },
            "Unsupported numeric selector",
        )
        require(
            r["valid_min"] is None
            and r["valid_max"] is None
            and bool(r["valid_range_basis"]),
            "Unexpected validity treatment",
        )
        require(
            finite(r["conversion_multiplier"]) is not None
            and finite(r["conversion_offset"]) is not None,
            "Invalid conversion",
        )
        require(
            bool(r["canonical_unit"])
            and (
                bool(r["source_unit"]) or r["missing_unit_rule"] == "explicit_item_unit"
            ),
            "Unreviewed missing unit",
        )
    return rules


def validate_ehr_contract(table, samples, schema, rules):
    numeric = [
        f"{n}_{a}"
        for n in sorted({r["feature_name"] for r in rules})
        for a in AGGREGATIONS
    ] + ["age_at_t0", "hours_from_admission"]
    require(
        schema["numeric_features"] == numeric
        and schema["categorical_features"] == CATEGORIES
        and set(schema["features"]) == set(numeric + CATEGORIES)
        and schema["key_column"] == "sample_id"
        and schema["lookback_hours"] == 24
        and schema["categorical_missing_value"] == "__MISSING__"
        and schema["preprocessing_applied"] is False,
        "Feature whitelist or missing-value contract changed",
    )
    require(
        table.column_names == ["sample_id", *numeric, *CATEGORIES]
        and table["sample_id"].type == pa.string(),
        "EHR columns changed",
    )
    require(
        set(indexed(table, "sample_id")) == set(samples["sample_id"].to_pylist()),
        "EHR alignment changed",
    )
    for name in numeric + CATEGORIES:
        col = table[name]
        require(
            col.type == (pa.float32() if name in numeric else pa.string())
            and not col.null_count,
            f"Invalid EHR type: {name}",
        )
        require(
            all(
                not math.isinf(v) if name in numeric else bool(v) and v == v.strip()
                for v in col.to_pylist()
            ),
            f"Invalid EHR value: {name}",
        )
    return {"numeric_features": len(numeric), "categorical_features": len(CATEGORIES)}


def compare_features(actual, expected, columns):
    rows = indexed(actual, "sample_id")
    require(set(rows) == set(expected), "Reconstructed feature keys changed")
    for key, source in expected.items():
        for column in columns:
            want, got = source[column], rows[key][column]
            if isinstance(want, (float, int, np.number)):
                want = np.float32(want).item()
                equal = math.isnan(want) and math.isnan(got) or want == got
                if (
                    column.endswith("_mean_24h")
                    and math.isfinite(want)
                    and math.isfinite(got)
                ):
                    equal = abs(want - got) <= abs(float(np.spacing(np.float32(want))))
            else:
                equal = want == got
            require(equal, f"Reconstruction mismatch: {key}/{column}")


def reconstruct_labs(lab_path, admission_table, cohort, rules, actual):
    """Stream the uncleaned bound subset; never trust published cleaning traces."""
    by_patient = defaultdict(list)
    for row in cohort.to_pylist():
        by_patient[row["subject_id"]].append(row)
    admissions = defaultdict(dict)
    for row in admission_table.to_pylist():
        patient, admission = integer(row["subject_id"]), integer(row["hadm_id"])
        pair = (timestamp(row["admittime"]), timestamp(row["dischtime"]))
        if patient is not None and admission is not None:
            require(
                admission not in admissions[patient]
                or admissions[patient][admission] == pair,
                "Conflicting admission intervals",
            )
            admissions[patient][admission] = pair
    rule_index = {(str(r["itemid"]), r["source_unit"]): r for r in rules}
    revisions, identifiers, selected_ids = {}, defaultdict(set), set()
    counts = Counter()
    for batch in pq.ParquetFile(lab_path).iter_batches(batch_size=8192):
        for raw in batch.to_pylist():
            counts["subset_records_scanned"] += 1
            rule = rule_index.get((raw["itemid"], raw["valueuom"]))
            patient = integer(raw["subject_id"])
            chart, stored = timestamp(raw["charttime"]), timestamp(raw["storetime"])
            if chart is None or stored is None or stored < chart:
                counts["invalid_or_missing_time"] += 1
                continue
            provided = raw["hadm_id"]
            if provided.strip():
                assigned = integer(provided)
                if assigned not in admissions[patient]:
                    counts["ownership_rejected"] += 1
                    continue
            else:
                membership = {
                    a
                    for a, (start, end) in admissions[patient].items()
                    if start is not None and end is not None and start <= chart <= end
                }
                if len(membership) != 1:
                    counts["ambiguous_or_absent_admission"] += 1
                    continue
                assigned = next(iter(membership))
            for study in by_patient.get(patient, []):
                t0, key = study["t0"], study["sample_id"]
                if (
                    assigned != study["hadm_id"]
                    or not t0 - timedelta(hours=24) <= chart <= t0
                    or stored > t0
                ):
                    continue
                identity = (key, raw["labevent_id"])
                identifiers[identity].add(raw["lab_record_key"])
                if rule is None:
                    continue
                counts["eligible_associations"] += 1
                require(
                    re.fullmatch(r"[1-9][0-9]*", raw["specimen_id"]),
                    f"Missing specimen in retained branch: {key}/{raw['lab_record_key']}",
                )
                numeric, text = finite(raw["valuenum"]), finite(raw["value"])
                require(
                    numeric is None or text is None or numeric == text,
                    f"Numeric/text disagreement in eligible revision: {key}/{raw['lab_record_key']}",
                )
                require(
                    re.fullmatch(r"[1-9][0-9]*", raw["labevent_id"]),
                    "Invalid retained labevent ID",
                )
                selected_ids.add(identity)
                group = (
                    key,
                    patient,
                    assigned,
                    raw["itemid"],
                    raw["specimen_id"],
                    chart,
                )
                order = (
                    stored,
                    integer(raw["labevent_id"]) or -1,
                    raw["lab_record_key"],
                )
                if group not in revisions or order > revisions[group][0]:
                    revisions[group] = (order, raw, rule, t0)
    require(
        all(len(identifiers[key]) == 1 for key in selected_ids),
        "Conflicting eligible labevent IDs",
    )
    values = defaultdict(list)
    for group, (order, raw, rule, t0) in revisions.items():
        numeric = finite(raw["valuenum"])
        if re.search(r"[<>≤≥]", raw["value"]) or numeric is None:
            counts["selected_nonnumeric_or_comparator"] += 1
            continue
        converted = numeric * rule["conversion_multiplier"] + rule["conversion_offset"]
        require(math.isfinite(converted), "Nonfinite converted value")
        values[(group[0], rule["feature_name"])].append(
            (group[-1], order, converted, t0)
        )
        counts["retained_associations"] += 1
    names = sorted({r["feature_name"] for r in rules})
    expected = {}
    for study in cohort.to_pylist():
        key = study["sample_id"]
        expected[key] = {}
        for name in names:
            records = values[(key, name)]
            result = [math.nan, math.nan, math.nan, math.nan, 0, math.nan]
            if records:
                last = max(records, key=lambda r: (r[0], r[1]))
                nums = [r[2] for r in records]
                result = [
                    last[2],
                    min(nums),
                    max(nums),
                    math.fsum(nums) / len(nums),
                    len(nums),
                    (study["t0"] - last[0]).total_seconds() / 3600,
                ]
            expected[key].update(
                {f"{name}_{a}": v for a, v in zip(AGGREGATIONS, result, strict=True)}
            )
    compare_features(
        actual, expected, [f"{n}_{a}" for n in names for a in AGGREGATIONS]
    )
    return dict(counts)


def reconstruct_static(cohort, evidence, patients, admissions, services, actual):
    known = indexed(evidence, "sample_id")
    p_index, a_index, s_index = defaultdict(list), defaultdict(list), defaultdict(list)
    for table, index, field in [
        (patients, p_index, "subject_id"),
        (admissions, a_index, "hadm_id"),
        (services, s_index, "hadm_id"),
    ]:
        unique = {json.dumps(r, sort_keys=True): r for r in table.to_pylist()}
        for row in unique.values():
            require(
                integer(row["subject_id"]) is not None
                and integer(row[field]) is not None,
                "Invalid static identity",
            )
            index[integer(row[field])].append(row)
    expected = {}
    for study in cohort.to_pylist():
        key, patient, admission, t0 = (
            study[k] for k in ("sample_id", "subject_id", "hadm_id", "t0")
        )
        base = known[key]
        require(
            all(
                study[k] == base[k]
                for k in (
                    "subject_id",
                    "hadm_id",
                    "t0",
                    "age_at_t0",
                    "careunit_at_t0",
                    "admittime",
                )
            ),
            "Static frozen evidence mismatch",
        )
        require(
            len(p_index[patient]) <= 1 and len(a_index[admission]) <= 1,
            "Conflicting static rows",
        )
        result: dict[str, Any] = dict.fromkeys(CATEGORIES, "__MISSING__")
        result.update(
            age_at_t0=base["age_at_t0"] if base["age_at_t0"] is not None else math.nan,
            hours_from_admission=math.nan,
            careunit_at_t0=base["careunit_at_t0"] or "__MISSING__",
        )
        if p_index[patient]:
            p = p_index[patient][0]
            result["sex"] = p["gender"].strip() or "__MISSING__"
            require(result["sex"] in ("F", "M", "__MISSING__"), "Invalid sex")
            if p["anchor_age"] and p["anchor_year"]:
                require(
                    int(p["anchor_age"]) + t0.year - int(p["anchor_year"])
                    == base["age_at_t0"],
                    "Age reconstruction mismatch",
                )
            else:
                require(
                    all(
                        not p[k] or re.fullmatch(r"[0-9]+", p[k])
                        for k in ("anchor_age", "anchor_year")
                    ),
                    "Invalid nonempty anchor age/year",
                )
        if a_index[admission]:
            a = a_index[admission][0]
            start = timestamp(a["admittime"])
            require(
                integer(a["subject_id"]) == patient
                and start == base["admittime"]
                and start <= t0,
                "Admission static conflict",
            )
            result["hours_from_admission"] = (t0 - start).total_seconds() / 3600
            result.update(
                {
                    k: a[k].strip() or "__MISSING__"
                    for k in ("admission_type", "admission_location")
                }
            )
        eligible = []
        for service in s_index[admission]:
            time = timestamp(service["transfertime"])
            require(
                time is not None and integer(service["subject_id"]) == patient,
                "Invalid service identity or timestamp",
            )
            if time <= t0:
                eligible.append(
                    (time, service["curr_service"].strip() or "__MISSING__")
                )
        if eligible:
            latest = max(r[0] for r in eligible)
            options = {r[1] for r in eligible if r[0] == latest}
            require(len(options) == 1, f"Conflicting latest service: {key}")
            result["current_service"] = options.pop()
        expected[key] = result
    compare_features(
        actual, expected, ["age_at_t0", "hours_from_admission", *CATEGORIES]
    )
    return {
        "studies_reconstructed": len(expected),
        "service_availability_proxy": "transfertime <= t0",
    }


def validate_pixels(path):
    image = np.load(path, allow_pickle=False)
    require(
        image.dtype == np.dtype("float32")
        and image.shape == (224, 224)
        and np.isfinite(image).all()
        and (image >= 0).all()
        and (image <= 1).all(),
        "Invalid NPY pixels, shape or dtype",
    )


def expected_summary(samples):
    result = []
    for group in ("all", *SPLITS):
        rows = [r for r in samples.to_pylist() if group == "all" or r["split"] == group]
        patients = {r["subject_id"] for r in rows}

        def add(name, count, unit, denominator):
            result.append(
                dict(
                    split=group,
                    metric=name,
                    value=count,
                    unit=unit,
                    denominator=denominator,
                )
            )

        for name, ids, unit in [
            ("studies", {r["sample_id"] for r in rows}, "study"),
            ("admissions", {r["hadm_id"] for r in rows}, "admission"),
            ("patients", patients, "patient"),
        ]:
            add(name, len(ids), unit, len(ids))
        maximum = {
            p: max(r["burden_score"] for r in rows if r["subject_id"] == p)
            for p in patients
        }
        for score in range(9):
            add(
                f"score_{score}_studies",
                sum(r["burden_score"] == score for r in rows),
                "study",
                len(rows),
            )
            add(
                f"maximum_score_{score}_patients",
                sum(v == score for v in maximum.values()),
                "patient",
                len(patients),
            )
        for event in (*EVENTS, "phys_death", "high_burden", "early_discharge"):
            add(event + "_studies", sum(r[event] for r in rows), "study", len(rows))
            add(
                event + "_patients",
                len({r["subject_id"] for r in rows if r[event]}),
                "patient",
                len(patients),
            )
    return result
