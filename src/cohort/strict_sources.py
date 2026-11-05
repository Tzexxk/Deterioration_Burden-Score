"""Scoped raw review evidence for strict audits, with portable provenance."""

from __future__ import annotations

import csv
from typing import Any

import duckdb

from src.cohort.burden_io import _csv_probe_order, record_key
from src.cohort.burden_support import AuditWorkspace
from src.cohort.raw_io import _file_version
from src.common.csv_io import _validate_csv_rejects
from src.common.paths import PipelineError, require

REVIEW_SOURCES = {
    "inputevents_full": (
        "mimic_iv",
        "subject_id hadm_id itemid starttime endtime amount rate orderid linkorderid statusdescription",
    ),
    "ingredientevents_full": (
        "mimic_iv",
        "subject_id hadm_id itemid starttime endtime orderid linkorderid",
    ),
    "poe_full": ("mimic_iv", "subject_id hadm_id poe_id ordertime"),
    "poe_detail_full": ("mimic_iv", "subject_id poe_id field_name field_value"),
    "pharmacy_full": ("mimic_iv", "subject_id hadm_id pharmacy_id poe_id medication"),
    "prescriptions_full": (
        "mimic_iv",
        "subject_id hadm_id pharmacy_id poe_id drug drug_type route",
    ),
    "emar_full": (
        "mimic_iv",
        "subject_id hadm_id emar_id poe_id pharmacy_id charttime",
    ),
    "emar_detail_full": (
        "mimic_iv",
        "subject_id emar_id administration_type infusion_rate_adjustment",
    ),
    "discharge_notes_full": ("mimic_note", "subject_id hadm_id note_id charttime text"),
}

REVIEW_TARGET_ITEMS = {
    "221289": "Epinephrine",
    "221653": "Dobutamine",
    "221662": "Dopamine",
    "221749": "Phenylephrine",
    "221906": "Norepinephrine",
    "222315": "Vasopressin",
    "229617": "Epinephrine.",
    "229630": "Phenylephrine (50/250)",
    "229631": "Phenylephrine (200/250)_OLD_1",
    "229632": "Phenylephrine (200/250)",
}


def review_source_paths(context, settings):
    sources = settings.get("review_evidence_sources")
    require(
        isinstance(sources, dict) and not (set(sources) - set(REVIEW_SOURCES)),
        "INVALID_REVIEW_SOURCES",
        "Declare review_evidence_sources explicitly using supported source names",
    )
    for child, parent in (
        ("poe_detail_full", "poe_full"),
        ("emar_detail_full", "emar_full"),
    ):
        require(
            child not in sources or parent in sources,
            "INVALID_REVIEW_SOURCES",
            "Review detail requires its ownership parent",
        )
    paths = {
        name: context.layout.path(REVIEW_SOURCES[name][0], path)
        for name, path in sources.items()
    }
    require(
        all(path.is_file() for path in paths.values()),
        "MISSING_REVIEW_SOURCE",
        "A configured review evidence source is missing",
    )
    return paths


def ingest_review_sources(connection: Any, sources: dict, run: AuditWorkspace) -> dict:
    """Read full source rows with strict CSV validation and unique parent ownership."""
    bindings = {}
    # Parents precede detail tables even when config ordering differs.
    order = [n for n in sources if n not in {"poe_detail_full", "emar_detail_full"}]
    order += [n for n in sources if n not in order]
    for name in order:
        path = sources[name]
        print(f"Read review source {name}", flush=True)
        before = _file_version(path)
        if not before["exists"]:
            raise FileNotFoundError(path)
        check = run.check_csv(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if (
            len(header) != len(set(header))
            or "review_owner_hadm_id" in header
            or set(REVIEW_SOURCES[name][1].split()) - set(header)
        ):
            raise ValueError(f"Missing or duplicate review columns: {name}")
        join = ""
        if name in {"poe_detail_full", "emar_detail_full"}:
            parent, key = (
                ("poe_full", "poe_id")
                if name == "poe_detail_full"
                else ("emar_full", "emar_id")
            )
            # Aggregate the retained parent once, before scanning the full detail CSV.
            # A correlated aggregate here can materialize keys from the entire CSV.
            owners = f"review_owners_{name}"
            connection.execute(
                f"CREATE OR REPLACE TEMP TABLE {owners} AS SELECT subject_id,{key}, "
                "CASE WHEN count(DISTINCT hadm_id)=1 AND "
                "count(*)=count(nullif(hadm_id,'')) THEN min(hadm_id) "
                f"END AS review_owner_hadm_id FROM {parent} GROUP BY subject_id,{key}"
            )
            owner = "o.review_owner_hadm_id"
            join = (
                f" INNER JOIN {owners} o ON o.subject_id=r.subject_id "
                f"AND o.{key}=r.{key}"
            )
            condition = "true"
        elif name in {"poe_full", "emar_full"}:
            # Conflicting parents outside a selected admission still affect ownership.
            owner = "r.hadm_id"
            condition = "try_cast(r.subject_id AS BIGINT) IN (SELECT subject_id FROM candidate_keys)"
        else:
            owner = "r.hadm_id"
            condition = "(try_cast(r.subject_id AS BIGINT),try_cast(r.hadm_id AS BIGINT)) IN (SELECT subject_id,hadm_id FROM candidate_keys)"
        if name == "inputevents_full":
            condition += (
                " AND r.itemid IN ("
                + ",".join("'" + k + "'" for k in REVIEW_TARGET_ITEMS)
                + ")"
            )
        try:
            with _csv_probe_order(connection):
                connection.execute(
                    f"CREATE TABLE {name} AS SELECT r.*, {owner} AS review_owner_hadm_id "
                    "FROM read_csv(?,columns=?,auto_detect=false,delim=',',header=true,"
                    "quote='\"',escape='\"',null_padding=false,strict_mode=true,ignore_errors=true,"
                    f"store_rejects=true,rejects_limit=0,rejects_table='reject_{name}',rejects_scan='scan_{name}') r{join} WHERE {condition}",
                    [str(path), dict.fromkeys(header, "VARCHAR")],
                )
        except duckdb.OutOfMemoryException:
            raise PipelineError(
                "REVIEW_SOURCE_OUT_OF_MEMORY",
                f"Memory or spill exhaustion reading {name}; no partial audit will be published",
            ) from None
        rejects = connection.execute(
            f"SELECT DISTINCT line_byte_position,csv_line FROM reject_{name}"
        ).fetchall()
        _validate_csv_rejects(path, check, rejects)
        if name == "inputevents_full":
            for item, label, target in connection.execute(
                "SELECT DISTINCT r.itemid,d.label,d.linksto FROM inputevents_full r "
                "LEFT JOIN d_items d ON r.itemid=d.itemid"
            ).fetchall():
                if label != REVIEW_TARGET_ITEMS[item] or target != "inputevents":
                    raise ValueError("Review target item identity changed")
        if before != _file_version(path):
            raise ValueError("Review source changed during ingestion")
        bindings[name] = {
            "source": run.checks.reference(path),
            "version": before,
            "columns": header,
        }
        connection.execute(f"DROP TABLE reject_{name}")
        connection.execute(f"DROP TABLE scan_{name}")
        print(f"Review source {name}: complete", flush=True)
    return bindings


def review_records(
    connection: Any, sources: dict, subject: int, admission: int
) -> list[dict]:
    result = []
    for name in sources:
        query = connection.execute(
            f"SELECT * FROM {name} WHERE try_cast(subject_id AS BIGINT)=? AND "
            "(try_cast(review_owner_hadm_id AS BIGINT)=? OR review_owner_hadm_id IS NULL)",
            [subject, admission],
        )
        for values in query.fetchall():
            record = dict(zip([c[0] for c in query.description], values, strict=True))
            owner = record.pop("review_owner_hadm_id")
            payload = {
                "subject_id": record["subject_id"],
                "hadm_id": owner,
                "record": record,
            }
            result.append({**payload, "source_record_key": record_key(name, payload)})
    return result


def write_review_templates(connection, run, directory, rules, context, generation):
    """Write pending suggestions, never automatic approvals or clinical labels."""
    import json

    from src.cohort.burden_support import write_csv

    identity_fields = {
        "version": context.version,
        "run": context.run,
        "source_audit_generation": generation,
        "review_policy": "strict",
    }
    common = "version run source_audit_generation review_policy sample_id source_record_keys decision reviewer reviewed_on rationale".split()
    policy_reasons = {
        policy.get("reason")
        for policy in rules.get("candidate_conflict_policies", {}).values()
        if isinstance(policy, dict)
    }
    policy_reasons.add(
        rules.get("candidate_medication_identity_policy", {}).get("reason")
    )
    columns = [
        entry[0]
        for entry in connection.execute("SELECT * FROM candidates LIMIT 0").description
    ]
    cohort = []
    for values in connection.execute(
        "SELECT * FROM candidates WHERE candidate_status<>'eligible' ORDER BY sample_id"
    ).fetchall():
        row = dict(zip(columns, values, strict=True))
        reasons = set((row["exclusion_reason"] or "").split(";"))
        if row["candidate_status"] != "needs_review" and not (
            reasons & (policy_reasons - {None, ""})
        ):
            continue
        cohort.append(
            {
                **dict.fromkeys(common, ""),
                **identity_fields,
                "sample_id": row["sample_id"],
                "reason": row["exclusion_reason"],
                "source_record_keys": row["candidate_evidence_keys"],
                "decision": "pending",
            }
        )
    write_csv(
        run.path(directory / "review_templates/cohort.csv"),
        common + ["reason", "care_level_at_t0"],
        [{**row, "care_level_at_t0": ""} for row in cohort],
    )
    groups = {}
    for sample, component, key, keys in connection.execute(
        "SELECT e.sample_id,e.component,e.source_record_key,e.source_record_keys FROM evidence e JOIN wide w USING(sample_id,horizon_hours) "
        "WHERE w.candidate_status<>'excluded' AND (e.qualifies=1 OR (e.qualifies IS NULL AND e.blocks_event=1)) ORDER BY e.sample_id,e.component,e.source_record_key"
    ).fetchall():
        groups.setdefault((sample, component), set()).update([key, *json.loads(keys)])
    blocked = set(
        connection.execute(
            "SELECT DISTINCT e.sample_id,e.component FROM evidence e JOIN wide w USING(sample_id,horizon_hours) "
            "WHERE w.candidate_status<>'excluded' AND e.qualifies IS NULL AND e.blocks_event=1"
        ).fetchall()
    )
    events = [
        {
            **dict.fromkeys(common, ""),
            **identity_fields,
            "sample_id": sample,
            "horizon_hours": 72,
            "component": component,
            "source_record_keys": json.dumps(sorted(groups[sample, component])),
            "decision": "pending",
            "event_flag": "",
            "first_event_time": "",
            "override_scope": "component",
        }
        for sample, component in sorted(blocked)
    ]
    write_csv(
        run.path(directory / "review_templates/events.csv"),
        common
        + "horizon_hours component event_flag first_event_time override_scope".split(),
        events,
    )
    run.path(directory / "review_templates/README.md").write_text(
        f"# Strict audit review suggestions\n\nData run: {context.key}; audit generation: {generation}.\n\n"
        "cohort.csv lists candidate blockers requiring review; events.csv lists components with blocking evidence. "
        "Hard eligibility exclusions remain in excluded_studies.csv. The CSVs are pending suggestions, not accepted decisions. "
        "Inspect source keys and raw evidence before adopting historical reviews. First-event times and flags stay blank. "
        "Bind reviewed rows to this source audit generation before rerunning the audit. Freeze only after the audit is ready.\n",
        encoding="utf-8",
        newline="\n",
    )
