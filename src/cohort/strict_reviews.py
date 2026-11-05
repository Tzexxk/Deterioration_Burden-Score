"""Current-run review decisions and retained audit evidence, without file hashes."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_io import (
    COMPONENTS,
    SUPPLEMENTARY_COLUMNS,
    TABLE_COLUMNS,
    clinical_time,
    identifier,
    record_key,
)
from src.cohort.strict_sources import REVIEW_SOURCES
from src.common.manifest import write_json
from src.common.paths import read_json, require

ARCHIVES = {
    "candidates": "candidate_studies.parquet",
    "candidate_evidence": "candidate_evidence.parquet",
    "event_evidence": "window_72h/event_evidence.parquet",
    "review_records": "review_source_records.parquet",
    "medication_identity": "medication_identity_audit.parquet",
    "ventilation_episodes": "ventilation_episodes.parquet",
}
SOURCE_GENERATION = "review_source_generation"
COMMON = "version run source_audit_generation review_policy sample_id source_record_keys decision reviewer reviewed_on rationale".split()
COHORT_COLUMNS = COMMON + [
    "reason",
    "care_level_at_t0",
    "historical_source",
    "historical_keys",
]
EVENT_COLUMNS = (
    COMMON
    + "horizon_hours component event_flag first_event_time override_scope baseline_status historical_source historical_keys".split()
)
REVIEW_FIELDS = (
    "overrides_path",
    "candidate_exclusions_path",
    "cohort_decisions_path",
    "event_decisions_path",
)


def load_decisions(path, context, event, check=None):
    if path is None:
        return []
    from src.common.csv_io import _csv_reader

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = _csv_reader(stream, check)
        required = set(COMMON) | (
            set(
                "horizon_hours component event_flag first_event_time override_scope".split()
            )
            if event
            else {"reason"}
        )
        require(
            reader.fieldnames
            and required <= set(reader.fieldnames)
            and len(reader.fieldnames) == len(set(reader.fieldnames)),
            "INCOMPLETE_REVIEW",
            "Review columns are incomplete",
        )
        rows = list(reader)
    targets = defaultdict(list)
    for row in rows:
        require(
            identifier(row["version"]) == context.version
            and identifier(row["run"]) == context.run,
            "REVIEW_VERSION_MISMATCH",
            "Review must belong to the current data run",
        )
        require(
            row["review_policy"] == "strict"
            and identifier(row["source_audit_generation"]) is not None
            and identifier(row["source_audit_generation"]) > 0,
            "INVALID_REVIEW_BINDING",
            "Review requires a strict source audit generation",
        )
        allowed = (
            {"pending", "set_event", "exclude_source"}
            if event
            else {"pending", "resolve_blocker", "exclude"}
        )
        require(
            row["decision"] in allowed,
            "INVALID_REVIEW_DECISION",
            "Unsupported review decision",
        )
        try:
            keys = json.loads(row["source_record_keys"] or "[]")
        except ValueError:
            require(False, "INVALID_REVIEW_KEYS", "Review keys must be a JSON array")
        require(
            isinstance(keys, list)
            and all(isinstance(key, str) and key for key in keys)
            and len(keys) == len(set(keys)),
            "INVALID_REVIEW_KEYS",
            "Review keys must be unique nonempty strings",
        )
        try:
            historical = json.loads(row.get("historical_keys") or "[]")
        except ValueError:
            require(
                False, "INVALID_REVIEW_KEYS", "Historical keys must be a JSON array"
            )
        require(
            isinstance(historical, list)
            and all(isinstance(key, str) for key in historical)
            and set(historical) <= set(keys)
            and (not historical or row.get("historical_source")),
            "INVALID_REVIEW_KEYS",
            "Historical-only keys must be cited and explicitly sourced",
        )
        if row["decision"] != "pending":
            require(
                keys and all(row[k].strip() for k in ("reviewer", "rationale")),
                "INCOMPLETE_REVIEW",
                "Final review requires evidence, reviewer and rationale",
            )
            if row["reviewed_on"]:
                try:
                    datetime.strptime(row["reviewed_on"], "%Y-%m-%d")
                except ValueError:
                    require(
                        False, "INCOMPLETE_REVIEW", "reviewed_on must be YYYY-MM-DD"
                    )
            else:
                require(
                    bool(row.get("historical_source", "").strip()),
                    "INCOMPLETE_REVIEW",
                    "Undated historical opinions require explicit historical provenance",
                )
        if event:
            require(
                identifier(row["horizon_hours"]) == 72
                and row["component"] in COMPONENTS
                and row["override_scope"] in {"component", "source_record"},
                "INVALID_REVIEW_SCOPE",
                "Only supported 72h event scopes are allowed",
            )
            flag, time = row["event_flag"], row["first_event_time"]
            if row["decision"] == "pending":
                require(
                    not flag and not time,
                    "INVALID_REVIEW_DECISION",
                    "Pending events must leave labels blank",
                )
            elif row["decision"] == "set_event":
                require(
                    flag in {"0", "1"}
                    and row["override_scope"] == "component"
                    and (
                        (flag == "0" and not time)
                        or (flag == "1" and clinical_time(time) is not None)
                    ),
                    "INVALID_REVIEW_DECISION",
                    "set_event requires a component flag and consistent time",
                )
            else:
                require(
                    flag in {"", "0"}
                    and not time
                    and row["override_scope"] == "source_record",
                    "INVALID_REVIEW_DECISION",
                    "exclude_source requires exact source scope and no event time",
                )
                row["event_flag"] = "0"
            target = (row["sample_id"], row["component"])
        else:
            require(
                bool(row["reason"].strip()),
                "INCOMPLETE_REVIEW",
                "Cohort review requires a specific blocker",
            )
            target = (row["sample_id"], row["reason"])
        if row["decision"] == "pending":
            continue
        for peer in targets[target]:
            if not event or "component" in {
                peer["override_scope"],
                row["override_scope"],
            }:
                require(
                    False,
                    "CONFLICTING_REVIEW",
                    "Only one current decision is allowed per target",
                )
            require(
                not set(json.loads(peer["source_record_keys"] or "[]")) & set(keys),
                "CONFLICTING_REVIEW",
                "Source exclusions cannot overlap",
            )
        targets[target].append(row)
    return rows


def clinical_rules(cohort):
    value = {
        **cohort,
        "burden_v2": {
            key: item
            for key, item in cohort["burden_v2"].items()
            if key not in REVIEW_FIELDS
        },
    }
    return value


def prepare_archive(context, store, work, required, generation):
    directory = work / "review_sources"
    prior_path = context.layout.resolve(store.manifest_ref("step05", "audit"))
    prior = read_json(prior_path) if prior_path.is_file() else None
    previous = generation - 1
    require(
        prior is None
        or (
            prior["scope"] == store.scope
            and prior["generation"] == previous
            and prior["status"] == "complete"
        ),
        "UPSTREAM_REBUILT",
        "Previous audit changed before review capture",
    )
    index = {"data_run": context.key, "snapshots": {}}
    outputs = {item["name"]: item for item in prior["outputs"]} if prior else {}

    def prior_output(name):
        source = context.layout.resolve(store.output_ref(prior, name))
        require(
            source.is_file() and source.stat().st_size == outputs[name]["bytes"],
            "REVIEW_SOURCE_CHANGED",
            "Previous audit evidence is missing or changed",
        )
        return source

    if "review_sources/index.json" in outputs:
        index = read_json(prior_output("review_sources/index.json"))
        require(
            index["data_run"] == context.key,
            "REVIEW_VERSION_MISMATCH",
            "Review archive belongs to another data run",
        )
        directory.mkdir(parents=True, exist_ok=True)
        files = ["index.json", *(key + ".parquet" for key in ARCHIVES)]
        if "review_sources/historical_evidence.parquet" in outputs:
            files.append("historical_evidence.parquet")
        for file in files:
            shutil.copyfile(prior_output("review_sources/" + file), directory / file)
    missing = required - {int(key) for key in index["snapshots"]}
    require(
        not missing or (prior is not None and missing == {previous}),
        "REVIEW_SOURCE_UNAVAILABLE",
        "Review must reference the previous audit or a retained source generation",
    )
    if missing:
        require(
            prior["details"]["review_policy"] == "strict",
            "REVIEW_POLICY_MISMATCH",
            "Review source must use strict policy",
        )
        metadata = read_json(prior_output("run_metadata.json"))
        configuration = read_json(prior_output("effective_configuration.json"))
        require(
            all(
                item.get("data_run") == context.key
                and item.get("audit_generation") == previous
                and item.get("review_policy") == "strict"
                for item in (metadata, configuration)
            ),
            "INVALID_REVIEW_ARCHIVE",
            "Previous audit snapshot identity is invalid",
        )
        directory.mkdir(parents=True, exist_ok=True)
        for key, name in ARCHIVES.items():
            source = prior_output(name)
            destination = directory / (key + ".parquet")
            current_schema = pq.read_schema(source)
            require(
                SOURCE_GENERATION not in current_schema.names,
                "INVALID_REVIEW_ARCHIVE",
                "Reserved source generation column",
            )
            schema = current_schema.append(pa.field(SOURCE_GENERATION, pa.int64()))
            temp = directory / (key + ".staged")
            with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
                if destination.exists():
                    require(
                        pq.read_schema(destination) == schema,
                        "REVIEW_SCHEMA_CHANGED",
                        "Source schema changed; review in a new data run",
                    )
                    for batch in pq.ParquetFile(destination).iter_batches(
                        batch_size=1024
                    ):
                        writer.write_batch(batch)
                for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
                    table = pa.Table.from_batches([batch]).append_column(
                        SOURCE_GENERATION,
                        pa.array([previous] * batch.num_rows, type=pa.int64()),
                    )
                    writer.write_table(table)
            temp.replace(destination)
        index["snapshots"][str(previous)] = {
            "audit_generation": previous,
            "review_policy": "strict",
            "inputs": prior["inputs"],
            "configuration": configuration,
            "input_versions": metadata["input_versions"],
            "review_evidence_sources": metadata.get("review_evidence_sources", {}),
        }
        write_json(directory / "index.json", index)
    return ReviewArchive(context, directory, index)


def raw_source_index(values):
    result = {}

    def collect(value, owner=None):
        if isinstance(value, dict):
            owner = value.get("hadm_id") or owner
            key = value.get("source_record_key")
            if key:
                table = key.split(":", 1)[0]
                fields = TABLE_COLUMNS.get(table, "").split()
                if table in REVIEW_SOURCES:
                    require(
                        isinstance(value.get("record"), dict),
                        "INVALID_REVIEW_EVIDENCE",
                        "Review record has lost its source payload",
                    )
                    payload = {
                        name: value.get(name)
                        for name in ("subject_id", "hadm_id", "record")
                    }
                    require(
                        record_key(table, payload) == key,
                        "REVIEW_EVIDENCE_CHANGED",
                        "Review content key no longer matches",
                    )
                    result[key] = {
                        "content": key,
                        "subject_id": identifier(value.get("subject_id")),
                        "hadm_id": identifier(value.get("hadm_id")),
                        "raw": value,
                    }
                if table == "ventilation_episode":
                    fields = "subject_id hadm_id source_record_keys start end".split()
                if fields:
                    base = list(fields)
                    fields += [
                        name
                        for name in SUPPLEMENTARY_COLUMNS.get(table, "").split()
                        if name in value
                    ]
                    raw = {
                        name: str(value.get(name))
                        if name in {"start", "end"}
                        else value.get(name)
                        for name in fields
                    }
                    result[key] = {
                        "content": record_key(table, raw),
                        "subject_id": identifier(value.get("subject_id")),
                        "hadm_id": identifier(owner),
                        "raw": value,
                    }
                    alias = value.get("legacy_source_record_key")
                    if alias:
                        old = {name: value.get(name) for name in base}
                        require(
                            record_key(table, old) == alias,
                            "REVIEW_EVIDENCE_CHANGED",
                            "Legacy record key has changed",
                        )
                        result[alias] = {**result[key], "content": alias}
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    collect(nested, owner)
        elif isinstance(value, list):
            for nested in value:
                collect(nested, owner)

    collect(values)
    return result


def current_records(context):
    result = [
        *context["transfers"],
        *context["stays"],
        *context["medication_events"],
        *context["ventilation_episodes"],
        *context["ventilation_rejected"],
        *context.get("review_records", []),
        {
            **context["admission"],
            "source_record_key": record_key("admissions", context["admission"]),
        },
    ]
    if context.get("patient"):
        result.append(
            {
                **context["patient"],
                "source_record_key": record_key("patients", context["patient"]),
            }
        )
    for table in ("chartevents", "procedureevents"):
        result += [
            {**row, "source_record_key": record_key(table, row)}
            for row in context[table]
        ]
    return result


class ReviewArchive:
    def __init__(self, context, directory, index):
        self.context, self.directory, self.index = context, directory, index
        self.cache_key = None
        self.cache = None
        self.checked = 0
        self.historical_keys = set()
        self.historical_rows = {}

    def source(self, generation, candidate):
        cache_key = (generation, candidate["subject_id"], candidate["hadm_id"])
        if cache_key != self.cache_key:
            saved = []
            candidates = {}
            filters = [
                (SOURCE_GENERATION, "=", generation),
                ("subject_id", "=", candidate["subject_id"]),
                ("hadm_id", "=", candidate["hadm_id"]),
            ]
            for name in ARCHIVES:
                rows = pq.read_table(
                    self.directory / (name + ".parquet"), filters=filters
                ).to_pylist()
                if name == "candidates":
                    candidates = {row["sample_id"]: row for row in rows}
                else:
                    for row in rows:
                        raw = json.loads(row["raw_evidence"])
                        if isinstance(raw, dict) and "source_record_key" not in raw:
                            raw["source_record_key"] = row["source_record_key"]
                        saved.append(raw)
            self.cache_key, self.cache = (
                cache_key,
                (candidates, raw_source_index(saved)),
            )
        return self.cache

    def bind(
        self, row, candidate, current_index, rules, versions, review_bindings, inputs
    ):
        generation = identifier(row["source_audit_generation"])
        snapshot = self.index["snapshots"].get(str(generation))
        require(
            snapshot is not None,
            "REVIEW_SOURCE_UNAVAILABLE",
            "Referenced source audit is not retained",
        )
        require(
            snapshot["inputs"] == inputs,
            "UPSTREAM_REBUILT",
            "Upstream generations changed since review source",
        )
        require(
            snapshot["input_versions"] == versions
            and snapshot["review_evidence_sources"] == review_bindings,
            "REVIEW_EVIDENCE_CHANGED",
            "Raw source versions or projections changed; audit and review again",
        )
        expected = clinical_rules(snapshot["configuration"]["cohort"])["burden_v2"]
        current = {
            key: value
            for key, value in rules.items()
            if key not in (*REVIEW_FIELDS, "review_policy")
        }
        require(
            expected == current,
            "REVIEW_RULES_CHANGED",
            "Clinical rules or dictionaries changed since the reviewed audit",
        )
        candidates, old_index = self.source(generation, candidate)
        original = candidates.get(candidate["sample_id"])
        require(
            original is not None
            and all(
                original[key] == candidate[key]
                for key in ("subject_id", "hadm_id", "study_id", "t0", "dischtime")
            ),
            "REVIEW_TARGET_CHANGED",
            "Review sample identity or observation boundaries changed",
        )
        historical = set(json.loads(row.get("historical_keys") or "[]"))
        for key in json.loads(row["source_record_keys"]):
            current = current_index.get(key)
            require(
                current is not None
                and current["subject_id"] == candidate["subject_id"]
                and current["hadm_id"] == candidate["hadm_id"],
                "REVIEW_EVIDENCE_CHANGED",
                "Cited evidence is unavailable or belongs to another admission",
            )
            old = old_index.get(key)
            if old is None and key in historical:
                # A historical opinion may cite a raw record not emitted in the
                # initial audit for an excluded candidate. The original key
                # still proves the exact current payload and its actual owner.
                require(
                    current["content"] == key,
                    "REVIEW_EVIDENCE_CHANGED",
                    "Historical raw record does not reproduce its cited key",
                )
                self.historical_keys.add(key)
                self.historical_rows[(generation, key)] = {
                    SOURCE_GENERATION: generation,
                    "source_record_key": key,
                    "subject_id": current["subject_id"],
                    "hadm_id": current["hadm_id"],
                    "raw_evidence": json.dumps(
                        current["raw"], sort_keys=True, default=str
                    ),
                }
            else:
                require(
                    old is not None
                    and all(
                        old[k] == current[k]
                        for k in ("content", "subject_id", "hadm_id")
                    ),
                    "REVIEW_EVIDENCE_CHANGED",
                    "Cited evidence differs from the retained audit",
                )
        if row["decision"] == "set_event" and row["event_flag"] == "1":
            time = clinical_time(row["first_event_time"])
            require(
                candidate["t0"]
                < time
                <= min(candidate["t0"] + timedelta(hours=72), candidate["dischtime"]),
                "REVIEW_TIME_OUTSIDE_WINDOW",
                "Reviewed event time is outside the observation window",
            )
        self.checked += 1

    def write_historical_evidence(self):
        if not self.historical_rows:
            return
        target = self.directory / "historical_evidence.parquet"
        values = {}
        if target.is_file():
            values = {
                (row[SOURCE_GENERATION], row["source_record_key"]): row
                for row in pq.read_table(target).to_pylist()
            }
        for key, row in self.historical_rows.items():
            values.setdefault(key, row)
        schema = pa.schema(
            [
                (SOURCE_GENERATION, pa.int64()),
                ("source_record_key", pa.string()),
                ("subject_id", pa.int64()),
                ("hadm_id", pa.int64()),
                ("raw_evidence", pa.string()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(
                [values[key] for key in sorted(values)], schema=schema
            ),
            target,
            compression="zstd",
        )


def cohort_decisions(candidate, decisions):
    resolved = {
        row["reason"] for row in decisions if row["decision"] == "resolve_blocker"
    }
    for row in decisions:
        if (
            row["decision"] == "resolve_blocker"
            and row["reason"] == "Conflicting care-unit sources at t0"
        ):
            level = row.get("care_level_at_t0")
            require(
                level in {"ward", "stepdown_intermediate", "icu_ccu"},
                "INVALID_CARE_LEVEL_REVIEW",
                "Supply the reviewed care level",
            )
            reasons = set(candidate["exclusion_reason"].split(";")) - {
                "",
                "transfers_icustays_conflict",
            }
            require(
                "transfers_icustays_conflict" in candidate["exclusion_reason"]
                or candidate["care_level_at_t0"] == level,
                "REVIEW_BLOCKER_CHANGED",
                "Care-level correction does not match the current blocker",
            )
            if "transfers_icustays_conflict" not in candidate["exclusion_reason"]:
                continue
            candidate["care_level_at_t0"] = level
            if level == "icu_ccu":
                reasons.add("icu_at_t0")
            candidate.update(
                exclusion_reason=";".join(sorted(reasons)),
                candidate_status="excluded"
                if "icu_at_t0" in reasons
                or (candidate["candidate_status"] == "excluded" and reasons)
                else "needs_review"
                if reasons
                else "eligible",
                review_status="reviewed",
            )
    exclusions = [row for row in decisions if row["decision"] == "exclude"]
    if not exclusions:
        return resolved, None
    first = exclusions[0]
    merged = {
        **first,
        "reason": ";".join(sorted({row["reason"] for row in exclusions})),
        "source_record_keys": json.dumps(
            sorted(
                {
                    key
                    for row in exclusions
                    for key in json.loads(row["source_record_keys"])
                }
            )
        ),
    }
    return resolved, merged
