"""Laboratory audit, reviewed decisions and generation-bound dictionary freeze."""

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import create_patient_split as split
from src.cohort import freeze_cohort
from src.cohort.burden_io import open_store
from src.cohort.burden_support import AuditWorkspace, write_csv
from src.cohort.raw_io import _file_version
from src.common.manifest import ManifestStore, write_json
from src.common.paths import read_json, require
from src.ehr import lab_audit as audit
from src.ehr import lab_review as review

CACHE_NAMES = {"labs": "cache/labs_candidate_subset.parquet", "admissions": "cache/admissions.parquet", "dictionary": "cache/d_labitems.parquet"}


def load_settings(context, path):
    settings = context.read(path, required=("algorithm_version", "tables", "reviewer", "reviews", "memory_limit", "max_temp_directory_size", "workspace_estimate_bytes"))
    require(settings["algorithm_version"] == "lab_audit_review_v1" and settings.get("lookback_hours") == 24,
            "INVALID_LAB_CONFIG", "Use the supported 24-hour laboratory audit policy")
    require(not {"audit_version", "cohort_version", "split_version", "dictionary_version"} & set(settings),
            "INVALID_LAB_CONFIG", "Data version identity is selected by the current v/r directory")
    require("vital_audit" not in settings, "UNSUPPORTED_VITAL_AUDIT", "Vital-sign scope integration is not implemented")
    require(isinstance(settings["reviewer"], str) and settings["reviewer"].strip(),
            "INVALID_LAB_CONFIG", "Set the responsible reviewer")
    require(set(settings["tables"]) == {"d_labitems", "labevents", "admissions"},
            "INVALID_LAB_CONFIG", "Configure all three laboratory source tables")
    for field, maximum in [("batch_item_limit", 10), ("example_limit", 100), ("comment_max_chars", 4096)]:
        require(type(settings.get(field)) is int and 1 <= settings[field] <= maximum, "INVALID_LAB_CONFIG", "Invalid report bound")
    for field in ("memory_limit", "max_temp_directory_size"):
        require(isinstance(settings[field], str) and re.fullmatch(r"[1-9][0-9]*(MB|GB)", settings[field]),
                "INVALID_LAB_CONFIG", "Use an explicit MB or GB resource bound")
    require(type(settings["workspace_estimate_bytes"]) is int and settings["workspace_estimate_bytes"] > 0,
            "INVALID_LAB_CONFIG", "Set a positive disk-space estimate")
    require(isinstance(settings["reviews"], list) and all(isinstance(path, str) for path in settings["reviews"]),
            "INVALID_LAB_CONFIG", "Configure current-run review paths")
    return settings


def verify_upstream(context, store, connection):
    inputs = split.load_frozen_cohort(context, store)
    expected, expected_audit = freeze_cohort.validate_cohort(connection, inputs)
    require(inputs["cohort"].equals(expected) and inputs["cohort_audit"].equals(expected_audit),
            "COHORT_MISMATCH", "Frozen cohort does not reproduce its source evidence")
    manifest = store.validate(store.manifest_ref("step09", "split"))
    record = read_json(context.layout.resolve(store.output_ref(manifest, "split_freeze.json")))
    settings = split.validate_settings(record["settings"])
    require(record.get("data_run") == context.key and record.get("patient_split_frozen") is True
            and record.get("split_checks_passed") is True, "SPLIT_NOT_FROZEN", "A checked patient split is required")
    patients = pq.read_table(context.layout.resolve(store.output_ref(manifest, "patient_split.parquet")))
    reproduced, quotas = split.allocate_patients(inputs["cohort"], settings)
    require(patients.equals(reproduced), "SPLIT_MISMATCH", "Patient assignments do not reproduce the frozen policy")
    checks = split.check_assignments(inputs["cohort"], patients, settings, quotas)
    split.check_source_leakage(connection, patients)
    checks["source_records_do_not_cross_splits"] = True
    _, checks = split.split_reports(inputs["cohort"], patients, checks)
    require(checks == record["checks"], "SPLIT_MISMATCH", "Frozen split checks differ")
    return inputs["cohort"]


def perform_audit(connection, run, cohort, sources, settings, identity):
    dictionary = audit.read_small_csv(sources["d_labitems"], ["itemid", "label", "fluid", "category"], run)
    dictionary = [{key: row[key] for key in ("itemid", "label", "fluid", "category")} for row in dictionary]
    ids = [row["itemid"] for row in dictionary]
    require(len(set(ids)) == len(ids) and all(re.fullmatch(r"[1-9][0-9]*", value) for value in ids),
            "INVALID_LAB_DICTIONARY", "Dictionary item identifiers must be unique positive integers")
    connection.register("audit_patients", pa.table({"subject_id": sorted(set(cohort["subject_id"].to_pylist()))}))
    condition = "try_cast(r.subject_id AS BIGINT) IN (SELECT subject_id FROM audit_patients)"
    audit.scan_csv(connection, sources["admissions"], "admissions", ["subject_id", "hadm_id", "admittime", "dischtime"], [], condition, run)
    audit.scan_csv(connection, sources["labevents"], "labs_raw", audit.LAB_REQUIRED, audit.LAB_OPTIONAL, condition, run)
    audit.prepare_labs(connection, settings)
    audit.build_views(connection, cohort)
    inventory, batches, outside = audit.inventory(connection, dictionary, settings)
    require(all(row["itemid"].isdigit() for row in inventory), "INVALID_LAB_RECORD", "Invalid observed item identifier")
    for name in ("labs", "admissions"):
        audit.copy_parquet(connection, name, run.path(Path("data") / CACHE_NAMES[name]))
    schema = pa.schema([(key, pa.string()) for key in ("itemid", "label", "fluid", "category")])
    pq.write_table(pa.Table.from_pylist(dictionary, schema=schema), run.path(Path("data") / CACHE_NAMES["dictionary"]))
    counts, examples, specimens = audit.write_reports(connection, run, Path("reports"), inventory, batches, outside, settings)
    review.write_templates(run, Path("reports"), inventory, identity, "scope", batches)
    audit.write_html(run.path(Path("reports/lab_review.html")), inventory, batches, examples, specimens, identity)
    return {"inventory": inventory, "batches": batches, "counts": counts}


def audit_mode(context, config_path):
    settings = load_settings(context, config_path)
    store = ManifestStore(context.layout, data=context)
    sources = {name: context.layout.path("mimic_iv", path) for name, path in settings["tables"].items()}
    before = {name: _file_version(path) for name, path in sources.items()}
    require(all(value["exists"] for value in before.values()), "MISSING_SOURCE", "A laboratory source is missing")
    details = {"horizon_hours": 72, "lookback_hours": 24}
    with store.publish("step10", "audit", inputs=[store.manifest_ref("step09", "split")],
                       configs=[context.config_ref(config_path)], data_files=tuple(CACHE_NAMES.values()),
                       details=details, required_bytes=settings["workspace_estimate_bytes"]) as work:
        run = AuditWorkspace(work, context.layout)
        with tempfile.TemporaryDirectory(dir=work, prefix=".labs_") as scratch:
            with open_store(Path(scratch), settings["memory_limit"], max_temp_directory_size=settings["max_temp_directory_size"]) as connection:
                cohort = verify_upstream(context, store, connection)
                record = perform_audit(connection, run, cohort, sources, settings, context.key)
        require(before == {name: _file_version(path) for name, path in sources.items()}, "SOURCE_CHANGED", "Laboratory sources changed during audit")
        run.checks.write_report(work / "csv_tail_checks.csv")
        record.update(data_run=context.key, mode="audit", audit_checks_passed=True, settings=settings,
                      sources={name: context.layout.ref("mimic_iv", settings["tables"][name]) for name in sources},
                      source_versions=before, source_csv_tail_checks=run.csv_tail_summary(),
                      review_complete=False, dictionary_frozen=False, ehr_features_ready=False)
        write_json(work / "lab_audit.json", record)
        write_json(work / "run_metadata.json", record)
        (work / "report.md").write_text("# Laboratory audit\n\nOpen lab_review.html and review the inventories, units, specimens, time and ownership diagnostics. "
                                       "No mapping or conversion is approved automatically. Complete current-run review CSVs, then run review mode.\n", encoding="utf-8", newline="\n")
        details.update(audit_checks_passed=True, counts=record["counts"], inventory_branches=len(record["inventory"]))
    return 0


def review_mode(context, reviews=None):
    store = ManifestStore(context.layout, data=context)
    prior_ref = store.manifest_ref("step10", "audit")
    prior = store.validate(prior_ref)
    record = read_json(context.layout.resolve(store.output_ref(prior, "lab_audit.json")))
    settings = record["settings"]
    paths = settings["reviews"] if reviews is None else reviews
    require(bool(paths), "MISSING_REVIEW", "Provide completed current-run review files")
    configs = [context.config_ref(path) for path in paths]
    files = [context.check_config_ref(ref) for ref in configs]
    details = {"audit_generation": prior["generation"]}
    with store.publish("step10", "review", inputs=[prior_ref], configs=configs, details=details,
                       required_bytes=settings["workspace_estimate_bytes"]) as work:
        run = AuditWorkspace(work, context.layout)
        with tempfile.TemporaryDirectory(dir=work, prefix=".review_") as scratch:
            with open_store(Path(scratch), settings["memory_limit"], max_temp_directory_size=settings["max_temp_directory_size"]) as connection:
                cache = {name: context.layout.resolve(store.output_ref(prior, path)) for name, path in CACHE_NAMES.items()}
                dictionary = audit.load_cache(connection, cache)
                cohort_manifest = store.validate(store.manifest_ref("step08", "freeze"))
                cohort = pq.read_table(context.layout.resolve(store.output_ref(cohort_manifest, "cohort.parquet")))
                audit.build_views(connection, cohort)
                decisions, acknowledgment = review.parse_reviews(files, record["inventory"], dictionary, record, settings, run)
                state, progress = review.review_progress(connection, record["inventory"], decisions, acknowledgment)
                conversions = review.validate_retained_conflicts(connection, decisions)
        write_csv(work / "review_progress.csv", ["review_key", "itemid", "source_unit", "status"], progress)
        columns = ["review_key", "branch_id", "feature_name", "missing_specimen", "numeric_disagreement", "conflicting_id", "numeric_records", "nonfinite_conversions", "converted_min", "converted_max"]
        write_csv(work / "lab_conversion_checks.csv", columns, conversions)
        run.checks.write_report(work / "csv_tail_checks.csv")
        result = {"data_run": context.key, "audit_generation": prior["generation"], **state,
                  "decisions": decisions, "scope_acknowledgment": acknowledgment, "review_files": configs}
        write_json(work / "review_state.json", result)
        write_json(work / "run_metadata.json", {**result, "mode": "review", "dictionary_frozen": False, "ehr_features_ready": False})
        details.update(**state)
    return 0


def freeze_mode(context):
    store = ManifestStore(context.layout, data=context)
    review_ref = store.manifest_ref("step10", "review")
    reviewed = store.validate(review_ref)
    state = read_json(context.layout.resolve(store.output_ref(reviewed, "review_state.json")))
    require(state.get("data_run") == context.key and state.get("review_complete") is True
            and state.get("scope_acknowledgment_complete") is True,
            "LAB_REVIEW_NOT_COMPLETE", "Resolve all branches and confirm scope before freezing")
    audit_ref = store.manifest_ref("step10", "audit")
    prior = store.validate(audit_ref)
    require(state["audit_generation"] == prior["generation"], "REVIEW_GENERATION_MISMATCH", "Review belongs to a different audit")
    record = read_json(context.layout.resolve(store.output_ref(prior, "lab_audit.json")))
    decisions = review.expanded_decisions(record["inventory"], state["decisions"], state["scope_acknowledgment"])
    table = review.dictionary_table(decisions, context.key)
    details = {"dictionary_frozen": True, "audit_generation": prior["generation"], "review_generation": reviewed["generation"]}
    with store.publish("step10", "freeze", inputs=[review_ref], details=details, data_files=("lab_dictionary.parquet",)) as work:
        pq.write_table(table, work / "lab_dictionary.parquet", compression="zstd")
        result = {"data_run": context.key, "dictionary_version": context.key, "dictionary_frozen": True,
                  "review_complete": True, "ehr_features_ready": False,
                  "audit": {"manifest": audit_ref, "generation": prior["generation"]},
                  "review": {"manifest": review_ref, "generation": reviewed["generation"]},
                  "retained_rule_rows": sum(row["keep"] for row in decisions),
                  "excluded_rule_rows": sum(not row["keep"] for row in decisions),
                  "frozen_utc": datetime.now(timezone.utc).isoformat()}
        write_json(work / "lab_dictionary_freeze.json", result)
        write_json(work / "run_metadata.json", result)
        details.update(checks_passed=True, retained_rule_rows=result["retained_rule_rows"], excluded_rule_rows=result["excluded_rule_rows"])
    return 0


def run_step10(context, mode, config_path="ehr/lab_processing.json", reviews=None):
    require(mode in {"audit", "review", "freeze"}, "INVALID_MODE", "Select audit, review or freeze")
    require(mode == "review" or reviews is None, "INVALID_MODE", "Review overrides apply only to review mode")
    if mode == "audit":
        return audit_mode(context, config_path)
    if mode == "review":
        return review_mode(context, reviews)
    return freeze_mode(context)
