"""Audit cached EHR features and publish a checked, unprocessed feature table."""

import re
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from src.cohort.burden_io import open_store
from src.cohort.burden_support import AuditWorkspace, write_csv
from src.cohort.raw_io import _file_version
from src.common.manifest import ManifestStore, write_json
from src.common.paths import read_json, require
from src.ehr import build_ehr_features as upstream
from src.ehr import feature_engine as engine
from src.ehr import lab_audit as lab


def load_settings(context, path):
    settings = context.read(path, required=("algorithm_version", "lookback_hours", "static_features", "aggregations",
                                           "tables", "memory_limit", "max_temp_directory_size", "workspace_estimate_bytes"))
    require(settings["algorithm_version"] == "ehr_features_v1" and settings["lookback_hours"] == 24
            and settings["static_features"] == engine.STATIC and settings["aggregations"] == engine.AGGREGATIONS
            and settings["tables"] == {name: f"hosp/{name}.csv" for name in engine.SOURCE_COLUMNS},
            "UNSUPPORTED_FEATURE_POLICY", "Use the supported 24-hour laboratory and static feature definitions")
    require(not {"audit_version", "cohort_version", "split_version", "dictionary_version", "feature_set_version"} & set(settings),
            "INVALID_FEATURE_CONFIG", "Data identities come from the selected version/run directory")
    require(set(settings) == {"algorithm_version", "lookback_hours", "static_features", "aggregations", "tables",
                              "memory_limit", "max_temp_directory_size", "workspace_estimate_bytes"},
            "INVALID_FEATURE_CONFIG", "Unknown feature settings are not supported")
    for name in ("memory_limit", "max_temp_directory_size"):
        require(isinstance(settings[name], str) and re.fullmatch(r"[1-9][0-9]*(MB|GB)", settings[name]),
                "INVALID_FEATURE_CONFIG", "Use explicit MB or GB resource bounds")
    require(type(settings["workspace_estimate_bytes"]) is int and settings["workspace_estimate_bytes"] > 0,
            "INVALID_FEATURE_CONFIG", "Set a positive disk-space estimate")
    return settings


def read_output(context, store, manifest, name, *, json=False):
    path = context.layout.resolve(store.output_ref(manifest, name))
    return read_json(path) if json else pq.read_table(path)


def load_inputs(context, store):
    frozen = store.validate(store.manifest_ref("step10", "freeze"))
    record = read_output(context, store, frozen, "lab_dictionary_freeze.json", json=True)
    require(record.get("data_run") == context.key and record.get("dictionary_version") == context.key
            and record.get("dictionary_frozen") is True and record.get("review_complete") is True,
            "DICTIONARY_NOT_FROZEN", "Use the current-run reviewed dictionary")
    rules = engine.validate_dictionary(read_output(context, store, frozen, "lab_dictionary.parquet"), context.key)
    cohort_manifest = store.validate(store.manifest_ref("step08", "freeze"))
    split_manifest = store.validate(store.manifest_ref("step09", "split"))
    return {
        "rules": rules,
        "cohort": read_output(context, store, cohort_manifest, "cohort.parquet"),
        "evidence": read_output(context, store, cohort_manifest, "cohort_audit.parquet"),
        "patients": read_output(context, store, split_manifest, "patient_split.parquet"),
    }


def perform_audit(connection, run, context, store, sources, settings):
    cohort = upstream.verify_upstream(context, store, connection)
    inputs = load_inputs(context, store)
    require(cohort.equals(inputs["cohort"]), "COHORT_MISMATCH", "Validated cohort differs from frozen inputs")
    prior = store.validate(store.manifest_ref("step10", "audit"))
    lab_record = read_output(context, store, prior, "lab_audit.json", json=True)
    require(lab_record.get("data_run") == context.key and lab_record.get("audit_checks_passed") is True,
            "LAB_AUDIT_NOT_PASSED", "A successful current-run laboratory audit is required")
    cache = {name: context.layout.resolve(store.output_ref(prior, path)) for name, path in upstream.CACHE_NAMES.items()}
    lab.load_cache(connection, cache)
    lab.build_views(connection, cohort)
    connection.register("cohort_patients", cohort.select(["subject_id"]))
    connection.register("cohort_admissions", cohort.select(["hadm_id"]))
    for name, columns in engine.SOURCE_COLUMNS.items():
        condition = "try_cast(r.subject_id AS BIGINT) IN (SELECT subject_id FROM cohort_patients)"
        if name in {"admissions", "services"}:
            condition += " OR try_cast(r.hadm_id AS BIGINT) IN (SELECT hadm_id FROM cohort_admissions)"
        lab.scan_csv(connection, sources[name], f"static_{name}", columns.split(), [], condition, run)
        lab.copy_parquet(connection, f"static_{name}", run.path(Path(f"data/cache/{name}.parquet")))
    static, issues = engine.static_features(connection, cohort, inputs["evidence"])
    lab_error = None
    try:
        engine.clean_labs(connection, inputs["rules"])
    except ValueError as error:
        lab_error = str(error)
        issues.append(dict(sample_id="", source="labevents", field="reviewed_rules", reason="laboratory_conflict", evidence=lab_error))
    write_csv(run.path(Path("reports/static_conflicts.csv")), engine.ISSUE_COLUMNS, [row for row in issues if row["source"] != "labevents"])
    write_csv(run.path(Path("reports/blocking_conflicts.csv")), engine.ISSUE_COLUMNS, issues)
    result = {"audit_checks_passed": not issues, "blocking_conflicts": len(issues), "candidate_available": False,
              "studies": cohort.num_rows, "numeric_features": 0, "categorical_features": 0}
    if lab_error:
        run.path(Path("reports/ehr_audit_report.md")).write_text(
            "# EHR audit blocked\n\nInspect blocking_conflicts.csv. Resolve laboratory conflicts and rerun audit before build.\n", encoding="utf-8", newline="\n")
        return result
    schema = engine.schema_for(inputs["rules"], context.key)
    rows = engine.aggregate_labs(connection, cohort, inputs["rules"])
    for row in rows:
        row.update(static[row["sample_id"]])
    table = engine.float32_table(rows, schema)
    engine.validate_wide(table, cohort, schema)
    summary, missingness = engine.feature_reports(connection, table, cohort, inputs["patients"], schema)
    pq.write_table(table, run.path(Path("data/cache/ehr_features_candidate.parquet")), compression="zstd")
    lab.copy_parquet(connection, "lab_trace", run.path(Path("data/cache/lab_cleaning_trace.parquet")))
    write_json(run.path(Path("reports/feature_schema_candidate.json")), schema)
    write_csv(run.path(Path("reports/ehr_feature_summary.csv")), list(summary[0]), summary)
    write_csv(run.path(Path("reports/lab_missingness.csv")), list(missingness[0]), missingness)
    lab.write_query_csv(connection, run, Path("reports/lab_cleaning_summary.csv"),
                        "SELECT feature_name,disposition,count(*) AS study_associations,count(DISTINCT lab_record_key) AS source_records FROM lab_trace GROUP BY ALL ORDER BY ALL")
    lab.write_query_csv(connection, run, Path("reports/lab_window_rejections.csv"),
                        "SELECT itemid,availability_reason,ownership_reason,count(*) AS study_associations,count(DISTINCT lab_record_key) AS source_records FROM associations WHERE availability_reason<>'available' OR ownership_reason<>'owned' GROUP BY ALL ORDER BY ALL")
    result.update(candidate_available=True, numeric_features=len(schema["numeric_features"]), categorical_features=len(schema["categorical_features"]))
    report = (
        "# EHR feature audit\n\n"
        f"Studies: {len(rows)}. Numeric features: {result['numeric_features']}. Categorical features: {result['categorical_features']}. Blocking conflicts: {len(issues)}.\n\n"
        "Inspect blocking_conflicts.csv first, then ehr_feature_summary.csv and lab_missingness.csv for all-missing, constant and changed coverage bands. "
        "All approved items are retained regardless of cleaned missingness. Numeric missing values remain NaN; categorical missing values remain __MISSING__. "
        "No imputation, scaling, category vocabulary or model fitting is applied.\n\n"
        "The laboratory trace records sample, source record, specimen, charttime, storetime and revision disposition. "
        "Source-record counts and study-association counts have separate denominators. Reference ranges are not validity limits. "
        "Service availability uses transfertime; deidentified anchor ages cannot recover true ages above 89. "
        "Audit publishes a candidate, not the final feature table. Review these reports before build.\n"
    )
    run.path(Path("reports/ehr_audit_report.md")).write_text(report, encoding="utf-8", newline="\n")
    return result


def audit_mode(context, config_path):
    settings = load_settings(context, config_path)
    store = ManifestStore(context.layout, data=context)
    sources = {name: context.layout.path("mimic_iv", path) for name, path in settings["tables"].items()}
    versions = {name: _file_version(path) for name, path in sources.items()}
    require(all(value["exists"] for value in versions.values()), "MISSING_SOURCE", "A static-feature source is missing")
    details = {"lookback_hours": 24, "horizon_hours": 72}
    refs = [store.manifest_ref("step10", "freeze"), store.manifest_ref("step09", "split")]
    with store.publish("step11", "audit", inputs=refs, configs=[context.config_ref(config_path)],
                       data_directories=("cache",), details=details, required_bytes=settings["workspace_estimate_bytes"]) as work:
        run = AuditWorkspace(work, context.layout)
        with tempfile.TemporaryDirectory(dir=work, prefix=".features_") as scratch:
            with open_store(Path(scratch), settings["memory_limit"], max_temp_directory_size=settings["max_temp_directory_size"]) as connection:
                result = perform_audit(connection, run, context, store, sources, settings)
        require(versions == {name: _file_version(path) for name, path in sources.items()},
                "SOURCE_CHANGED", "Static-feature sources changed during audit")
        run.checks.write_report(work / "csv_tail_checks.csv")
        result.update(data_run=context.key, mode="audit", settings=settings, source_versions=versions,
                      sources={name: context.layout.ref("mimic_iv", path) for name, path in settings["tables"].items()},
                      source_csv_tail_checks=run.csv_tail_summary(), ehr_features_ready=False)
        write_json(work / "ehr_audit.json", result)
        write_json(work / "run_metadata.json", result)
        details.update({name: result[name] for name in ("audit_checks_passed", "blocking_conflicts", "candidate_available", "studies", "numeric_features", "categorical_features")})
    return 0 if result["audit_checks_passed"] else 2


def build_mode(context):
    store = ManifestStore(context.layout, data=context)
    ref = store.manifest_ref("step11", "audit")
    prior = store.validate(ref)
    record = read_output(context, store, prior, "ehr_audit.json", json=True)
    require(record.get("data_run") == context.key and record.get("audit_checks_passed") is True
            and record.get("blocking_conflicts") == 0 and record.get("candidate_available") is True,
            "FEATURE_AUDIT_BLOCKED", "Resolve audit conflicts and rerun audit before build")
    details = {"ehr_features_ready": True, "audit_generation": prior["generation"]}
    with store.publish("step11", "build", inputs=[ref], data_files=("ehr_features_raw.parquet", "feature_schema.json"),
                       details=details, required_bytes=record["settings"]["workspace_estimate_bytes"]) as work:
        inputs = load_inputs(context, store)
        schema = read_output(context, store, prior, "feature_schema_candidate.json", json=True)
        require(schema == engine.schema_for(inputs["rules"], context.key), "FEATURE_SCHEMA_MISMATCH", "Candidate schema differs from reviewed definitions")
        table = read_output(context, store, prior, "cache/ehr_features_candidate.parquet")
        engine.validate_wide(table, inputs["cohort"], schema)
        pq.write_table(table, work / "ehr_features_raw.parquet", compression="zstd")
        write_json(work / "feature_schema.json", schema)
        result = {"data_run": context.key, "mode": "build", "feature_set_version": context.key,
                  "audit": {"manifest": ref, "generation": prior["generation"]}, "ehr_features_ready": True,
                  "audit_checks_passed": True, "blocking_conflicts": 0, "studies": table.num_rows,
                  "numeric_features": len(schema["numeric_features"]), "categorical_features": len(schema["categorical_features"]),
                  "preprocessing_applied": False, "source_csv_tail_checks": record["source_csv_tail_checks"]}
        write_json(work / "ehr_features_freeze.json", result)
        write_json(work / "run_metadata.json", result)
        details.update(studies=table.num_rows, numeric_features=result["numeric_features"], categorical_features=result["categorical_features"])
    return 0


def run_step11(context, mode, config_path="ehr/features.json"):
    require(mode in {"audit", "build"}, "INVALID_MODE", "Select audit or build")
    return audit_mode(context, config_path) if mode == "audit" else build_mode(context)
