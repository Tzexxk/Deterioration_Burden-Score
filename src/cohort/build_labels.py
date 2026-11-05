"""Publish formal 72-hour labels from a generation-bound frozen audit."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.cohort import audit_burden_labels as audit
from src.cohort.audit_burden_labels import DATA_ARTIFACTS
from src.cohort.burden_io import COMPONENTS, open_store
from src.cohort.burden_support import AuditWorkspace
from src.cohort.burden_validation import validate_labels
from src.common.manifest import ManifestStore, write_json
from src.common.paths import read_json, require

LABEL_SCHEMA_VERSION = "burden_v2_labels_v1"
DATA_OUTPUTS = ("study_burden_labels.parquet", "event_evidence.parquet")


def frozen_inputs(context, store):
    freeze_ref = store.manifest_ref("step05", "freeze")
    audit_ref = store.manifest_ref("step05", "audit")
    frozen = store.validate(freeze_ref)
    prior = store.validate(audit_ref)
    decision = read_json(context.layout.resolve(store.output_ref(frozen, "endpoint_decision.json")))
    expected = {"decision_status": "frozen", "horizon_frozen": True,
                "label_rules_frozen": True, "version": context.version, "run": context.run,
                "horizon_hours": 72, "checks_passed": True, "sample_unit": "study",
                "outcome_version": "burden_v2", "event_window": "(t0, min(t0+H, dischtime)]",
                "target_formula": "max(icu,stepdown)+2*max(imv,hypoxemia)+2*vaso_inotrope+3*death"}
    require(all(type(decision.get(k)) is type(v) and decision[k] == v for k, v in expected.items()),
            "INVALID_ENDPOINT_DECISION", "A checked, frozen 72-hour decision is required")
    require(decision["audit"] == {"manifest": audit_ref, "generation": prior["generation"]}
            and frozen["inputs"] == [decision["audit"]]
            and frozen["details"].get("audit_generation") == prior["generation"],
            "INVALID_ENDPOINT_DECISION", "Decision must bind the current audit generation")
    policy = decision.get("review_policy")
    require(policy in {"lenient", "strict"} and all(m["details"].get("review_policy") == policy for m in (prior, frozen)),
            "REVIEW_POLICY_MISMATCH", "Use a supported and consistently frozen review policy")
    require(all(isinstance(decision.get(k), str) and decision[k].strip()
                for k in ("decision_version", "reviewer", "reviewed_on", "rationale")),
            "INVALID_ENDPOINT_DECISION", "Frozen review provenance is incomplete")
    artifacts = {key: store.output_ref(prior, name) for key, name in zip(
        ("candidates", "candidate_evidence", "observations", "wide", "evidence"), DATA_ARTIFACTS, strict=True)}
    require(decision.get("artifacts") == artifacts, "INVALID_ENDPOINT_DECISION", "Frozen artifact references differ")
    snapshot_ref = store.output_ref(prior, "effective_configuration.json")
    require(decision.get("effective_configuration") == snapshot_ref,
            "INVALID_ENDPOINT_DECISION", "Frozen configuration snapshot differs")
    snapshot = read_json(context.layout.resolve(snapshot_ref))
    metadata = read_json(context.layout.resolve(store.output_ref(prior, "run_metadata.json")))
    for item in (snapshot, metadata):
        require(item.get("data_run") == context.key and item.get("audit_generation") == prior["generation"]
                and item.get("review_policy") == policy,
                "INVALID_ENDPOINT_DECISION", "Audit snapshot identity differs")
    require(all(metadata.get(k) is True for k in ("audit_checks_passed", "dictionary_handling_review_complete", "audit_ready_for_window_decision")),
            "AUDIT_NOT_READY", "Audit must be ready for formal labels")
    require(all(decision[k] == metadata[k] for k in ("rules_version", "dictionary_version", "event_window", "target_formula")),
            "INVALID_ENDPOINT_DECISION", "Frozen rules differ from the audit snapshot")
    return frozen, prior, decision, metadata, snapshot, {k: context.layout.resolve(v) for k, v in artifacts.items()}


def _write_query(connection, run, path, query):
    connection.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(run.path(path))])


def _publish_tables(connection, run, context, decision, audit_manifest, freeze_manifest):
    data, audit_dir = Path("data"), Path("reports")
    connection.execute(
        "CREATE TABLE publication_version(decision_version VARCHAR, rules_version VARCHAR, dictionary_version VARCHAR, label_schema_version VARCHAR, data_version INTEGER, data_run INTEGER, source_audit_generation INTEGER, source_freeze_generation INTEGER)"
    )
    connection.execute(
        "INSERT INTO publication_version VALUES (?,?,?,?,?,?,?,?)",
        [
            decision["decision_version"],
            decision["rules_version"],
            decision["dictionary_version"],
            LABEL_SCHEMA_VERSION,
            context.version,
            context.run,
            audit_manifest["generation"],
            freeze_manifest["generation"],
        ],
    )
    _write_query(
        connection,
        run,
        data / "study_burden_labels.parquet",
        "SELECT * FROM scored CROSS JOIN publication_version ORDER BY sample_id",
    )
    _write_query(
        connection,
        run,
        data / "event_evidence.parquet",
        "SELECT e.* FROM evidence e JOIN eligible USING(sample_id) ORDER BY sample_id,component,source_record_key,event_time",
    )
    flags = [f"early_3h_{c}" for c in COMPONENTS]
    early = ",".join(
        f"cast(coalesce({c}_first_event_time<=least(t0+INTERVAL '3 hours',observation_end),false) AS UTINYINT) {name}"
        for c, name in zip(COMPONENTS, flags)
    )
    times = ",".join(f"{c}_first_event_time" for c in COMPONENTS)
    connection.execute(
        f"CREATE VIEW early AS SELECT *, {early}, least({times}) first_early_event_time FROM scored"
    )
    formula = f"greatest({flags[0]},{flags[1]})+2*greatest({flags[2]},{flags[3]})+2*{flags[4]}+3*{flags[5]}"
    audit.write_query(
        connection,
        run,
        audit_dir / "early_3h_events.csv",
        f"SELECT sample_id,subject_id,hadm_id,study_id,t0,horizon_hours,observation_end,{','.join(flags)},first_early_event_time,{formula} early_3h_score,burden_score FROM early WHERE first_early_event_time<=least(t0+INTERVAL '3 hours',observation_end) ORDER BY sample_id",
    )
    audit.write_query(
        connection,
        run,
        audit_dir / "early_discharge.csv",
        "SELECT sample_id,subject_id,hadm_id,study_id,t0,dischtime,horizon_hours,observation_end,observed_hours,"
        + ",".join(COMPONENTS)
        + ",burden_score,phys_death,high_burden FROM scored WHERE early_discharge=1 ORDER BY sample_id",
    )
    component_counts = audit.query_rows(
        connection,
        "SELECT "
        + ",".join(
            f"count(*) FILTER(WHERE {c}=1) {c}_studies,count(DISTINCT subject_id) FILTER(WHERE {c}=1) {c}_patients"
            for c in COMPONENTS
        )
        + " FROM scored",
    )[0]
    source_event_counts = audit.query_rows(
        connection,
        "SELECT component,count(DISTINCT source_record_key) source_record_count "
        "FROM evidence JOIN eligible USING(sample_id) WHERE qualifies=1 "
        "GROUP BY component ORDER BY component",
    )
    distribution = audit.query_rows(
        connection,
        "SELECT r.range score,count(s.sample_id) study_count FROM range(9) r LEFT JOIN scored s ON s.burden_score=r.range GROUP BY r.range ORDER BY r.range",
    )
    early_summary = audit.query_rows(
        connection,
        "SELECT count(*) study_count,count(DISTINCT subject_id) patient_count FROM early WHERE first_early_event_time<=least(t0+INTERVAL '3 hours',observation_end)",
    )[0]
    early_scores = audit.query_rows(
        connection,
        f"SELECT {formula} early_3h_score,count(*) study_count FROM early WHERE first_early_event_time<=least(t0+INTERVAL '3 hours',observation_end) GROUP BY ALL ORDER BY 1",
    )
    discharge_summary = audit.query_rows(
        connection,
        "SELECT count(*) study_count,count(DISTINCT subject_id) patient_count,avg(cast(burden_score=0 AS FLOAT)) zero_score_proportion,min(observed_hours) min_hours,median(observed_hours) median_hours,max(observed_hours) max_hours FROM scored WHERE early_discharge=1",
    )[0]
    coverage = audit.query_rows(
        connection,
        "SELECT "
        + ",".join(
            f"count(*) FILTER(WHERE {c}=0) {c}_missing_studies"
            for c in (
                "spo2_source_available",
                "emar_source_available",
                "procedure_source_available",
            )
        )
        + " FROM scored",
    )[0]
    return {"component_counts": component_counts, "qualifying_source_counts": source_event_counts,
            "score_distribution": distribution, "early_3h": early_summary,
            "early_3h_score_distribution": early_scores, "early_discharge": discharge_summary,
            "source_coverage": coverage}


def build_burden_labels(context):
    started = datetime.now(timezone.utc)
    store = ManifestStore(context.layout, data=context)
    frozen, prior, decision, source_metadata, snapshot, artifacts = frozen_inputs(context, store)
    ref = store.manifest_ref("step05", "freeze")
    details = {"review_policy": decision["review_policy"], "horizon_hours": 72,
               "audit_generation": prior["generation"], "freeze_generation": frozen["generation"]}
    required_bytes = sum(p.stat().st_size for p in artifacts.values()) * 4 + 8 * 1024 * 1024
    with store.publish("step06", "build", inputs=[ref], details=details,
                       data_files=DATA_OUTPUTS, required_bytes=required_bytes) as work:
        run = AuditWorkspace(work, context.layout)
        with tempfile.TemporaryDirectory(dir=work, prefix=".labels_") as scratch:
            settings = snapshot["ehr"]["burden_v2"]
            with open_store(Path(scratch), settings["memory_limit"],
                            max_temp_directory_size=settings["max_temp_directory_size"]) as connection:
                counts = validate_labels(connection, artifacts, 72, review_policy=decision["review_policy"])
                require(counts == decision["counts"] == frozen["details"]["counts"],
                        "LABEL_COUNT_MISMATCH", "Reconstructed labels differ from the frozen decision")
                summaries = _publish_tables(connection, run, context, decision, prior, frozen)
        run.checks.write_report(work / "csv_tail_checks.csv")
        metadata = {"data_run": context.key, "outcome_version": "burden_v2",
                    "label_schema_version": LABEL_SCHEMA_VERSION, "horizon_hours": 72,
                    "review_policy": decision["review_policy"], "horizon_frozen": True,
                    "label_rules_frozen": True, "formal_labels_published": True,
                    "label_checks_passed": True, "common_cohort_frozen": False,
                    "image_qc_performed": False, "decision_version": decision["decision_version"],
                    "source_freeze": {"manifest": ref, "generation": frozen["generation"]},
                    "source_audit": decision["audit"], "rules_version": decision["rules_version"],
                    "dictionary_version": decision["dictionary_version"],
                    "effective_configuration": decision["effective_configuration"],
                    "csv_tail_checks": run.csv_tail_summary(),
                    "source_csv_tail_checks": source_metadata["csv_tail_checks"],
                    "counts": counts, **summaries,
                    "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
        write_json(work / "run_metadata.json", metadata)
        text = ("# Formal 72-hour burden labels\n\n"
                f"Data run: {context.key}. Review policy: {decision['review_policy']}. "
                f"Audit generation: {prior['generation']}. Freeze generation: {frozen['generation']}.\n\n"
                "Evidence ownership, event boundaries, first times and independently reconstructed scores passed validation.\n"
                + "\n".join(f"- {key}: {value}" for key, value in counts.items())
                + "\n\nSee run_metadata.json for component counts, score distributions, early events, "
                "early discharge and source coverage; early_3h_events.csv and early_discharge.csv contain details.\n\n"
                "This step reads no raw clinical CSV. csv_tail_checks.csv records no_csv_inputs; "
                "source CSV warnings remain in the Step05 audit. Missing recorded events are operational zeros, "
                "not proof of clinical absence. These labels precede image QC and common cohort freezing. "
                "The source audit and endpoint decision remain unchanged.\n")
        (work / "report.md").write_text(text, encoding="utf-8", newline="\n")
        details.update(checks_passed=True, counts=counts, formal_labels_published=True)
    return 0
