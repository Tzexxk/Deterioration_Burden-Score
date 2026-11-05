"""Freeze the common intersection of validated formal labels and reviewed images."""

import copy
import json
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import audit_burden_labels as audit
from src.cohort.build_labels import LABEL_SCHEMA_VERSION, frozen_inputs
from src.cohort.burden_io import COMPONENTS, open_store
from src.cohort.burden_support import AuditWorkspace, write_csv
from src.cohort.burden_validation import _reject, validate_labels
from src.cohort.raw_io import _file_version
from src.common.manifest import ManifestStore, write_json
from src.common.paths import read_json, require
from src.imaging import prepare_images as images
from src.imaging.preprocessing import validate_pixels, validate_settings
from src.utils import master_reuse

PUBLICATION_FIELDS = "decision_version rules_version dictionary_version label_schema_version data_version data_run source_audit_generation source_freeze_generation".split()
QC_FIELDS = {"npy_relpath": "npy_relpath", "processing_version": "processing_version",
             "processing_status": "processing_status", "qc_status": "qc_status", "qc_reason": "qc_reason",
             "qc_reviewer": "reviewer", "qc_reviewed_on": "reviewed_on", "qc_review_rationale": "review_rationale"}
TABLE_NAMES = ("cohort.parquet", "cohort_audit.parquet")


def _counts(rows: list[dict]) -> dict:
    return {
        "studies": len(rows),
        "admissions": len({r["hadm_id"] for r in rows}),
        "patients": len({r["subject_id"] for r in rows}),
    }


def verify_inputs(data_context, store):
    frozen, prior, decision, source05, clinical_snapshot, artifacts = frozen_inputs(data_context, store)
    labels, indexed, _ = images.load_inputs(data_context, store)
    label_manifest = store.validate(store.manifest_ref("step06", "build"))
    image_manifest = store.validate(store.manifest_ref("step07", "prepare"))
    def image_output(name):
        return data_context.layout.resolve(store.output_ref(image_manifest, name))
    metadata = read_json(image_output("run_metadata.json"))
    snapshot = read_json(image_output("effective_configuration.json"))
    require(snapshot.get("data_run") == metadata.get("data_run") == data_context.key
            and snapshot.get("processing") == metadata.get("processing_settings"),
            "INVALID_IMAGE_SNAPSHOT", "Image configuration snapshot differs from published metadata")
    settings = snapshot["processing"]
    validate_settings(settings)
    require(all(metadata.get(key) is True for key in ("image_checks_passed","qc_review_complete","ready_for_cohort_freeze")),
            "IMAGE_QC_NOT_READY", "Resolve all required image reviews before cohort freeze")
    tables = [pq.read_table(image_output(name)) for name in ("image_processing.parquet", "image_qc_status.parquet")]
    require(all(table.schema == images.SCHEMA for table in tables) and tables[0].equals(tables[1]),
            "INVALID_IMAGE_TABLE", "Processing and QC tables disagree")
    records = tables[1].to_pylist()
    by_sample = {row["sample_id"]: row for row in records}
    require(len(by_sample) == len(records) and set(by_sample) == {row["sample_id"] for row in labels},
            "INVALID_IMAGE_TABLE", "QC keys must exactly cover formal labels")
    for label in labels:
        row = by_sample[label["sample_id"]]
        require(all(row[key] == label[key] for key in images.KEY_COLUMNS),
                "INVALID_IMAGE_TABLE", "QC sample or image identity differs from formal labels")
        require(row["qc_status"] in {"pass", "fail"} and row["review_complete"] is True,
                "IMAGE_QC_NOT_READY", "Unresolved QC status")
        require(row["processing_version"] == settings["processing_version"]
                and json.loads(row["source_version"]) == _file_version(indexed[row["sample_id"]]["path"]),
                "SOURCE_CHANGED", "Image source version or processing version changed")
    reviews = {row["sample_id"]: row for row in snapshot["reviews"]}
    require(len(reviews) == len(snapshot["reviews"]), "INVALID_IMAGE_SNAPSHOT", "Duplicate review snapshot keys")
    replay = copy.deepcopy(records)
    # Duplicate groups are persisted identities, not content checksums. Rebuild the original
    # grouping equivalence classes to replay sampling and decisions without raw DICOM reads.
    for row in replay:
        row["issues"] = json.loads(row["issues"])
        groups = json.loads(row["duplicate_groups"])
        require(isinstance(groups, list) and all(isinstance(group, str) for group in groups),
                "INVALID_IMAGE_TABLE", "Invalid duplicate groups")
        for field in ("_source_duplicate", "_decoded_duplicate", "_prepared_duplicate"):
            prefix = field.removeprefix("_") + ":"
            matched = [group for group in groups if group.startswith(prefix)]
            require(len(matched) <= 1, "INVALID_IMAGE_TABLE", "Ambiguous duplicate group")
            row[field] = matched[0] if matched else "unique:" + row["sample_id"]
            if field == "_prepared_duplicate" and not row["npy_relpath"]:
                row[field] = ""
    images.assign_qc(replay, settings, reviews)
    fields = "qc_status auto_qc_status qc_reason review_required review_complete sampled_for_review reviewer reviewed_on review_rationale".split()
    require(all(all(a[key] == b[key] for key in fields) for a,b in zip(records,replay,strict=True)),
            "INVALID_IMAGE_REVIEW", "QC sampling and decisions do not reproduce the snapshot")
    counts = {status: _counts([row for row in records if row["qc_status"] == status]) for status in ("pass","fail","pending_review")}
    require(counts == metadata["qc_counts"] and _counts(records) == {key:metadata["total_"+key] for key in ("studies","admissions","patients")}
            and metadata["unreviewed_studies"] == 0 and counts["pass"]["studies"] > 0
            and metadata["review_required_studies"] == sum(row["review_required"] for row in records),
            "IMAGE_COUNT_MISMATCH", "QC counts do not reconcile")
    for row in records:
        if row["qc_status"] != "pass":
            continue
        expected = f"images/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.npy"
        require(row["processing_status"] == "complete" and row["npy_relpath"] == expected,
                "INVALID_IMAGE_TABLE", "Included NPY layout or transform is invalid")
        path = image_output(row["npy_relpath"])
        validate_pixels(np.load(path,allow_pickle=False),settings["size"])
    return {"data_context": data_context, "decision": decision, "artifacts": artifacts,
            "labels_path": data_context.layout.resolve(store.output_ref(label_manifest,"study_burden_labels.parquet")),
            "evidence_path": data_context.layout.resolve(store.output_ref(label_manifest,"event_evidence.parquet")),
            "qc_records": records, "source05": source05, "source07": metadata,
            "clinical_snapshot": clinical_snapshot, "release_version": data_context.key,
            "audit_generation": prior["generation"], "endpoint_generation": frozen["generation"],
            "image_generation": image_manifest["generation"]}


def validate_cohort(connection, context: dict) -> tuple[pa.Table, pa.Table]:
    """Reconstruct final evidence, then retain original label types in the intersection."""
    decision = context["decision"]
    artifacts = {**context["artifacts"], "evidence": context["evidence_path"]}
    policy = decision["review_policy"]
    counts = validate_labels(
        connection, artifacts, decision["horizon_hours"], review_policy=policy
    )
    allowed_status = (
        "(review_status IN ('not_required','reviewed') "
        "AND event_status IN ('positive','preexisting_only','no_qualifying_record'))"
    )
    if policy == "strict":
        # validate_labels has already checked the uncertainty classification.
        allowed_status += (
            " OR (qualifies IS NULL AND blocks_event=0 "
            "AND review_status='pending' AND event_status='unresolved')"
        )
    _reject(
        connection,
        f"SELECT count(*) FROM evidence WHERE ({allowed_status}) IS NOT TRUE",
        "Unresolved final evidence status",
    )
    if counts != decision["counts"]:
        raise ValueError("Formal label counts disagree with frozen decision")
    labels = pq.read_table(context["labels_path"])
    wide_schema = pq.read_schema(context["artifacts"]["wide"])
    if (
        len(set(labels.column_names)) != len(labels.column_names)
        or set(labels.column_names) != set(wide_schema.names) | set(PUBLICATION_FIELDS)
        or any(
            labels.schema.field(k).type != field.type
            for k, field in zip(wide_schema.names, wide_schema)
        )
    ):
        raise ValueError("Formal label schema changed")
    connection.register("formal", labels)
    mismatch = " OR ".join(f"l.{c} IS DISTINCT FROM w.{c}" for c in wide_schema.names)
    _reject(
        connection,
        f"SELECT count(*) FROM formal l FULL JOIN scored w USING(sample_id) WHERE l.sample_id IS NULL OR w.sample_id IS NULL OR {mismatch}",
        "Formal labels disagree with reconstructed final evidence",
    )
    _reject(
        connection,
        "SELECT count(*) FROM (SELECT sample_id FROM formal GROUP BY sample_id HAVING count(*)<>1)",
        "Duplicate formal sample key",
    )
    _reject(
        connection,
        "SELECT count(*) FROM evidence e LEFT JOIN formal l USING(sample_id) WHERE l.sample_id IS NULL",
        "Formal evidence contains excluded candidates",
    )
    versions = dict(
        decision_version=decision["decision_version"],
        rules_version=decision["rules_version"],
        dictionary_version=decision["dictionary_version"],
        data_version=context["data_context"].version,
        data_run=context["data_context"].run,
        source_audit_generation=context["audit_generation"],
        source_freeze_generation=context["endpoint_generation"],
        label_schema_version=LABEL_SCHEMA_VERSION,
    )
    label_rows = sorted(labels.to_pylist(), key=lambda r: r["sample_id"])
    if any(any(row[k] != v for k, v in versions.items()) for row in label_rows):
        raise ValueError("Formal label version provenance mismatch")
    qc = {r["sample_id"]: r for r in context["qc_records"]}
    formal = {r["sample_id"]: r for r in label_rows}
    extra_fields = [(k, pa.string()) for k in QC_FIELDS] + [
        ("cohort_version", pa.string())
    ]
    included = []
    for row in label_rows:
        if qc[row["sample_id"]]["qc_status"] == "pass":
            included.append(
                {
                    **row,
                    **{k: qc[row["sample_id"]][v] for k, v in QC_FIELDS.items()},
                    "cohort_version": context["release_version"],
                }
            )
    cohort = pa.Table.from_pylist(
        included, schema=pa.schema(list(labels.schema) + extra_fields)
    )
    audit_fields = (
        [(k, labels.schema.field(k).type) for k in PUBLICATION_FIELDS]
        + extra_fields
        + [
            ("cohort_included", pa.uint8()),
            ("qc_performed", pa.uint8()),
            ("cohort_exclusion_stage", pa.string()),
            ("cohort_exclusion_reason", pa.string()),
        ]
    )
    rows = []
    for row in pq.read_table(context["artifacts"]["wide"]).to_pylist():
        key = row["sample_id"]
        image = qc.get(key)
        selected = image is not None and image["qc_status"] == "pass"
        rows.append(
            {
                **row,
                **{
                    k: formal[key][k] if key in formal else None
                    for k in PUBLICATION_FIELDS
                },
                **{k: image[v] if image else None for k, v in QC_FIELDS.items()},
                "cohort_version": context["release_version"],
                "cohort_included": int(selected),
                "qc_performed": int(image is not None),
                "cohort_exclusion_stage": ""
                if selected
                else "image_qc"
                if image
                else "upstream_candidate",
                "cohort_exclusion_reason": ""
                if selected
                else (image["review_rationale"] or image["qc_reason"])
                if image
                else row["exclusion_reason"],
            }
        )
    cohort_audit = pa.Table.from_pylist(
        sorted(rows, key=lambda r: r["sample_id"]),
        schema=pa.schema(list(wide_schema) + audit_fields),
    )
    connection.register("common", cohort)
    connection.register("cohort_audit", cohort_audit)
    return cohort, cohort_audit


def write_reports(
    connection,
    run: AuditWorkspace,
    directory: Path,
    context: dict,
    cohort: pa.Table,
    candidate_audit: pa.Table,
) -> dict:
    rows, candidates = cohort.to_pylist(), candidate_audit.to_pylist()
    excluded = [r for r in candidates if r["cohort_exclusion_stage"] == "image_qc"]
    flow = [
        {"stage": stage, **_counts(group)}
        for stage, group in (
            ("all_candidates", candidates),
            ("upstream_excluded", [r for r in candidates if r["qc_performed"] == 0]),
            ("formal_labels", [r for r in candidates if r["qc_performed"] == 1]),
            ("qc_excluded", excluded),
            ("common_cohort", rows),
        )
    ]
    removed = {
        "wholly_removed_admissions": len(
            {r["hadm_id"] for r in excluded} - {r["hadm_id"] for r in rows}
        ),
        "wholly_removed_patients": len(
            {r["subject_id"] for r in excluded} - {r["subject_id"] for r in rows}
        ),
    }
    write_csv(
        run.path(directory / "cohort_flow.csv"),
        ["stage", "studies", "admissions", "patients"],
        flow,
    )
    exclusion_columns = [
        *images.KEY_COLUMNS,
        "burden_score",
        "phys_death",
        "high_burden",
        "qc_reason",
        "qc_review_rationale",
        "qc_reviewer",
        "qc_reviewed_on",
    ]
    write_csv(
        run.path(directory / "cohort_qc_exclusions.csv"),
        exclusion_columns,
        [{k: r[k] for k in exclusion_columns} for r in excluded],
    )
    feasibility = []
    for kind, values in (
        (
            "cohort",
            [("total", rows), ("nonzero", [r for r in rows if r["burden_score"] > 0])],
        ),
        (
            "score",
            [
                (str(score), [r for r in rows if r["burden_score"] == score])
                for score in range(9)
            ],
        ),
        (
            "endpoint",
            [
                (name, [r for r in rows if r[name] == 1])
                for name in (
                    *COMPONENTS,
                    "phys_death",
                    "high_burden",
                    "early_discharge",
                )
            ],
        ),
        (
            "coverage",
            [
                (name, [r for r in rows if r[name] == 1])
                for name in (
                    "spo2_source_available",
                    "emar_source_available",
                    "procedure_source_available",
                )
            ],
        ),
        ("qc", [("excluded", excluded)]),
        (
            "early_3h",
            [
                (
                    "any",
                    [
                        r
                        for r in rows
                        if any(
                            r[f"{c}_first_event_time"] is not None
                            and r[f"{c}_first_event_time"]
                            <= min(r["t0"] + timedelta(hours=3), r["observation_end"])
                            for c in COMPONENTS
                        )
                    ],
                )
            ],
        ),
    ):
        feasibility.extend(
            {"category": kind, "value": value, **_counts(group)}
            for value, group in values
        )
    per_patient = Counter(r["subject_id"] for r in rows)
    for count in sorted(set(per_patient.values())):
        feasibility.append(
            {
                "category": "studies_per_patient",
                "value": str(count),
                **_counts([r for r in rows if per_patient[r["subject_id"]] == count]),
            }
        )
    write_csv(
        run.path(directory / "training_feasibility.csv"),
        [
            "category",
            "value",
            "studies",
            "admissions",
            "patients",
            "denominator_studies",
            "study_proportion",
        ],
        [
            {
                **r,
                "denominator_studies": len(context["qc_records"])
                if r["category"] == "qc"
                else len(rows),
                "study_proportion": r["studies"]
                / (len(context["qc_records"]) if r["category"] == "qc" else len(rows)),
            }
            for r in feasibility
        ],
    )
    overlap = audit.query_rows(
        connection,
        "SELECT e.component,e.source_file,e.source_record_key,count(DISTINCT e.sample_id) linked_studies,count(DISTINCT e.hadm_id) linked_admissions,count(DISTINCT e.subject_id) linked_patients,count(DISTINCT e.sample_id) FILTER(WHERE e.qualifies=1) qualifying_linked_studies,string_agg(DISTINCT e.sample_id, ';' ORDER BY e.sample_id) sample_ids FROM evidence e JOIN common c USING(sample_id) GROUP BY e.component,e.source_file,e.source_record_key ORDER BY e.component,e.source_record_key",
    )
    # One stable source key identifies one file within its component.
    _reject(
        connection,
        "SELECT count(*) FROM (SELECT e.component,e.source_record_key FROM evidence e JOIN common c USING(sample_id) GROUP BY ALL HAVING count(DISTINCT source_file)<>1 OR count(DISTINCT e.subject_id)<>1 OR count(DISTINCT e.hadm_id)<>1)",
        "Conflicting stable source ownership",
    )
    write_csv(
        run.path(directory / "source_overlap.csv"),
        [
            "component",
            "source_file",
            "source_record_key",
            "linked_studies",
            "linked_admissions",
            "linked_patients",
            "qualifying_linked_studies",
            "sample_ids",
        ],
        overlap,
    )
    summary = {
        "counts": _counts(rows),
        "candidate_counts": _counts(candidates),
        "flow": flow,
        "qc_loss": {**_counts(excluded), **removed},
        "nonzero_studies": sum(r["burden_score"] > 0 for r in rows),
        "phys_death_studies": sum(r["phys_death"] for r in rows),
        "high_burden_studies": sum(r["high_burden"] for r in rows),
        "feasibility": feasibility,
        "unique_component_source_records": len(overlap),
        "shared_component_source_records": sum(
            r["linked_studies"] > 1 for r in overlap
        ),
        "qualifying_component_source_records": sum(
            r["qualifying_linked_studies"] > 0 for r in overlap
        ),
        "shared_qualifying_component_source_records": sum(
            r["qualifying_linked_studies"] > 1 for r in overlap
        ),
        "source_counts": dict(Counter(r["component"] for r in overlap)),
        "early_discharge_proportion": sum(r["early_discharge"] for r in rows)
        / len(rows),
    }
    text = ("# Common cohort freeze\n\n"
            f"Data run: {context['release_version']}. Horizon: 72 hours.\n\n"
            + "\n".join(f"- {row['stage']}: {row['studies']} studies, {row['admissions']} admissions, {row['patients']} patients" for row in flow)
            + "\n\nLabels, evidence and reviewed QC are reconciled. Upstream exclusions retain null labels "
            "and null QC fields; image exclusions retain formal labels. No new exclusions are introduced. "
            "See cohort_qc_exclusions.csv, training_feasibility.csv and source_overlap.csv for details. "
            "NPYs remain in the generation-bound Step07 image cache and are not copied here. "
            "This is not a final training-package release; patient splits and EHR features follow.\n")
    run.path(directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    return summary


def freeze_master_cohort(context, store, image):
    reuse, _, cohort, audit_table, _, records = master_reuse.rebuild_cohort(context, store, image)
    summary = read_json(master_reuse.copied(store, reuse, "parent_cohort_freeze.json", "step08", "freeze",
                                           "cohort_freeze.json"))["summary"]
    counts = _counts(cohort.to_pylist())
    require(summary["counts"] == counts and summary["candidate_counts"] == _counts(audit_table.to_pylist()),
            "MASTER_COHORT_CHANGED", "Master cohort counts differ from the parent")
    for row in records:
        if row["qc_status"] == "pass":
            path = master_reuse.published(store, image, row["npy_relpath"])
            validate_pixels(np.load(path, allow_pickle=False), 1024)
    parent_run = image["details"]["parent_run"]
    details = {"horizon_hours": 72, "common_cohort_frozen": True, "cohort_checks_passed": True,
               "parent_run": parent_run, "counts": counts, "candidate_counts": summary["candidate_counts"],
               "image_generation": image["generation"]}
    image_ref = {"manifest": image["self"], "generation": image["generation"]}
    with store.publish("step08", "freeze", inputs=[reuse["self"], image["self"]], details=details,
                       data_files=TABLE_NAMES, required_bytes=128 * 1024 * 1024) as work:
        require(store.validate(image["self"]) == image and store.validate(reuse["self"]) == reuse,
                "UPSTREAM_REBUILT", "Master inputs changed during cohort reconstruction")
        for table, name in zip((cohort, audit_table), TABLE_NAMES, strict=True):
            pq.write_table(table, work / name, compression="zstd")
        metadata = {"data_run": context.key, "cohort_version": context.key, **details, "summary": summary,
                    "image_manifest": image_ref,
                    "image_cache": context.layout.ref("processed", f"{context.key}/steps/step07/prepare")}
        write_json(work / "cohort_freeze.json", metadata)
        write_json(work / "run_metadata.json", metadata)
        (work / "report.md").write_text(
            "# Master image cohort\n\nParent membership and clinical values verified. "
            "Image fields reference the new master publication.\n", encoding="utf-8", newline="\n"
        )
    return 0


def freeze_common_cohort(data_context):
    started = datetime.now(timezone.utc)
    store = ManifestStore(data_context.layout,data=data_context)
    image = store.validate(store.manifest_ref("step07", "prepare"))
    if "parent_run" in image["details"]:
        return freeze_master_cohort(data_context, store, image)
    refs = [store.manifest_ref(step,phase) for step,phase in [("step06","build"),("step07","prepare")]]
    details = {"horizon_hours":72}
    with store.publish("step08","freeze",inputs=refs,details=details,data_files=TABLE_NAMES,
                       required_bytes=128*1024*1024) as work:
        context = verify_inputs(data_context,store)
        run = AuditWorkspace(work,data_context.layout)
        with tempfile.TemporaryDirectory(dir=work,prefix=".cohort_") as scratch:
            settings = context["clinical_snapshot"]["ehr"]["burden_v2"]
            with open_store(Path(scratch),settings["memory_limit"],max_temp_directory_size=settings["max_temp_directory_size"]) as connection:
                cohort,candidate_audit = validate_cohort(connection,context)
                for table,name in zip((cohort,candidate_audit),TABLE_NAMES,strict=True):
                    pq.write_table(table,work/name,compression="zstd")
                summary = write_reports(connection,run,Path("reports"),context,cohort,candidate_audit)
        run.checks.write_report(work/"csv_tail_checks.csv")
        frozen = {"data_run":data_context.key,"cohort_version":data_context.key,"horizon_hours":72,
                  "common_cohort_frozen":True,"cohort_checks_passed":True,"summary":summary,
                  "image_manifest":{"manifest":refs[1],"generation":context["image_generation"]},
                  "image_cache":data_context.layout.ref("processed",f"{data_context.key}/steps/step07/prepare"),
                  "source_csv_tail_checks":context["source05"]["csv_tail_checks"],
                  "review_csv_tail_checks":context["source07"]["csv_tail_checks"],
                  "csv_tail_checks":run.csv_tail_summary(),"started_utc":started.isoformat(),
                  "finished_utc":datetime.now(timezone.utc).isoformat()}
        write_json(work/"cohort_freeze.json",frozen)
        write_json(work/"run_metadata.json",frozen)
        details.update(common_cohort_frozen=True,cohort_checks_passed=True,counts=summary["counts"],
                       candidate_counts=summary["candidate_counts"],image_generation=context["image_generation"])
    return 0
