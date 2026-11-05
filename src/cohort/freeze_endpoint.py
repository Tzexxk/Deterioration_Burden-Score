"""Freeze a reviewed 72-hour audit with generation and policy identity."""

import tempfile
from datetime import datetime
from pathlib import Path

from src.cohort.audit_burden_labels import DATA_ARTIFACTS, select_policy
from src.cohort.burden_io import open_store
from src.cohort.burden_validation import validate_labels
from src.common.manifest import ManifestStore, write_json
from src.common.paths import require


def freeze_endpoint(
    context,
    cohort_path="cohort/burden_rules.json",
    review_path="cohort/reviews/endpoint_review.json",
    review_policy=None,
):
    cohort = context.read(cohort_path, required=("review_policy", "burden_v2"))
    policy = select_policy(cohort, review_policy)
    review = context.read(
        review_path,
        required=(
            "confirmed",
            "version",
            "run",
            "audit_generation",
            "review_policy",
            "horizon_hours",
            "decision_version",
            "reviewer",
            "reviewed_on",
            "rationale",
            "rules_version",
            "review_records",
        ),
    )
    require(
        review["confirmed"] is True,
        "REVIEW_NOT_CONFIRMED",
        "Confirm the endpoint review after inspecting the current audit",
    )
    require(
        type(review["version"]) is int
        and type(review["run"]) is int
        and (review["version"], review["run"]) == (context.version, context.run),
        "REVIEW_VERSION_MISMATCH",
        "Endpoint review must belong to the current data run",
    )
    require(
        review["review_policy"] == policy,
        "REVIEW_POLICY_MISMATCH",
        "Endpoint review uses a different review policy",
    )
    require(
        type(review["horizon_hours"]) is int and review["horizon_hours"] == 72,
        "UNSUPPORTED_OUTCOME_WINDOW",
        "Only 72-hour endpoints can be frozen",
    )
    for field in (
        "decision_version",
        "reviewer",
        "reviewed_on",
        "rationale",
        "rules_version",
    ):
        require(
            isinstance(review[field], str) and bool(review[field].strip()),
            "INCOMPLETE_REVIEW",
            "Endpoint review provenance is incomplete",
        )
    try:
        datetime.strptime(review["reviewed_on"], "%Y-%m-%d")
    except ValueError:
        require(False, "INCOMPLETE_REVIEW", "reviewed_on must use YYYY-MM-DD")
    store = ManifestStore(context.layout, data=context)
    audit_ref = store.manifest_ref("step05", "audit")
    audit = store.validate(audit_ref)
    require(
        audit["details"].get("review_policy") == policy,
        "REVIEW_POLICY_MISMATCH",
        "Audit and freeze must use the same policy",
    )
    require(
        type(review["audit_generation"]) is int
        and review["audit_generation"] == audit["generation"],
        "REVIEW_GENERATION_MISMATCH",
        "Endpoint review does not bind the current audit generation",
    )
    require(
        audit["configs"][0] == context.config_ref(cohort_path),
        "REVIEW_CONFIG_MISMATCH",
        "Freeze must use the audit's rules configuration",
    )
    metadata = context.layout.resolve(store.output_ref(audit, "run_metadata.json"))
    from src.common.paths import read_json

    metadata = read_json(metadata)
    require(
        metadata.get("data_run") == context.key
        and metadata.get("audit_generation") == audit["generation"]
        and metadata.get("review_policy") == policy
        and metadata.get("audit_horizons_hours") == [72],
        "INVALID_MANIFEST",
        "Audit metadata identity does not match the selected publication",
    )
    require(
        metadata["audit_checks_passed"] is True
        and metadata["audit_ready_for_window_decision"] is True
        and metadata["dictionary_handling_review_complete"] is True,
        "AUDIT_NOT_READY",
        "Resolve evidence, candidate and dictionary handling issues before freezing",
    )
    require(
        review["rules_version"] == metadata["rules_version"],
        "REVIEW_CONFIG_MISMATCH",
        "Review and audit rules versions differ",
    )
    require(
        isinstance(review["review_records"], list),
        "INCOMPLETE_REVIEW",
        "review_records must list the audit decision inputs",
    )
    records = [context.layout.ref("project", name) for name in review["review_records"]]
    for ref in records:
        context.check_config_ref(ref)
    require(
        len(records) == len({(r["root"], r["path"]) for r in records})
        and {(r["root"], r["path"]) for r in records}
        == {(r["root"], r["path"]) for r in audit["configs"][3:]},
        "REVIEW_CONFIG_MISMATCH",
        "Endpoint review must cite all and only the audit's record-decision inputs",
    )
    artifacts = {
        key: context.layout.resolve(store.output_ref(audit, name))
        for key, name in zip(
            ("candidates", "candidate_evidence", "observations", "wide", "evidence"),
            DATA_ARTIFACTS,
            strict=True,
        )
    }
    snapshot_ref = store.output_ref(audit, "effective_configuration.json")
    snapshot = read_json(context.layout.resolve(snapshot_ref))
    require(
        snapshot.get("data_run") == context.key
        and snapshot.get("audit_generation") == audit["generation"]
        and snapshot.get("review_policy") == policy,
        "INVALID_MANIFEST",
        "Audit configuration snapshot identity differs",
    )
    ehr = snapshot["ehr"]
    details = {"review_policy": policy, "horizon_hours": 72}
    with store.publish(
        "step05",
        "freeze",
        inputs=[audit_ref],
        configs=[context.config_ref(review_path)],
        details=details,
        required_bytes=sum(p.stat().st_size for p in artifacts.values())
        + 8 * 1024 * 1024,
    ) as work:
        with tempfile.TemporaryDirectory(dir=work, prefix=".validate_") as scratch:
            with open_store(
                Path(scratch),
                ehr["burden_v2"]["memory_limit"],
                max_temp_directory_size=ehr["burden_v2"]["max_temp_directory_size"],
            ) as connection:
                counts = validate_labels(connection, artifacts, 72, review_policy=policy)
        require(
            store.validate(audit_ref)["generation"] == review["audit_generation"],
            "REVIEW_GENERATION_MISMATCH",
            "Audit changed during freeze",
        )
        decision = {
            "decision_status": "frozen",
            "horizon_frozen": True,
            "label_rules_frozen": True,
            "version": context.version,
            "run": context.run,
            "review_policy": policy,
            "horizon_hours": 72,
            "decision_version": review["decision_version"],
            "reviewer": review["reviewer"],
            "reviewed_on": review["reviewed_on"],
            "rationale": review["rationale"],
            "rules_version": metadata["rules_version"],
            "dictionary_version": metadata["dictionary_version"],
            "audit": {"manifest": audit_ref, "generation": audit["generation"]},
            "review": context.config_ref(review_path),
            "review_records": records,
            "artifacts": {
                key: store.output_ref(audit, name)
                for key, name in zip(artifacts, DATA_ARTIFACTS, strict=True)
            },
            "counts": counts,
            "checks_passed": True,
            "outcome_version": "burden_v2",
            "sample_unit": "study",
            "event_window": metadata["event_window"],
            "target_formula": metadata["target_formula"],
            "effective_configuration": snapshot_ref,
        }
        write_json(work / "endpoint_decision.json", decision)
        details.update(
            checks_passed=True, counts=counts, audit_generation=audit["generation"]
        )
    return 0
