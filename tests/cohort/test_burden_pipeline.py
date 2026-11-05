"""Synthetic single-window audit, review binding, freeze and recovery checks."""

import csv
import json
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.audit_burden_labels import DATA_ARTIFACTS, audit_burden_labels
from src.cohort.freeze_endpoint import freeze_endpoint
from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError
from tests.cohort import test_lenient_clinical as clinical

PROJECT = Path(__file__).resolve().parents[2]


def prepare(fixture):
    root = fixture.root
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
    write_json(root / "configs/environment/paths.local.json", {"roots": {key: Path(value).relative_to(root).as_posix() for key, value in fixture.paths.items()}})
    write_json(root / "configs/version_run.json", {"version": 1, "run": 1})
    for name, value in [("cohort/burden_rules.json", fixture.cohort), ("ehr/burden_sources.json", fixture.ehr),
                        ("imaging/raw_audit.json", fixture.imaging),
                        ("cohort/admission_matching.json", {"sample_unit": "study", "admission_matching": fixture.cohort["admission_matching"]})]:
        write_json(root / "configs/data/v1_r1" / name, value)
    for name, fields in clinical.TABLE_COLUMNS.items():
        path = fixture.iv / fixture.ehr["burden_v2"]["tables"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields.split(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(fixture.tables[name])
    context = DataContext.load(Layout(root), 1, 1)
    store = ManifestStore(context.layout, data=context)
    selections = []
    for row in fixture.matches:
        path = fixture.cxr / f"dataset/{row['dicom_id']}.dcm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic upstream image")
        selections.append({"subject_id": row["subject_id"], "study_id": row["study_id"],
                           "selected_dicom_id": row["dicom_id"], "acquisition_datetime": row["t0"],
                           "dicom_path": path.relative_to(fixture.cxr).as_posix(),
                           "view_position": "AP", "selection_reason": "single_frontal"})
    with store.publish("step03", "select", data_files=("study_image_selection.parquet",), details={"counts": {"selected_studies": len(selections)}}) as work:
        pq.write_table(pa.Table.from_pylist(selections), work / "study_image_selection.parquet")
    with store.publish("step04", "match", inputs=[store.manifest_ref("step03", "select")],
                       configs=[context.config_ref("cohort/admission_matching.json")],
                       data_files=("cxr_admission_matches.parquet",), details={"counts": {"selected_studies": len(fixture.matches)}}) as work:
        pq.write_table(pa.Table.from_pylist(fixture.matches), work / "cxr_admission_matches.parquet")
    return context, store


class BurdenPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = clinical.LenientClinicalTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.capture = redirect_stdout(StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)
        self.context, self.store = prepare(self.fixture)
        self.ref = self.store.manifest_ref("step05", "audit")
        self.freeze_ref = self.store.manifest_ref("step05", "freeze")
        self.output = self.context.layout.resolve(self.ref).parent
        self.rules_path = self.fixture.root / "configs/data/v1_r1/cohort/burden_rules.json"

    def audit(self, **kwargs):
        return audit_burden_labels(self.context, review_policy="lenient", **kwargs)

    def metadata(self):
        return json.loads((self.output / "run_metadata.json").read_text(encoding="utf-8"))

    def review(self, **updates):
        value = {"confirmed": True, "version": 1, "run": 1,
                 "audit_generation": self.store.validate(self.ref)["generation"], "review_policy": "lenient",
                 "horizon_hours": 72, "decision_version": "fixture", "reviewer": "synthetic",
                 "reviewed_on": "2026-11-05", "rationale": "synthetic review", "rules_version": self.fixture.cohort["burden_v2"]["rules_version"], "review_records": []}
        value.update(updates)
        write_json(self.fixture.root / "configs/data/v1_r1/cohort/reviews/endpoint_review.json", value)

    def error(self, code, action):
        with self.assertRaises(PipelineError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_audit_single_window_inventory_policy_and_provenance(self):
        self.assertEqual(self.audit(), 0)
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(manifest["details"]["review_policy"], "lenient")
        self.assertEqual(manifest["inputs"], [{"manifest": self.store.manifest_ref(step, phase), "generation": 1} for step, phase in [("step03", "select"), ("step04", "match")]])
        self.assertEqual({entry["name"] for entry in manifest["outputs"] if entry["root"] == "processed"}, set(DATA_ARTIFACTS))
        self.assertEqual(self.metadata()["audit_horizons_hours"], [72])
        self.assertTrue(self.metadata()["audit_ready_for_window_decision"])
        self.assertFalse(any("48" in item["name"] for item in manifest["outputs"]))
        self.assertFalse(any("sha256" in key for key in self.metadata()))
        self.assertFalse(any(self.output.rglob(".clinical_*")))

    def test_lenient_keeps_published_flag_and_score_behavior(self):
        self.audit()
        manifest = self.store.validate(self.ref)
        wide = pq.read_table(self.context.layout.resolve(self.store.output_ref(manifest, "window_72h/study_event_audit.parquet"))).to_pylist()
        rows = {row["sample_id"]: row for row in wide}
        self.assertEqual(rows["p1_h10_s100"]["burden_score"], 8)
        self.assertEqual(rows["p1_h10_s101"]["burden_score"], 3)
        self.assertEqual(rows["p2_h20_s200"]["candidate_status"], "eligible")
        self.assertEqual(rows["p2_h20_s200"]["burden_score"], 0)
        self.assertEqual(rows["p2_h20_s200"]["early_discharge"], 1)

    def test_policy_mismatch_and_missing_strict_sources_are_rejected(self):
        self.error("REVIEW_POLICY_MISMATCH", lambda: audit_burden_labels(self.context, review_policy="strict"))
        value = json.loads(self.rules_path.read_text(encoding="utf-8"))
        value["review_policy"] = "strict"
        write_json(self.rules_path, value)
        self.error("MISSING_CONFIG_FIELD", lambda: audit_burden_labels(self.context, review_policy="strict"))
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_missing_policy_and_cross_run_review_are_rejected(self):
        value = json.loads(self.rules_path.read_text(encoding="utf-8"))
        value.pop("review_policy")
        write_json(self.rules_path, value)
        self.error("MISSING_CONFIG_FIELD", self.audit)
        value["review_policy"] = "lenient"
        value["burden_v2"]["overrides_path"] = "configs/data/v1_r2/cohort/reviews/events.csv"
        write_json(self.rules_path, value)
        self.error("CROSS_VERSION_CONFIG", self.audit)

    def test_upstream_rebuild_invalidates_audit(self):
        self.audit()
        with self.store.publish("step03", "select", data_files=("study_image_selection.parquet",), details={"counts": {"selected_studies": 0}}) as work:
            pq.write_table(pa.table({"subject_id": pa.array([], type=pa.int64())}), work / "study_image_selection.parquet")
        self.error("UPSTREAM_REBUILT", self.audit)
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(self.ref))

    def test_truncated_tail_is_reported_and_middle_damage_blocks_publication(self):
        path = self.fixture.iv / self.fixture.ehr["burden_v2"]["tables"]["chartevents"]
        with path.open("ab") as stream:
            stream.write(b"bad")
        self.audit()
        self.assertEqual(self.metadata()["csv_tail_checks"]["skipped_record_count"], 1)
        before = self.context.layout.resolve(self.ref).read_bytes()
        content = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([content[0], "bad", *content[1:-1]]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.audit()
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))

    def test_failed_rerun_preserves_previous_payload_and_generation(self):
        self.audit()
        before = self.context.layout.resolve(self.ref).read_bytes()
        with patch("src.cohort.audit_burden_labels.reports", side_effect=RuntimeError("synthetic report failure")), self.assertRaises(RuntimeError):
            self.audit()
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        self.audit()
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_freeze_checks_evidence_and_binds_audit_generation(self):
        self.audit()
        self.review()
        self.assertEqual(freeze_endpoint(self.context, review_policy="lenient"), 0)
        frozen = self.store.validate(self.freeze_ref)
        self.assertEqual(frozen["details"]["counts"]["eligible_studies"], 3)
        self.assertEqual(frozen["inputs"], [{"manifest": self.ref, "generation": 1}])
        self.assertEqual(frozen["details"]["review_policy"], "lenient")
        self.audit()
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(self.freeze_ref))
        self.error("REVIEW_GENERATION_MISMATCH", lambda: freeze_endpoint(self.context))

    def test_unconfirmed_wrong_policy_or_wrong_version_review_cannot_freeze(self):
        self.audit()
        for updates, code in [({"confirmed": False}, "REVIEW_NOT_CONFIRMED"), ({"review_policy": "strict"}, "REVIEW_POLICY_MISMATCH"),
                              ({"version": 2}, "REVIEW_VERSION_MISMATCH"), ({"audit_generation": 999}, "REVIEW_GENERATION_MISMATCH")]:
            self.review(**updates)
            self.error(code, lambda: freeze_endpoint(self.context))
        self.assertFalse(self.context.layout.resolve(self.freeze_ref).exists())

    def test_unreviewed_dictionary_blocks_freeze(self):
        value = json.loads(self.rules_path.read_text(encoding="utf-8"))
        value["burden_v2"]["careunits"]["Ward"]["confirmed"] = False
        write_json(self.rules_path, value)
        self.audit()
        self.review()
        self.error("AUDIT_NOT_READY", lambda: freeze_endpoint(self.context))

    def test_score_tampering_is_detected_without_hash_checks(self):
        self.audit()
        self.review()
        manifest = self.store.validate(self.ref)
        path = self.context.layout.resolve(self.store.output_ref(manifest, "window_72h/study_event_audit.parquet"))
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["burden_score"] = 0
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        entry = next(r for r in manifest["outputs"] if r["name"] == "window_72h/study_event_audit.parquet")
        entry["bytes"] = path.stat().st_size
        write_json(self.context.layout.resolve(self.ref), manifest)
        with self.assertRaisesRegex(ValueError, "scores disagree"):
            freeze_endpoint(self.context)

    def test_record_review_input_is_applied_and_must_be_cited_by_freeze(self):
        self.audit()
        manifest = self.store.validate(self.ref)
        evidence = pq.read_table(self.context.layout.resolve(self.store.output_ref(manifest, "window_72h/event_evidence.parquet"))).to_pylist()
        row = next(r for r in evidence if r["sample_id"] == "p1_h10_s100" and r["component"] == clinical.COMPONENTS[3] and r["qualifies"] == 1)
        name = "configs/data/v1_r1/cohort/reviews/event_evidence.csv"
        path = self.fixture.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = (PROJECT / "configs/data.template/v1_r1/cohort/reviews/event_evidence.csv").read_text(encoding="utf-8").strip().split(",")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerow({"sample_id": row["sample_id"], "horizon_hours": 72, "component": row["component"],
                             "event_flag": 0, "first_event_time": "", "source_record_keys": row["source_record_keys"],
                             "reviewer": "synthetic", "rationale": "fixture decision", "version": "fixture", "override_scope": "component"})
        settings = json.loads(self.rules_path.read_text(encoding="utf-8"))
        settings["burden_v2"]["overrides_path"] = name
        write_json(self.rules_path, settings)
        self.audit()
        self.assertEqual(self.metadata()["override_count"], 1)
        self.review()
        self.error("REVIEW_CONFIG_MISMATCH", lambda: freeze_endpoint(self.context))
        self.review(review_records=[name])
        self.assertEqual(freeze_endpoint(self.context), 0)

    def test_admission_source_must_match_step04_configuration(self):
        settings = json.loads(self.rules_path.read_text(encoding="utf-8"))
        settings["admission_matching"]["admissions_path"] = "hosp/another.csv"
        write_json(self.rules_path, settings)
        self.error("ADMISSION_SOURCE_MISMATCH", self.audit)

    def test_unknown_record_review_sample_is_rejected(self):
        path = self.fixture.root / "configs/data/v1_r1/cohort/reviews/candidate_exclusions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('sample_id,reason,source_record_keys,reviewer,rationale,version\nunknown,fixture,[],synthetic,fixture,fixture\n', encoding="utf-8")
        settings = json.loads(self.rules_path.read_text(encoding="utf-8"))
        settings["burden_v2"]["candidate_exclusions_path"] = path.relative_to(self.fixture.root).as_posix()
        write_json(self.rules_path, settings)
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate candidate exclusion"):
            self.audit()

    def test_cli_works_from_unrelated_directory(self):
        shutil.copytree(PROJECT / "src", self.fixture.root / "src")
        shutil.copytree(PROJECT / "scripts", self.fixture.root / "scripts")
        command = [sys.executable, "-B", str(self.fixture.root / "scripts/data/05_audit_burden_labels.py"), "--version", "1", "--run", "1", "--mode", "audit", "--review-policy", "lenient"]
        result = subprocess.run(command, cwd=self.fixture.root.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["mode"], "audit")
        result = subprocess.run(command[:-1] + ["strict"], cwd=self.fixture.root.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
