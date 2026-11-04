"""Formal publication, frozen dependencies and rollback on generated clinical data."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.build_labels import DATA_OUTPUTS, build_burden_labels
from src.cohort.freeze_endpoint import freeze_endpoint
from src.common.manifest import write_json
from src.common.paths import PipelineError
from tests.cohort import test_burden_pipeline as pipeline


class BuildLabelTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = pipeline.BurdenPipelineTests()
        self.pipeline.setUp()
        self.addCleanup(self.pipeline.doCleanups)
        self.context, self.store = self.pipeline.context, self.pipeline.store
        self.ref = self.store.manifest_ref("step06", "build")

    def freeze(self):
        self.pipeline.audit()
        self.pipeline.review()
        freeze_endpoint(self.context)

    def output(self, name):
        return self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), name))

    def error(self, code, action):
        with self.assertRaises(PipelineError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def decision_path(self):
        frozen = self.store.validate(self.pipeline.freeze_ref)
        return self.context.layout.resolve(self.store.output_ref(frozen, "endpoint_decision.json"))

    def test_publication_labels_evidence_provenance_and_directory_split(self):
        self.freeze()
        before = self.decision_path().read_bytes()
        # Formal publication consumes only the frozen snapshot and Parquet artifacts.
        self.pipeline.fixture.iv.rename(self.pipeline.fixture.iv.with_name("raw_offline"))
        self.assertEqual(build_burden_labels(self.context), 0)
        manifest = self.store.validate(self.ref)
        self.assertEqual({r["name"] for r in manifest["outputs"] if r["root"] == "processed"}, set(DATA_OUTPUTS))
        self.assertEqual(manifest["inputs"], [{"manifest": self.pipeline.freeze_ref, "generation": 1}])
        labels = pq.read_table(self.output("study_burden_labels.parquet")).to_pylist()
        self.assertEqual(len(labels), 3)
        self.assertEqual({r["sample_id"]: r["burden_score"] for r in labels}, {"p1_h10_s100": 8, "p1_h10_s101": 3, "p2_h20_s200": 0})
        for row in labels:
            self.assertEqual((row["data_version"], row["data_run"], row["source_audit_generation"], row["source_freeze_generation"]), (1, 1, 1, 1))
            self.assertEqual(row["horizon_hours"], 72)
            self.assertEqual(row["label_status"], "complete")
        evidence = pq.read_table(self.output("event_evidence.parquet")).to_pylist()
        self.assertTrue(evidence)
        self.assertTrue({r["sample_id"] for r in evidence} <= {r["sample_id"] for r in labels})
        metadata = json.loads(self.output("run_metadata.json").read_text())
        self.assertEqual(metadata["csv_tail_checks"]["status"], "no_csv_inputs")
        self.assertEqual(metadata["counts"]["eligible_studies"], 3)
        self.assertFalse(metadata["image_qc_performed"])
        self.assertEqual(self.decision_path().read_bytes(), before)
        self.assertFalse(any(self.output("report.md").parent.rglob(".labels_*")))

    def test_missing_freeze_is_rejected(self):
        self.pipeline.audit()
        with self.assertRaises(PipelineError):
            build_burden_labels(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_rebuilt_audit_invalidates_freeze_and_existing_labels(self):
        self.freeze()
        build_burden_labels(self.context)
        self.pipeline.audit()
        self.error("UPSTREAM_REBUILT", lambda: build_burden_labels(self.context))
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(self.ref))

    def test_decision_identity_artifacts_and_counts_are_checked(self):
        self.freeze()
        path = self.decision_path()
        decision = json.loads(path.read_text())
        for field, value in [("horizon_hours", 48), ("version", 2), ("artifacts", {})]:
            write_json(path, {**decision, field: value})
            with self.assertRaises(PipelineError):
                build_burden_labels(self.context)
        write_json(path, {**decision, "counts": {**decision["counts"], "eligible_studies": 2}})
        self.error("LABEL_COUNT_MISMATCH", lambda: build_burden_labels(self.context))
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_score_tampering_cannot_publish(self):
        self.freeze()
        prior = self.store.validate(self.pipeline.ref)
        path = self.context.layout.resolve(self.store.output_ref(prior, "window_72h/study_event_audit.parquet"))
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index("burden_score"), "burden_score", pa.array([7, 3, 0], type=pa.float32()))
        pq.write_table(table, path)
        with self.assertRaises((PipelineError, ValueError)):
            build_burden_labels(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_failed_rerun_preserves_payload_and_success_increments_generation(self):
        self.freeze()
        build_burden_labels(self.context)
        path = self.output("study_burden_labels.parquet")
        before = path.read_bytes()
        with patch("src.cohort.build_labels._publish_tables", side_effect=RuntimeError("synthetic failure")), self.assertRaises(RuntimeError):
            build_burden_labels(self.context)
        self.assertEqual(path.read_bytes(), before)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        build_burden_labels(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_cli_requires_version_and_run_from_any_working_directory(self):
        script = Path(__file__).resolve().parents[2] / "scripts/data/06_build_burden_labels.py"
        result = subprocess.run([sys.executable, "-B", str(script)], cwd=self.pipeline.fixture.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--version", result.stderr)
        self.assertIn("--run", result.stderr)
