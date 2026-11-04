"""Common-cohort membership, clinical evidence, reviewed QC and transactional recovery."""

import csv
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.build_labels import build_burden_labels
from src.cohort.freeze_cohort import freeze_common_cohort
from src.cohort.freeze_endpoint import freeze_endpoint
from src.common.paths import PipelineError, read_json
from src.imaging.prepare_images import prepare_images
from tests.cohort import test_image_preparation as imaging


class FreezeCohortTests(unittest.TestCase):
    def setUp(self):
        self.fixture = imaging.ImagePreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.context, self.store = self.fixture.context, self.fixture.store
        self.ref = self.store.manifest_ref("step08", "freeze")

    def ready(self, exclude_first=False):
        prepare_images(self.context)
        self.fixture.review_all()
        if exclude_first:
            with self.fixture.reviews_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["decision"] = "exclude"
            self.fixture.write_reviews(rows)
        prepare_images(self.context)

    def output(self, name):
        manifest = self.store.validate(self.ref)
        return self.context.layout.resolve(self.store.output_ref(manifest, name))

    def test_intersection_preserves_excluded_labels_and_generation(self):
        self.ready(exclude_first=True)
        freeze_common_cohort(self.context)
        cohort = pq.read_table(self.output("cohort.parquet"))
        audit = pq.read_table(self.output("cohort_audit.parquet"))
        self.assertEqual((cohort.num_rows, audit.num_rows), (2, 3))
        excluded = [r for r in audit.to_pylist() if r["cohort_exclusion_stage"] == "image_qc"]
        self.assertEqual(len(excluded), 1)
        self.assertIsNotNone(excluded[0]["burden_score"])
        self.assertEqual(excluded[0]["qc_performed"], 1)
        self.assertEqual(cohort.schema.field("burden_score").type, pa.float32())
        self.assertEqual(set(cohort["cohort_version"].to_pylist()), {"v1_r1"})
        record = read_json(self.output("cohort_freeze.json"))
        self.assertEqual(record["summary"]["counts"]["studies"], 2)
        self.assertEqual(record["csv_tail_checks"]["status"], "no_csv_inputs")
        self.assertTrue(record["common_cohort_frozen"])
        self.assertFalse(any(r["name"].endswith(".npy") for r in self.store.validate(self.ref)["outputs"]))
        freeze_common_cohort(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_upstream_exclusions_keep_null_labels_and_null_qc(self):
        pipeline = self.fixture.pipeline
        path = pipeline.fixture.iv / pipeline.fixture.ehr["burden_v2"]["tables"]["patients"]
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["subject_id"] == "2":
                row["anchor_age"] = "17"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        pipeline.audit()
        pipeline.review()
        freeze_endpoint(self.context)
        build_burden_labels(self.context)
        self.ready(exclude_first=True)
        freeze_common_cohort(self.context)
        rows = pq.read_table(self.output("cohort_audit.parquet")).to_pylist()
        upstream = [r for r in rows if r["cohort_exclusion_stage"] == "upstream_candidate"]
        self.assertEqual(len(upstream), 1)
        self.assertIsNone(upstream[0]["burden_score"])
        self.assertIsNone(upstream[0]["qc_status"])
        self.assertEqual(upstream[0]["qc_performed"], 0)
        self.assertEqual(pq.read_table(self.output("cohort.parquet")).num_rows, 1)

    def test_pending_review_cannot_freeze(self):
        prepare_images(self.context)
        with self.assertRaises(PipelineError):
            freeze_common_cohort(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_invalid_npy_values_are_rejected(self):
        self.ready()
        path = self.fixture.output("images/p1/p1/s100.npy")
        pixels = np.load(path, allow_pickle=False)
        pixels[0, 0] = np.nan
        np.save(path, pixels, allow_pickle=False)
        with self.assertRaises(ValueError):
            freeze_common_cohort(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_changed_formal_score_is_rejected_by_reconstruction(self):
        self.ready()
        manifest = self.store.validate(self.store.manifest_ref("step06", "build"))
        path = self.context.layout.resolve(self.store.output_ref(manifest, "study_burden_labels.parquet"))
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index("burden_score"), "burden_score", pa.array([7, 3, 0], type=pa.float32()))
        pq.write_table(table, path)
        with self.assertRaises((PipelineError, ValueError)):
            freeze_common_cohort(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_failed_rerun_preserves_cohort_and_retry_increments_generation(self):
        self.ready()
        freeze_common_cohort(self.context)
        path = self.output("cohort.parquet")
        before = path.read_bytes()
        with patch("src.cohort.freeze_cohort.write_reports", side_effect=RuntimeError("synthetic report failure")), self.assertRaises(RuntimeError):
            freeze_common_cohort(self.context)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(PipelineError):
            self.store.validate(self.ref)
        freeze_common_cohort(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_image_rebuild_invalidates_existing_cohort(self):
        self.ready()
        freeze_common_cohort(self.context)
        prepare_images(self.context)
        with self.assertRaises(PipelineError):
            self.store.validate(self.ref)
