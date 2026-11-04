"""Image provenance, review iteration, variable outputs and recovery."""

import csv
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.build_labels import build_burden_labels
from src.cohort.burden_io import clinical_time
from src.cohort.freeze_endpoint import freeze_endpoint
from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.imaging.prepare_images import REVIEW_COLUMNS, prepare_images
from tests.cohort import test_burden_pipeline as pipeline
from tests.cohort.test_image_transform import dicom

PROJECT = Path(__file__).resolve().parents[2]


class ImagePreparationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = pipeline.BurdenPipelineTests()
        self.pipeline.setUp()
        self.addCleanup(self.pipeline.doCleanups)
        self.context, self.store = self.pipeline.context, self.pipeline.store
        selected = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.pipeline.store.manifest_ref("step03", "select")), "study_image_selection.parquet"))).to_pylist()
        indexed = []
        self.dicoms = []
        for number, row in enumerate(selected):
            row["dicom_path"] = f"dataset/p{row['subject_id']}/s{row['study_id']}/{row['selected_dicom_id']}.dcm"
            path = self.pipeline.fixture.cxr / row["dicom_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            values = np.roll(np.arange(620, dtype=np.uint16).reshape(20, 31), number * 3, axis=1)
            ds = dicom(values)
            ds.AcquisitionDateTime = clinical_time(row["acquisition_datetime"]).strftime("%Y%m%d%H%M%S")
            ds.ViewPosition = row["view_position"]
            ds.save_as(path, enforce_file_format=True)
            self.dicoms.append(path)
            indexed.append({"subject_id": row["subject_id"], "study_id": row["study_id"],
                            "dicom_id": row["selected_dicom_id"], "dicom_path": row["dicom_path"],
                            "view_position": row["view_position"], "acquisition_datetime": row["acquisition_datetime"],
                            "rows": 20, "columns": 31})
        index_ref = self.store.manifest_ref("step02", "build")
        select_ref = self.store.manifest_ref("step03", "select")
        with self.store.publish("step02", "build", data_files=("cxr_index.parquet",)) as work:
            pq.write_table(pa.Table.from_pylist(indexed), work / "cxr_index.parquet")
        with self.store.publish("step03", "select", inputs=[index_ref], data_files=("study_image_selection.parquet",), details={"counts": {"selected_studies": len(selected)}}) as work:
            pq.write_table(pa.Table.from_pylist(selected), work / "study_image_selection.parquet")
        with self.store.publish("step04", "match", inputs=[select_ref], configs=[self.context.config_ref("cohort/admission_matching.json")], data_files=("cxr_admission_matches.parquet",), details={"counts": {"selected_studies": len(selected)}}) as work:
            pq.write_table(pa.Table.from_pylist(self.pipeline.fixture.matches), work / "cxr_admission_matches.parquet")
        self.pipeline.audit()
        self.pipeline.review()
        freeze_endpoint(self.context)
        build_burden_labels(self.context)
        settings = read_json(PROJECT / "configs/data.template/v1_r1/imaging/processing.json")
        self.settings_path = self.context.layout.path("project", self.context.directory + "/imaging/processing.json")
        write_json(self.settings_path, settings)
        self.reviews_path = self.settings_path.parent / "reviews/image_qc.csv"
        self.write_reviews([])
        self.ref = self.store.manifest_ref("step07", "prepare")

    def write_reviews(self, rows):
        self.reviews_path.parent.mkdir(parents=True, exist_ok=True)
        with self.reviews_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def output(self, name):
        return self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), name))

    def review_all(self, decision="pass"):
        with self.output("image_qc_review.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.write_reviews([{key: row[key] if key in {"sample_id", "dicom_id"} else
                             {"decision": decision, "rationale": "synthetic review", "reviewer": "synthetic", "reviewed_on": "2026-11-05"}[key]
                             for key in REVIEW_COLUMNS} for row in rows])

    def test_first_empty_review_then_reviewed_rerun_and_directory_split(self):
        prepare_images(self.context)
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(manifest["details"]["unreviewed_studies"], 3)
        self.assertFalse(manifest["details"]["ready_for_cohort_freeze"])
        self.assertEqual(sum(r["root"] == "processed" for r in manifest["outputs"]), 5)
        self.assertTrue(self.output("previews/after/p1_h10_s100.png").is_file())
        self.assertEqual(self.store.output_ref(manifest, "previews/after/p1_h10_s100.png")["root"], "outputs")
        gallery = self.output("image_qc_review.html").read_text(encoding="utf-8")
        self.assertNotIn("burden_score", gallery)
        before = self.output("images/p1/p1/s100.npy").read_bytes()
        self.review_all()
        prepare_images(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)
        self.assertTrue(self.store.validate(self.ref)["details"]["ready_for_cohort_freeze"])
        self.assertEqual(self.output("images/p1/p1/s100.npy").read_bytes(), before)
        status = pq.read_table(self.output("image_qc_status.parquet"))
        self.assertEqual(set(status["qc_status"].to_pylist()), {"pass"})
        self.assertFalse(any("sha256" in name for name in status.schema.names))

    def test_missing_source_can_be_acknowledged_but_not_manually_passed(self):
        self.dicoms[0].unlink()
        prepare_images(self.context)
        manifest = self.store.validate(self.ref)
        self.assertEqual(sum(r["name"].endswith(".npy") for r in manifest["outputs"]), 2)
        self.review_all()
        with self.assertRaises(ValueError):
            prepare_images(self.context)
        self.write_reviews([])
        prepare_images(self.context)
        self.review_all("exclude")
        prepare_images(self.context)
        self.assertEqual(self.store.validate(self.ref)["details"]["unreviewed_studies"], 0)

    def test_wrong_identity_cross_version_and_invalid_review_rejected(self):
        prepare_images(self.context)
        self.review_all()
        with self.reviews_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["dicom_id"] = "different"
        self.write_reviews(rows)
        with self.assertRaises(ValueError):
            prepare_images(self.context)
        with self.assertRaises(PipelineError):
            prepare_images(self.context, reviews_path="../v1_r2/reviews.csv")

    def test_failed_rerun_preserves_arrays_and_retry_increments_generation(self):
        prepare_images(self.context)
        path = self.output("images/p1/p1/s100.npy")
        before = path.read_bytes()
        with patch("src.imaging.prepare_images._gallery", side_effect=RuntimeError("synthetic report failure")), self.assertRaises(RuntimeError):
            prepare_images(self.context)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(PipelineError):
            self.store.validate(self.ref)
        prepare_images(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_upstream_label_rebuild_invalidates_images(self):
        prepare_images(self.context)
        build_burden_labels(self.context)
        with self.assertRaises(PipelineError):
            self.store.validate(self.ref)

    def test_duplicate_pixels_are_detected_without_persisting_fingerprints(self):
        import pydicom
        first = pydicom.dcmread(self.dicoms[0])
        second = pydicom.dcmread(self.dicoms[1])
        second.PixelData = first.PixelData
        second.save_as(self.dicoms[1], enforce_file_format=True)
        prepare_images(self.context)
        metadata = read_json(self.output("run_metadata.json"))
        self.assertEqual(metadata["duplicate_studies"], 2)
        text = self.output("duplicate_images.csv").read_text()
        self.assertIn("decoded_duplicate", text)
        self.assertNotIn("sha256", text)
