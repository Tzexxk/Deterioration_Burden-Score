"""Small CXR-index checks using generated DICOM files and CSV fixtures."""

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from src.cohort.audit_raw_data import run_audit
from src.cohort.build_cxr_index import INDEX_SCHEMA, _dimension, build_cxr_index
from src.common.manifest import write_json
from src.common.paths import PipelineError
from tests.cohort import test_raw_audit as audit_tests

PROJECT = Path(__file__).resolve().parents[2]


class CxrIndexTests(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_tests.RawAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.context = self.fixture.context
        self.store = self.fixture.store
        self.upstream = self.fixture.ref
        self.ref = self.store.manifest_ref("step02", "build")
        self.output = self.context.layout.resolve(self.ref).parent

    def prepare(self):
        run_audit(self.context)

    def build(self):
        return build_cxr_index(self.context)

    def summary(self):
        return json.loads((self.output / "cxr_index_summary.json").read_text(encoding="utf-8"))

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_schema_rows_counts_paths_and_generation_binding(self):
        self.prepare()
        self.assertEqual(self.build(), 0)
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(manifest["inputs"], [{"manifest": self.upstream, "generation": 1}])
        self.assertEqual(len(manifest["outputs"]), 3)
        table = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), "cxr_index.parquet")))
        self.assertEqual(table.schema, INDEX_SCHEMA)
        self.assertEqual(table.num_rows, 2)
        rows = table.to_pylist()
        self.assertEqual([row["dicom_id"] for row in rows], ["abc1", "abc2"])
        self.assertEqual(rows[0]["rows"], 2)
        self.assertEqual(rows[0]["dicom_path"], "dataset/p10000000/s100/abc1.dcm")
        self.assertEqual(self.summary()["counts"], {"index_records": 3, "local_dicom_files": 2,
            "matched_local_files": 2, "output_rows": 2, "missing_index_paths": 1,
            "unreferenced_local_files": 0, "readable_images": 2, "unreadable_images": 0,
            "frontal_images": 1, "images_with_acquisition_time": 2})
        self.assertEqual(self.summary()["difference_from_prior_audit"], 0)
        self.assertNotIn("sha256", json.dumps(manifest).lower())
        self.assertFalse(any(self.output.parent.rglob("temp")))

    def test_raw_audit_is_required_before_indexing(self):
        with patch("src.cohort.build_cxr_index._read_local_dicom") as read:
            self.error("MISSING_FILE", self.build)
            read.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_upstream_blocks_indexing(self):
        self.prepare()
        (self.fixture.iv / "hosp/patients.csv").unlink()
        self.error("RAW_AUDIT_INCOMPLETE", lambda: run_audit(self.context))
        self.error("UPSTREAM_INCOMPLETE", self.build)
        self.assertFalse(self.output.exists())

    def test_upstream_rebuild_invalidates_index_and_rebuild_rebinds(self):
        self.prepare()
        self.build()
        self.prepare()
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(self.ref))
        self.build()
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["generation"], 2)
        self.assertEqual(manifest["inputs"][0]["generation"], 2)

    def test_duplicate_record_key_rejected_without_partial_publication(self):
        self.prepare()
        rows = [*self.fixture.rows, self.fixture.rows[0]]
        audit_tests.write_csv(self.fixture.record, list(rows[0]), rows)
        with self.assertRaisesRegex(ValueError, "Duplicate CXR record key"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(any(self.output.parent.rglob("temp")))

    def test_record_study_must_exist_in_study_index(self):
        self.prepare()
        audit_tests.write_csv(self.fixture.cxr / "cxr-study-list.csv", ["subject_id", "study_id", "path"],
                              [{key: self.fixture.rows[0][key] for key in ("subject_id", "study_id", "path")}])
        with self.assertRaisesRegex(ValueError, "study absent from study-list"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_unsafe_index_path_rejected(self):
        self.prepare()
        rows = [{**self.fixture.rows[0], "path": "../outside.dcm"}, *self.fixture.rows[1:]]
        audit_tests.write_csv(self.fixture.record, list(rows[0]), rows)
        with self.assertRaisesRegex(ValueError, "unsafe_index_path"):
            self.build()

    def test_standard_root_layout_is_supported(self):
        target = self.fixture.cxr / self.fixture.rows[0]["path"]
        target.parent.mkdir(parents=True)
        self.fixture.images[0].rename(target)
        self.prepare()
        self.build()
        rows = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), "cxr_index.parquet"))).to_pylist()
        self.assertEqual(rows[0]["path_resolution"], "standard_root")
        self.assertEqual(rows[0]["dicom_path"], self.fixture.rows[0]["path"])
        self.assertEqual(self.summary()["difference_from_prior_audit"], 0)

    def test_unreferenced_local_files_are_counted(self):
        shutil.copyfile(self.fixture.images[0], self.fixture.images[0].with_name("unused.dcm"))
        self.prepare()
        self.build()
        self.assertEqual(self.summary()["counts"]["local_dicom_files"], 3)
        self.assertEqual(self.summary()["counts"]["unreferenced_local_files"], 1)
        self.assertEqual(self.summary()["counts"]["output_rows"], 2)

    def test_unreadable_dicom_is_retained_with_error_status(self):
        self.fixture.images[0].write_bytes(b"invalid dicom")
        self.prepare()
        self.build()
        rows = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), "cxr_index.parquet"))).to_pylist()
        self.assertFalse(rows[0]["is_readable"])
        self.assertEqual(rows[0]["decode_error"], "metadata_read_error:InvalidDicomError")
        self.assertEqual(self.summary()["counts"]["unreadable_images"], 1)

    def test_dimension_values_require_positive_integers(self):
        self.assertEqual(_dimension({"Rows": "2"}, "Rows"), 2)
        for value in (None, "bad", 0, -1):
            self.assertIsNone(_dimension({"Rows": value}, "Rows"))

    def test_incomplete_final_csv_records_are_skipped_and_reported(self):
        for path in (self.fixture.record, self.fixture.cxr / "cxr-study-list.csv"):
            with path.open("ab") as stream:
                stream.write(b"999")
        self.prepare()
        self.build()
        self.assertEqual(self.summary()["counts"]["output_rows"], 2)
        self.assertEqual(self.summary()["csv_tail_checks"]["skipped_record_count"], 2)

    def test_failure_preserves_previous_index_then_retry_increments_once(self):
        self.prepare()
        self.build()
        previous = self.context.layout.resolve(self.ref).read_bytes()
        with patch("src.cohort.build_cxr_index._read_local_dicom", side_effect=RuntimeError("synthetic failure")), self.assertRaises(RuntimeError):
            self.build()
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), previous)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        self.build()
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_disabled_decode_rejected_before_publication(self):
        self.prepare()
        path = self.root / "configs/data/v1_r1/imaging/raw_audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_audit"]["decode_pixels"] = False
        write_json(path, value)
        self.error("INVALID_CONFIG", self.build)

    def test_cli_runs_from_other_directory_and_wrong_run_does_not_modify_output(self):
        self.prepare()
        shutil.copytree(PROJECT / "src", self.root / "src")
        scripts = self.root / "scripts/data"
        scripts.mkdir(parents=True)
        script = scripts / "02_build_cxr_index.py"
        shutil.copyfile(PROJECT / "scripts/data/02_build_cxr_index.py", script)
        result = subprocess.run([sys.executable, "-B", str(script), "--version", "1", "--run", "1"],
                                cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(str(self.root), result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["counts"]["output_rows"], 2)
        before = self.context.layout.resolve(self.ref).read_bytes()
        wrong = subprocess.run([sys.executable, "-B", str(script), "--version", "1", "--run", "2"],
                               cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(wrong.returncode, 2)
        self.assertIn("VERSION_MISMATCH", wrong.stderr)
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
