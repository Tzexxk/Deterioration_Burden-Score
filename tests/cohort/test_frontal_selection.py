"""Synthetic selection rules and generation-bound study-index checks."""

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.audit_raw_data import run_audit
from src.cohort.build_cxr_index import INDEX_SCHEMA, build_cxr_index
from src.cohort.select_frontal_image import (
    SELECTION_SCHEMA,
    _parse_image_time,
    _pixel_area,
    _select_study,
    select_frontal_images,
)
from src.common.paths import PipelineError
from tests.cohort import test_raw_audit as audit_tests

PROJECT = Path(__file__).resolve().parents[2]


def image(dicom="a", *, time="2020-01-01T10:00:00", area=100, view="AP", readable=True, subject=1, study=10):
    return {"subject_id": subject, "study_id": study, "dicom_id": dicom,
            "dicom_path": f"dataset/{dicom}.dcm", "view_position": view,
            "acquisition_datetime": time, "acquisition_time_source": "AcquisitionDateTime" if time else "missing",
            "rows": area, "columns": 1, "is_readable": readable,
            "decode_error": None if readable else "synthetic_decode_error", "is_frontal": view in {"AP", "PA"},
            "metadata_readable": True, "time_failure_reason": None if time else "missing_time",
            "index_path": f"files/{dicom}.dcm", "path_resolution": "flattened"}


def rank(rows):
    return _select_study([{**row, "_parsed_time": _parse_image_time(row)} for row in rows])


class SelectionRuleTests(unittest.TestCase):
    def test_single_frontal_and_counts(self):
        selected, reason = rank([image(), image("b", view="LATERAL")])
        self.assertEqual(reason, "single_frontal")
        self.assertEqual((selected["n_images"], selected["n_frontal_images"], selected["n_eligible_frontal_images"]), (2, 1, 1))

    def test_single_eligible_frontal(self):
        selected, reason = rank([image("a", readable=False), image("b", view="PA")])
        self.assertEqual((selected["selected_dicom_id"], reason), ("b", "single_eligible_frontal"))

    def test_earliest_time_precedes_pixel_area(self):
        selected, reason = rank([image("a", time="2020-01-01T11:00:00", area=1000), image("b", area=1)])
        self.assertEqual((selected["selected_dicom_id"], reason), ("b", "earliest_frontal"))

    def test_largest_area_breaks_time_tie(self):
        selected, reason = rank([image("a", area=1), image("b", area=200, view="PA")])
        self.assertEqual((selected["selected_dicom_id"], reason), ("b", "largest_pixel_area"))

    def test_dicom_id_tie_break_is_independent_of_input_order(self):
        rows = [image("b"), image("a")]
        for order in (rows, list(reversed(rows))):
            selected, reason = rank(order)
            self.assertEqual((selected["selected_dicom_id"], reason), ("a", "dicom_id_tiebreak"))

    def test_missing_dimensions_have_lowest_area(self):
        self.assertEqual(_pixel_area(image(area=None)), -1)
        selected, reason = rank([image("a", area=None), image("b", area=1)])
        self.assertEqual((selected["selected_dicom_id"], reason), ("b", "largest_pixel_area"))

    def test_no_frontal_exclusion(self):
        self.assertEqual(rank([image(view="LATERAL")]), (None, "no_frontal"))

    def test_no_readable_frontal_exclusion(self):
        self.assertEqual(rank([image(readable=False)]), (None, "no_readable_frontal"))

    def test_no_reliable_time_exclusion(self):
        self.assertEqual(rank([image(time=None)]), (None, "no_reliable_time"))

    def test_timezone_awareness_must_be_consistent_within_study(self):
        with self.assertRaisesRegex(ValueError, "mixes timezone"):
            rank([image("a"), image("b", time="2020-01-01T10:00:00+08:00")])

    def test_aware_times_are_ranked_by_actual_instant(self):
        selected, reason = rank([image("a", time="2020-01-01T11:00:00+02:00"),
                                 image("b", time="2020-01-01T10:00:00+00:00")])
        self.assertEqual((selected["selected_dicom_id"], reason), ("a", "earliest_frontal"))

    def test_time_source_and_value_must_agree(self):
        for row in [{**image(), "acquisition_time_source": "missing"},
                    {**image(), "acquisition_time_source": "unknown"}, image(time="invalid")]:
            with self.assertRaises(ValueError):
                _parse_image_time(row)


class SelectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_tests.RawAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.context = self.fixture.context
        self.root = self.fixture.root
        self.store = self.fixture.store
        self.upstream = self.store.manifest_ref("step02", "build")
        self.ref = self.store.manifest_ref("step03", "select")
        self.output = self.context.layout.resolve(self.ref).parent

    def prepare(self, rows=None, schema=INDEX_SCHEMA):
        run_audit(self.context)
        if rows is None:
            build_cxr_index(self.context)
        else:
            with self.store.publish("step02", "build", data_files=("cxr_index.parquet",), inputs=[self.fixture.ref], details={"counts": {"output_rows": len(rows)}}) as work:
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), work / "cxr_index.parquet")

    def select(self):
        return select_frontal_images(self.context)

    def summary(self):
        return json.loads((self.output / "study_image_selection_summary.json").read_text(encoding="utf-8"))

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_complete_selection_schema_counts_and_upstream_generation(self):
        self.prepare()
        self.assertEqual(self.select(), 0)
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(manifest["inputs"], [{"manifest": self.upstream, "generation": 1}])
        self.assertEqual(len(manifest["outputs"]), 2)
        table = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), "study_image_selection.parquet")))
        self.assertEqual(table.schema, SELECTION_SCHEMA)
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(table.to_pylist()[0]["selected_dicom_id"], "abc1")
        self.assertEqual(self.summary()["counts"], {"input_images": 2, "input_studies": 2, "selected_studies": 1, "excluded_studies": 1})
        self.assertEqual(self.summary()["exclusion_reasons"], {"no_frontal": 1})
        self.assertFalse(any(self.output.parent.rglob("temp")))

    def test_index_is_required_before_selection(self):
        self.error("MISSING_FILE", self.select)
        self.assertFalse(self.output.exists())

    def test_missing_columns_are_rejected(self):
        schema = pa.schema([pa.field("subject_id", pa.int64())])
        self.prepare([{"subject_id": 1}], schema=schema)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.select()
        self.assertFalse(self.output.exists())

    def test_duplicate_image_keys_are_rejected(self):
        self.prepare([image(), image()])
        with self.assertRaisesRegex(ValueError, "Duplicate CXR index image"):
            self.select()

    def test_study_cannot_belong_to_multiple_subjects(self):
        self.prepare([image("a", subject=1), image("b", subject=2)])
        with self.assertRaisesRegex(ValueError, "multiple subjects"):
            self.select()

    def test_inconsistent_view_flag_is_rejected(self):
        self.prepare([{**image(), "is_frontal": False}])
        with self.assertRaisesRegex(ValueError, "Inconsistent frontal"):
            self.select()

    def test_unsafe_image_path_is_rejected(self):
        self.prepare([{**image(), "dicom_path": "../outside.dcm"}])
        self.error("INVALID_PATH", self.select)

    def test_input_row_count_must_match_upstream_manifest(self):
        run_audit(self.context)
        with self.store.publish("step02", "build", data_files=("cxr_index.parquet",), inputs=[self.fixture.ref], details={"counts": {"output_rows": 99}}) as work:
            pq.write_table(pa.Table.from_pylist([image()], schema=INDEX_SCHEMA), work / "cxr_index.parquet")
        self.error("UPSTREAM_COUNT_MISMATCH", self.select)

    def test_all_excluded_studies_publish_an_empty_typed_table(self):
        self.prepare([image(view="LATERAL")])
        self.select()
        table = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), "study_image_selection.parquet")))
        self.assertEqual(table.num_rows, 0)
        self.assertEqual(table.schema, SELECTION_SCHEMA)
        self.assertEqual(self.summary()["counts"]["excluded_studies"], 1)

    def test_transitive_rebuild_blocks_selection_until_index_is_rebuilt(self):
        self.prepare()
        self.select()
        run_audit(self.context)
        self.error("UPSTREAM_REBUILT", self.select)
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(self.ref))
        build_cxr_index(self.context)
        self.select()
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["generation"], 2)
        self.assertEqual(manifest["inputs"][0]["generation"], 2)

    def test_failed_rerun_keeps_old_selection_and_retry_increments_once(self):
        self.prepare()
        self.select()
        before = self.context.layout.resolve(self.ref).read_bytes()
        with patch("src.cohort.select_frontal_image.pq.write_table", side_effect=OSError("synthetic write failure")), self.assertRaises(OSError):
            self.select()
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        self.select()
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_cli_works_from_other_directory_and_rejects_wrong_run(self):
        self.prepare()
        shutil.copytree(PROJECT / "src", self.root / "src")
        scripts = self.root / "scripts/data"
        scripts.mkdir(parents=True)
        script = scripts / "03_select_frontal_image.py"
        shutil.copyfile(PROJECT / "scripts/data/03_select_frontal_image.py", script)
        result = subprocess.run([sys.executable, "-B", str(script), "--version", "1", "--run", "1"],
                                cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(str(self.root), result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["counts"]["selected_studies"], 1)
        before = self.context.layout.resolve(self.ref).read_bytes()
        wrong = subprocess.run([sys.executable, "-B", str(script), "--version", "1", "--run", "2"],
                               cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(wrong.returncode, 2)
        self.assertIn("VERSION_MISMATCH", wrong.stderr)
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
