"""Synthetic admission boundaries, ambiguity, and generation-bound publication."""

import json
import shutil
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.audit_raw_data import run_audit
from src.cohort.build_cxr_index import build_cxr_index
from src.cohort.match_admissions import (
    FAILURE_SCHEMA,
    MATCH_SCHEMA,
    _load_selections,
    _match_one,
    match_admissions,
)
from src.cohort.select_frontal_image import select_frontal_images
from src.common.manifest import write_json
from src.common.paths import PipelineError
from tests.cohort import test_raw_audit as audit_tests

PROJECT = Path(__file__).resolve().parents[2]
T0 = "2020-01-01T12:00:00"


def admission(start="2020-01-01T00:00:00", end="2020-01-02T00:00:00", hadm=1):
    return {"hadm_id": hadm, "admittime": start, "dischtime": end,
            "_start": datetime.fromisoformat(start) if start else None,
            "_end": datetime.fromisoformat(end) if end else None}


def match(rows, time=T0):
    return _match_one({"subject_id": 1, "study_id": 10, "selected_dicom_id": "a",
                       "acquisition_datetime": time, "_t0": datetime.fromisoformat(time)}, rows)


class AdmissionRuleTests(unittest.TestCase):
    def test_half_open_boundaries(self):
        self.assertEqual(match([admission(start=T0)])[0]["match_status"], "unique_match")
        self.assertEqual(match([admission(end=T0)])[0]["failure_reason"], "outside_all_admissions")

    def test_no_patient_retains_empty_candidate_detail(self):
        row, detail = match([])
        self.assertEqual(row["failure_reason"], "no_patient_admissions")
        self.assertEqual(len(detail), 1)
        self.assertIsNone(detail[0]["candidate_hadm_id"])

    def test_overlaps_are_ambiguous(self):
        row, detail = match([admission(), admission(hadm=2)])
        self.assertEqual((row["match_status"], row["failure_reason"]), ("multiple_match", "multiple_confirmed"))
        self.assertIsNone(row["hadm_id"])
        self.assertEqual(len(detail), 2)

    def test_uncertain_competitor_blocks_unique_match(self):
        row, detail = match([admission(), admission(end=None, hadm=2), admission(end=T0, hadm=3)])
        self.assertEqual(row["failure_reason"], "uncertain_competing_admission")
        self.assertEqual((row["n_confirmed_matches"], row["n_uncertain_candidates"]), (1, 1))
        self.assertEqual(len(detail), 3)

    def test_incomplete_intervals_outside_by_known_boundary(self):
        for item in [admission(start=None, end=T0), admission(start="2020-01-02T00:00:00", end=None)]:
            self.assertEqual(match([item])[0]["failure_reason"], "outside_all_admissions")

    def test_invalid_or_missing_boundaries_and_id_are_uncertain(self):
        for item in [admission(start=None), admission(end=None), admission(start=None, end=None),
                     admission(start="2020-01-03T00:00:00"), admission(hadm=None)]:
            self.assertEqual(match([item])[0]["failure_reason"], "uncertain_interval")

    def test_timezone_mismatch_is_uncertain(self):
        self.assertEqual(match([admission()], time=T0 + "+00:00")[1][0]["candidate_reason"], "timezone_mismatch")

    def test_equivalent_aware_instants_match(self):
        row, detail = match([admission(start="2020-01-01T13:00:00+02:00", end="2020-01-01T15:00:00+02:00")], time=T0 + "+00:00")
        self.assertEqual(row["match_status"], "unique_match")
        self.assertEqual(detail, [])


class AdmissionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_tests.RawAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.context, self.root, self.store = self.fixture.context, self.fixture.root, self.fixture.store
        config = json.loads((PROJECT / "configs/data.template/v1_r1/cohort/admission_matching.json").read_text(encoding="utf-8"))
        self.config = self.root / "configs/data/v1_r1/cohort/admission_matching.json"
        write_json(self.config, config)
        self.ref = self.store.manifest_ref("step04", "match")
        self.upstream = self.store.manifest_ref("step03", "select")
        self.output = self.context.layout.resolve(self.ref).parent
        self.csv = self.fixture.iv / "hosp/admissions.csv"

    def admissions(self, rows=None):
        columns = self.context.read("ehr/raw_audit.json")["raw_audit"]["required_tables"]["admissions"]["columns"]
        rows = rows if rows is not None else [{"subject_id": 10000000, "hadm_id": 123, "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-02 00:00:00"}]
        audit_tests.write_csv(self.csv, columns, [{**dict.fromkeys(columns, ""), **row} for row in rows])

    def prepare(self):
        self.admissions()
        run_audit(self.context)
        build_cxr_index(self.context)
        select_frontal_images(self.context)

    def run_match(self):
        return match_admissions(self.context)

    def error(self, code, action):
        with self.assertRaises(PipelineError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_end_to_end_schemas_counts_and_binding(self):
        self.prepare()
        self.assertEqual(self.run_match(), 0)
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["inputs"], [{"manifest": self.upstream, "generation": 1}])
        self.assertEqual(manifest["details"]["counts"]["unique_match"], 1)
        self.assertEqual(len(manifest["outputs"]), 4)
        table = pq.read_table(self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref), "cxr_admission_matches.parquet")))
        self.assertEqual(table.schema, MATCH_SCHEMA)
        self.assertEqual(table.to_pylist()[0]["hadm_id"], 123)
        failure = pq.read_table(self.output / "cxr_admission_match_failures.parquet")
        self.assertEqual(failure.schema, FAILURE_SCHEMA)
        self.assertEqual(failure.num_rows, 0)
        serialized = json.dumps(manifest)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("sha256", serialized)
        self.assertFalse(any(self.output.parent.rglob("temp")))

    def test_missing_upstream_is_rejected(self):
        self.error("MISSING_FILE", self.run_match)

    def test_transitive_rebuild_blocks_match(self):
        self.prepare()
        run_audit(self.context)
        self.error("UPSTREAM_REBUILT", self.run_match)

    def test_no_cross_version_config_fallback(self):
        self.config.unlink()
        self.error("MISSING_FILE", self.run_match)

    def test_missing_admissions_is_rejected(self):
        self.prepare()
        self.csv.unlink()
        self.error("MISSING_INPUT", self.run_match)

    def test_duplicate_hadm_even_in_unselected_subject_is_rejected(self):
        self.prepare()
        self.admissions([{"subject_id": 999, "hadm_id": 7, "admittime": "", "dischtime": ""}] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate admissions hadm_id"):
            self.run_match()

    def test_malformed_csv_header_and_subject_are_rejected(self):
        self.prepare()
        for text in ["subject_id,hadm_id\n1,1\n", "subject_id,hadm_id,admittime,dischtime\ninvalid,1,,\n"]:
            self.csv.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.run_match()

    def test_selection_row_count_must_match_manifest(self):
        self.prepare()
        path = self.context.layout.resolve(self.store.output_ref(self.store.validate(self.upstream), "study_image_selection.parquet"))
        table = pq.read_table(path)
        with self.store.publish("step03", "select", data_files=("study_image_selection.parquet",), details={"counts": {"selected_studies": 99}}) as work:
            pq.write_table(table, work / path.name)
        self.error("UPSTREAM_COUNT_MISMATCH", self.run_match)

    def test_truncated_final_record_is_reported_and_skipped(self):
        self.prepare()
        with self.csv.open("ab") as stream:
            stream.write(b"999")
        self.run_match()
        check = self.store.validate(self.ref)["details"]["csv_tail_checks"]
        self.assertEqual((check["incomplete_file_count"], check["skipped_record_count"]), (1, 1))

    def test_configured_admission_path_cannot_escape_root(self):
        self.config.write_text(json.dumps({"sample_unit": "study", "admission_matching": {"admissions_path": "../outside.csv"}}), encoding="utf-8")
        self.error("ABSOLUTE_PATH_FORBIDDEN", self.run_match)

    def test_duplicate_invalid_and_empty_selection_rejected(self):
        path = self.root / "selection.parquet"
        row = {"subject_id": 1, "study_id": 10, "selected_dicom_id": "a", "acquisition_datetime": T0}
        schema = pa.schema([("subject_id", pa.int64()), ("study_id", pa.int64()), ("selected_dicom_id", pa.string()), ("acquisition_datetime", pa.string())])
        for rows in [[], [row, row], [row, {**row, "subject_id": 2}], [{**row, "acquisition_datetime": "invalid"}]]:
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
            with self.assertRaises(ValueError):
                _load_selections(path)

    def test_failed_overwrite_preserves_complete_previous_output(self):
        self.prepare()
        self.run_match()
        before = self.context.layout.resolve(self.ref).read_bytes()
        with patch("src.cohort.match_admissions.pq.write_table", side_effect=OSError("synthetic write failure")), self.assertRaises(OSError):
            self.run_match()
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        self.run_match()
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)

    def test_cli_from_unrelated_directory_and_wrong_run(self):
        self.prepare()
        shutil.copytree(PROJECT / "src", self.root / "src")
        shutil.copytree(PROJECT / "scripts", self.root / "scripts")
        command = [sys.executable, "-B", str(self.root / "scripts/data/04_match_admissions.py"), "--version", "1", "--run", "1"]
        result = subprocess.run(command, cwd=self.root.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["status"], "complete")
        result = subprocess.run(command[:-1] + ["2"], cwd=self.root.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
