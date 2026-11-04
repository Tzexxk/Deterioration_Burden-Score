"""Tiny real DICOM files and synthetic CSVs; no patient data is accessed."""

import csv
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from src.cohort.audit_raw_data import _audit_index_row, run_audit
from src.cohort.audit_support import PortableWriter
from src.cohort.raw_io import _acquisition_time, _indexed_dicom_path
from src.common.config import DataContext
from src.common.csv_io import CSV_TAIL_BYTES, _check_csv_tail, _csv_reader
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError

PROJECT = Path(__file__).resolve().parents[2]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class RawAuditTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT / "outputs/step01_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
        write_json(self.root / "configs/environment/paths.local.json", {"roots": {
            "mimic_cxr": "local_inputs/cxr", "mimic_iv": "local_inputs/iv"}})
        write_json(self.root / "configs/version_run.json", {"version": 1, "run": 1})
        for group in ("imaging", "ehr"):
            value = json.loads((PROJECT / f"configs/data.template/v1_r1/{group}/raw_audit.json").read_text(encoding="utf-8"))
            if group == "imaging":
                value["raw_audit"].update(dicom_workers=1, dicom_batch_size=2)
            else:
                value["raw_audit"]["lab_sample_rows"] = 2
            write_json(self.root / f"configs/data/v1_r1/{group}/raw_audit.json", value)
        self.layout = Layout(self.root)
        self.context = DataContext.load(self.layout, 1, 1)
        self.cxr = self.layout.roots["mimic_cxr"]
        self.iv = self.layout.roots["mimic_iv"]
        self.rows = [{"subject_id": "10000000", "study_id": str(100 + i), "dicom_id": f"abc{i+1}",
                      "path": f"files/p10/p10000000/s{100+i}/abc{i+1}.dcm"} for i in range(3)]
        self.record = self.cxr / "cxr-record-list.csv"
        write_csv(self.record, list(self.rows[0]), self.rows)
        write_csv(self.cxr / "cxr-study-list.csv", ["subject_id", "study_id", "path"],
                  [{key: row[key] for key in ("subject_id", "study_id", "path")} for row in self.rows])
        self.images = []
        for i, view in enumerate(("AP", "LATERAL")):
            path = self.cxr / f"dataset/p10000000/s{100+i}/abc{i+1}.dcm"
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            data = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            data.SOPClassUID = meta.MediaStorageSOPClassUID
            data.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            data.Rows = data.Columns = 2
            data.SamplesPerPixel = 1
            data.PhotometricInterpretation = "MONOCHROME2"
            data.BitsAllocated = data.BitsStored = 16
            data.HighBit = 15
            data.PixelRepresentation = 0
            data.PixelData = struct.pack("<4H", 1, 2, 3, 4)
            data.ViewPosition = view
            if i == 0:
                data.AcquisitionDateTime = "20200101120000"
            else:
                data.StudyDate = "20200101"
                data.StudyTime = "130000"
            data.save_as(path, enforce_file_format=True)
            self.images.append(path)
        settings = self.context.read("ehr/raw_audit.json")["raw_audit"]
        for specification in settings["required_tables"].values():
            row = {column: "1" for column in specification["columns"]}
            write_csv(self.iv / specification["path"], specification["columns"], [row])
        columns = settings["required_tables"]["labevents"]["columns"]
        write_csv(self.iv / "hosp/labevents.csv", columns, [
            {**dict.fromkeys(columns, "1"), "charttime": "2020-01-01 00:00:00", "storetime": "2020-01-01 00:00:00"},
            {**dict.fromkeys(columns, "1"), "charttime": "invalid", "storetime": "2020-01-01 01:00:00"},
            {**dict.fromkeys(columns, "1"), "charttime": "2020-01-01 00:00:00", "storetime": "2020-01-01 02:00:00"},
        ])
        self.store = ManifestStore(self.layout, data=self.context)
        self.ref = self.store.manifest_ref("step01", "audit", area="outputs")
        self.output = self.layout.resolve(self.ref).parent
        capture = redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def read_csv(self, name):
        with (self.output / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_complete_audit_counts_dicom_pixels_and_partial_lab_screen(self):
        self.assertEqual(run_audit(self.context), 0)
        manifest = self.store.validate(self.ref)
        counts = manifest["details"]["counts"]
        self.assertEqual(counts, {"physical_dicom_files": 2, "index_records": 3,
            "indexed_paths_existing": 2, "metadata_readable": 2, "unique_indexed_files_existing": 2,
            "path_resolutions": {"flattened": 2}, "pixel_readable": 2,
            "unique_pixel_readable_files": 2, "time_recoverable": 2,
            "frontal_pixel_readable": 1, "failures": {"missing_path": 1}})
        self.assertEqual(len(manifest["outputs"]), 8)
        self.assertEqual(manifest["generation"], 1)
        rows = self.read_csv("dicom_readability.csv")
        self.assertTrue(rows[0]["resolved_path"].startswith("mimic_cxr:dataset/"))
        ehr = {row["metric"]: row for row in self.read_csv("ehr_time_field_screen.csv")}
        self.assertEqual(ehr["sample_rows"]["count"], "2")
        self.assertEqual(ehr["charttime_invalid"]["count"], "1")
        for path in self.output.iterdir():
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(str(self.root), text)
            self.assertNotIn(self.root.as_posix(), text)
            self.assertNotIn("sha256", text.lower())
        self.assertFalse(any(self.output.parent.rglob("temp")))

    def test_failed_rerun_preserves_complete_files_and_saves_diagnostic(self):
        run_audit(self.context)
        original = self.layout.resolve(self.ref).read_bytes()
        (self.iv / "hosp/patients.csv").unlink()
        self.error("RAW_AUDIT_INCOMPLETE", lambda: run_audit(self.context))
        self.assertEqual(self.layout.resolve(self.ref).read_bytes(), original)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        diagnostic = self.layout.path("outputs", "data/v1_r1/diagnostics/step01_last_failure.json")
        failed = json.loads(diagnostic.read_text(encoding="utf-8"))
        self.assertEqual(failed["status"], "incomplete")
        self.assertIn("mimic_iv:hosp/patients.csv: missing_file", failed["critical_issues"])
        self.assertFalse(any(self.output.parent.rglob("temp")))

    def test_truncated_last_csv_record_is_skipped_once(self):
        with self.record.open("ab") as stream:
            stream.write(b"999")
        run_audit(self.context)
        summary = self.store.validate(self.ref)["details"]
        self.assertEqual(summary["counts"]["index_records"], 3)
        self.assertEqual(summary["csv_tail_checks"]["skipped_record_count"], 1)

    def test_middle_csv_error_is_not_silently_skipped(self):
        self.record.write_text("subject_id,study_id,dicom_id,path\n1,2\n" +
                               ",".join(self.rows[0].values()) + "\n", encoding="utf-8")
        self.error("RAW_AUDIT_INCOMPLETE", lambda: run_audit(self.context))
        self.assertFalse(self.layout.resolve(self.ref).exists())

    def test_disabled_pixel_decode_cannot_publish_success(self):
        path = self.root / "configs/data/v1_r1/imaging/raw_audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_audit"]["decode_pixels"] = False
        write_json(path, value)
        self.error("RAW_AUDIT_INCOMPLETE", lambda: run_audit(self.context))

    def test_corrupt_dicom_is_reported_as_a_per_image_failure(self):
        self.images[0].write_bytes(b"invalid dicom")
        run_audit(self.context)
        counts = self.store.validate(self.ref)["details"]["counts"]
        self.assertEqual(counts["pixel_readable"], 1)
        self.assertEqual(counts["failures"]["metadata_read_error:InvalidDicomError"], 1)

    def test_parallel_workers_keep_the_same_order_and_counts(self):
        path = self.root / "configs/data/v1_r1/imaging/raw_audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_audit"].update(dicom_workers=2, dicom_batch_size=2)
        write_json(path, value)
        run_audit(self.context)
        self.assertEqual(self.store.validate(self.ref)["details"]["counts"]["pixel_readable"], 2)
        self.assertEqual([row["dicom_id"] for row in self.read_csv("dicom_readability.csv")], [row["dicom_id"] for row in self.rows])

    def test_cli_from_other_directory_uses_fixture_project_and_rejects_wrong_run(self):
        shutil.copytree(PROJECT / "src", self.root / "src")
        scripts = self.root / "scripts/data"
        scripts.mkdir(parents=True)
        script = scripts / "01_audit_data.py"
        shutil.copyfile(PROJECT / "scripts/data/01_audit_data.py", script)
        result = subprocess.run([sys.executable, "-B", str(script), "--version", "1", "--run", "1"],
                                cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(str(self.root), result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["counts"]["physical_dicom_files"], 2)
        before = self.layout.resolve(self.ref).read_bytes()
        wrong = subprocess.run([sys.executable, "-B", str(script), "--version", "1", "--run", "2"],
                               cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(wrong.returncode, 2)
        self.assertIn("VERSION_MISMATCH", wrong.stderr)
        self.assertEqual(self.layout.resolve(self.ref).read_bytes(), before)

    def test_flattened_and_standard_resolution_and_key_checks(self):
        row = self.rows[0]
        standard = self.cxr / "dataset" / row["path"]
        flat = self.images[0]
        for layout, expected in (("auto", flat), ("flattened", flat), ("standard", standard)):
            resolved = _indexed_dicom_path(self.cxr, self.cxr / "dataset", row, layout, {expected})
            self.assertEqual(resolved[0], expected)
        with self.assertRaisesRegex(ValueError, "index_path_key_mismatch"):
            _indexed_dicom_path(self.cxr, self.cxr / "dataset", {**row, "subject_id": "20000000"}, "auto", set())

    def test_unsafe_index_path_is_redacted_in_serialized_diagnostic_row(self):
        unsafe = {**self.rows[0], "path": str(self.root / "outside.dcm")}
        row = _audit_index_row(unsafe, self.cxr, self.cxr / "dataset", "auto", set(), True)
        self.assertEqual(row["failure_reason"], "unsafe_index_path")
        stream = io.StringIO()
        writer = PortableWriter(stream, list(row), self.layout)
        writer.writeheader()
        writer.writerow(row)
        self.assertIn("<invalid_index_path>", stream.getvalue())
        self.assertNotIn(str(self.root), stream.getvalue())

    def test_time_priority_timezone_and_fallback(self):
        base = {"StudyDate": "20200101", "StudyTime": "120000"}
        self.assertEqual(_acquisition_time(base), ("2020-01-01T12:00:00", "StudyDate+StudyTime", ""))
        self.assertEqual(_acquisition_time({**base, "AcquisitionDateTime": "bad"})[1], "StudyDate+StudyTime")
        self.assertEqual(_acquisition_time({**base, "TimezoneOffsetFromUTC": "+0800"})[0], "2020-01-01T12:00:00+08:00")
        self.assertEqual(_acquisition_time({**base, "TimezoneOffsetFromUTC": "+0860"})[2], "invalid_timezone_offset")

    def test_python_csv_quoted_records_and_bounded_tail(self):
        path = self.root / "small.csv"
        path.write_bytes(b'a,b,c\n1,"two, commas",3\n2,"multi\nline",')
        check = _check_csv_tail(path)
        with path.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(_csv_reader(stream, check))), 2)
        self.assertEqual(check["status"], "complete")
        path.write_bytes(b'a,b,c\n1,"' + b"x" * (CSV_TAIL_BYTES + 20) + b'\nline",3')
        check = _check_csv_tail(path)
        self.assertEqual(check["status"], "unverified")
        with path.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(_csv_reader(stream, check))), 1)
        self.assertEqual(check["status"], "complete")


if __name__ == "__main__":
    unittest.main()
