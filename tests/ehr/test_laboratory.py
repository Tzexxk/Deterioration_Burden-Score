"""Generated laboratory CSVs, cached review, explicit freeze and rollback."""

import csv
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_support import write_csv
from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.ehr import build_ehr_features as step
from src.ehr import lab_audit as audit
from src.ehr import lab_review as review
from tests.common import test_foundation as foundation

PROJECT = Path(__file__).resolve().parents[2]


class LaboratoryTests(unittest.TestCase):
    def test_slash_number_raw_result_is_preserved_in_published_reports(self):
        self.labs[0].update(value="/18.", valuenum="")
        self.write_sources()
        self.run_mode()
        with self.output("audit", "lab_examples.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(any(row["value"] == "/18." for row in rows))
        self.assertTrue(self.record()["audit_checks_passed"])

    def setUp(self):
        self.fixture = foundation.FoundationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.layout.project
        paths = self.root / "configs/environment/paths.local.json"
        settings = read_json(paths)
        settings["roots"]["mimic_iv"] = "raw"
        write_json(paths, settings)
        self.context = DataContext.load(Layout(self.root), 1, 1)
        self.store = ManifestStore(self.context.layout, data=self.context)
        self.raw = self.context.layout.roots["mimic_iv"]
        self.raw.mkdir(exist_ok=True)
        self.settings = read_json(PROJECT / "configs/data.template/lenient/ehr/lab_processing.json")
        self.settings.update(reviewer="synthetic", memory_limit="128MB", max_temp_directory_size="64MB", workspace_estimate_bytes=1024*1024, comment_max_chars=32)
        self.config = self.root / self.context.directory / "ehr/lab_processing.json"
        write_json(self.config, self.settings)
        self.cohort = pa.table({"sample_id": ["synthetic_a", "synthetic_b"], "subject_id": pa.array([1,1],type=pa.int64()),
                                "hadm_id": pa.array([10,10],type=pa.int64()),
                                "t0": pa.array([datetime(2020,1,2,12),datetime(2020,1,2,13)],type=pa.timestamp("us"))})
        with self.store.publish("step08", "freeze", data_files=("cohort.parquet",)) as work:
            pq.write_table(self.cohort, work / "cohort.parquet")
        with self.store.publish("step09", "split", inputs=[self.store.manifest_ref("step08","freeze")]) as work:
            write_json(work / "split_freeze.json", {"synthetic": True})
        # Clinical/cohort replay is tested in preceding stages; exercise real CSV/domain code here.
        guard = patch.object(step, "verify_upstream", return_value=self.cohort)
        guard.start()
        self.addCleanup(guard.stop)
        self.dictionary = [dict(itemid="100",label="Hemoglobin",fluid="Blood",category="Hematology"),
                           dict(itemid="101",label="Specimen Type",fluid="Blood",category="Blood Gas"),
                           dict(itemid="102",label="pH",fluid="Blood",category="Blood Gas"),
                           dict(itemid="900",label="Unrelated synthetic analyte",fluid="Blood",category="Other")]
        self.admissions = [dict(subject_id="1",hadm_id="10",admittime="2020-01-01 00:00:00",dischtime="2020-01-05 00:00:00")]
        self.labs = [self.lab("1"),self.lab("2",itemid="900",valueuom="other"),self.lab("3",itemid="102",valueuom="",value="7.2",valuenum="7.2")]
        self.write_sources()
        self.audit_ref = self.store.manifest_ref("step10", "audit")
        self.review_ref = self.store.manifest_ref("step10", "review")
        self.freeze_ref = self.store.manifest_ref("step10", "freeze")

    def lab(self, identifier, **changes):
        row = dict.fromkeys([*audit.LAB_REQUIRED,*audit.LAB_OPTIONAL], "")
        row.update(labevent_id=identifier,subject_id="1",hadm_id="10",specimen_id="20",itemid="100",
                   charttime="2020-01-02 11:00:00",storetime="2020-01-02 11:30:00",value="12",valuenum="12",valueuom="g/dL",comments="Synthetic evidence only")
        row.update(changes)
        return row

    def write_sources(self):
        hosp = self.raw / "hosp"
        hosp.mkdir(exist_ok=True)
        write_csv(hosp / "d_labitems.csv", ["itemid","label","fluid","category"], self.dictionary)
        write_csv(hosp / "admissions.csv", ["subject_id","hadm_id","admittime","dischtime"], self.admissions)
        write_csv(hosp / "labevents.csv", [*audit.LAB_REQUIRED,*audit.LAB_OPTIONAL], self.labs)

    def run_mode(self, mode="audit"):
        return step.run_step10(self.context, mode)

    def output(self, phase, name):
        manifest = self.store.validate(self.store.manifest_ref("step10",phase))
        return self.context.layout.resolve(self.store.output_ref(manifest,name))

    def record(self):
        return read_json(self.output("audit","lab_audit.json"))

    def write_reviews(self, complete=True):
        rows = []
        for entry in self.record()["inventory"]:
            if entry["suggested_group"] == "out_of_scope":
                continue
            row = dict.fromkeys(review.REVIEW_COLUMNS, "")
            row.update({key:entry[key] for key in ("review_key","itemid","label","fluid","category","source_unit")})
            row.update(record_type="decision",branch_id="default",reviewer="synthetic",reviewed_on="2026-11-05",
                       rationale="Synthetic decision",specimen_rule='{"kind":"item_identity"}')
            if row["source_unit"]:
                row.update(keep="1",feature_name="synthetic_hemoglobin",canonical_unit="g/dL",conversion_multiplier="1",conversion_offset="0",
                           valid_range_basis="Synthetic range policy",duplicate_rule='{"kind":"latest_available","conflicting_id":"exclude","missing_specimen":"record_id"}',
                           numeric_result_rule='{"kind":"valuenum_only","comparator":"exclude","text_disagreement":"exclude"}')
            else:
                row.update(keep="0",missing_unit_rule="exclude",exclusion_reason="Synthetic missing-unit exclusion")
            rows.append(row)
        acknowledgment = dict.fromkeys(review.REVIEW_COLUMNS, "")
        acknowledgment.update(record_type="scope_ack",review_key="scope",reviewer="synthetic",reviewed_on="2026-11-05",rationale="Synthetic scope acknowledgment")
        for name, values in zip(self.settings["reviews"], [rows if complete else [], [acknowledgment]], strict=True):
            path = self.root / self.context.directory / name
            path.parent.mkdir(parents=True,exist_ok=True)
            write_csv(path,review.REVIEW_COLUMNS,values)

    def test_audit_cache_inventory_and_record_keys(self):
        self.run_mode()
        record = self.record()
        self.assertEqual(record["counts"]["patient_scan_raw_rows"],3)
        self.assertEqual(record["counts"]["study_window_associations"],6)
        self.assertEqual(len(record["inventory"]),3)
        self.assertTrue(all(row["review_key"].startswith("lab_") for row in record["inventory"]))
        self.assertEqual({r["root"] for r in self.store.validate(self.audit_ref)["outputs"] if r["name"].startswith("cache/")},{"processed"})
        self.assertFalse(record["dictionary_frozen"])

    def test_review_and_freeze_use_cache_without_raw_csvs(self):
        self.run_mode()
        self.write_reviews()
        self.raw.rename(self.raw.with_name("raw_offline"))
        self.run_mode("review")
        self.assertTrue(read_json(self.output("review","review_state.json"))["review_complete"])
        self.run_mode("freeze")
        table = pq.read_table(self.output("freeze","lab_dictionary.parquet"))
        self.assertEqual(table.num_rows,3)
        self.assertEqual(sum(table["keep"].to_pylist()),1)
        self.assertEqual(set(table["dictionary_version"].to_pylist()),{"v1_r1"})
        self.assertNotIn("audit_fingerprint",table.column_names)
        self.assertEqual(self.store.validate(self.freeze_ref)["inputs"],[{"manifest":self.review_ref,"generation":1}])

    def test_incomplete_review_blocks_freeze(self):
        self.run_mode()
        self.write_reviews(complete=False)
        self.run_mode("review")
        with self.assertRaises(PipelineError):
            self.run_mode("freeze")
        self.assertFalse(self.context.layout.resolve(self.freeze_ref).exists())

    def test_identity_or_authorization_changes_reject_review(self):
        self.run_mode()
        self.write_reviews()
        path = self.root / self.context.directory / self.settings["reviews"][0]
        with path.open(encoding="utf-8",newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["label"] = "Different analyte"
        write_csv(path,review.REVIEW_COLUMNS,rows)
        with self.assertRaises(ValueError):
            self.run_mode("review")

    def test_exact_duplicates_and_comment_truncation_preserve_raw_keys(self):
        self.labs.append(self.labs[0].copy())
        self.labs.append(self.lab("4",comments="x"*100))
        self.write_sources()
        self.run_mode()
        self.assertEqual(self.record()["counts"]["patient_scan_raw_rows"],5)
        self.assertEqual(self.record()["counts"]["patient_scan_distinct_records"],4)
        rows = pq.read_table(self.output("audit", "cache/labs_candidate_subset.parquet")).to_pylist()
        self.assertTrue(any(row["comments_truncated"] and len(row["comments"])==32 for row in rows))
        self.assertTrue(all(len(row["lab_record_key"])==64 for row in rows))

    def test_time_availability_and_admission_inference(self):
        self.labs += [self.lab("4",storetime="2020-01-03 00:00:00"),self.lab("5",charttime="bad"),self.lab("6",hadm_id="")]
        self.write_sources()
        self.run_mode()
        record = self.record()
        self.assertEqual(record["counts"]["patient_scan_invalid_charttime_records"],1)
        self.assertLess(record["counts"]["eligible_associations"],record["counts"]["study_window_associations"])

    def test_incomplete_final_csv_record_is_reported(self):
        with (self.raw/"hosp/labevents.csv").open("ab") as stream:
            stream.write(b"bad")
        self.run_mode()
        self.assertEqual(self.record()["source_csv_tail_checks"]["skipped_record_count"],1)

    def test_audit_failure_rollback_retry_and_review_invalidation(self):
        self.run_mode()
        self.write_reviews()
        self.run_mode("review")
        path = self.output("audit","cache/labs_candidate_subset.parquet")
        before = path.read_bytes()
        with patch.object(step,"perform_audit",side_effect=RuntimeError("synthetic audit failure")),self.assertRaises(RuntimeError):
            self.run_mode()
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaises(PipelineError):
            self.store.validate(self.review_ref)
        self.run_mode()
        self.assertEqual(self.store.validate(self.audit_ref)["generation"],2)
        with self.assertRaises(PipelineError):
            self.store.validate(self.review_ref)

    def test_cross_version_review_and_unsupported_vital_scope_rejected(self):
        self.run_mode()
        with self.assertRaises(PipelineError):
            step.run_step10(self.context,"review",reviews=["../v1_r2/review.csv"])
        write_json(self.config,{**self.settings,"vital_audit":{}})
        with self.assertRaises(PipelineError):
            self.run_mode()
