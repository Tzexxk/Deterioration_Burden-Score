"""Generated clinical edge cases, cached feature audit and publication checks."""

import math
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_support import write_csv as _write_csv
from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.ehr import build_features as pipeline
from src.ehr import feature_engine as step
from src.ehr import lab_audit as lab
from src.ehr import lab_review as review
from tests.common import test_foundation as foundation

PROJECT = Path(__file__).resolve().parents[2]

def approved_dictionary():
    rows = []
    for item, (name, canonical) in step.APPROVED_FEATURES.items():
        units = (
            ["%", "g/dL"]
            if item == "51249"
            else [""]
            if item == "51237"
            else [canonical]
        )
        for unit in units:
            row: dict[str, Any] = dict.fromkeys(review.STRING_FIELDS, "")
            row.update(
                itemid=int(item),
                keep=True,
                feature_name=name,
                source_unit=unit,
                review_key=lab.review_key(item, unit),
                branch_id="item_identity",
                canonical_unit=canonical,
                dictionary_version="v1_r1",
                conversion_multiplier=1.0,
                conversion_offset=0.0,
                valid_min=None,
                valid_max=None,
                specimen_rule='{"kind":"item_identity"}',
                duplicate_rule='{"kind":"latest_available","conflicting_id":"error","missing_specimen":"error"}',
                numeric_result_rule='{"kind":"valuenum_only","comparator":"exclude","text_disagreement":"error"}',
                valid_range_basis="Synthetic reviewed scope",
                missing_unit_rule="explicit_item_unit" if item == "51237" else "",
            )
            rows.append(row)
    return pa.Table.from_pylist(rows, schema=review.DICTIONARY_SCHEMA)


class FeatureTests(unittest.TestCase):
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
        (self.raw / "hosp").mkdir(parents=True)
        self.output = self.context.layout.roots["outputs"]
        self.processed = self.context.layout.roots["processed"]
        self.cohort = pa.table(
            {
                "sample_id": ["a", "b", "c"],
                "subject_id": [1, 1, 2],
                "hadm_id": [10, 10, 20],
                "t0": pa.array(
                    [
                        datetime(2020, 1, 2, 12),
                        datetime(2020, 1, 2, 13),
                        datetime(2020, 1, 2, 12),
                    ],
                    type=pa.timestamp("us"),
                ),
            }
        )
        self.evidence = pa.table(
            {
                **self.cohort.to_pydict(),
                "admittime": [datetime(2020, 1, 1)] * 3,
                "age_at_t0": [50, 50, 60],
                "careunit_at_t0": ["Ward", "Ward", "Ward"],
            }
        )
        self.dictionary = approved_dictionary()
        self.rules = step.validate_dictionary(self.dictionary, "v1_r1")
        self.labs = [
            self.lab(
                "1", charttime="2020-01-01 12:00:00", storetime="2020-01-01 12:00:00"
            ),
            self.lab("2", specimen_id="21", valueuom="g/dL", value="34", valuenum="34"),
            self.lab(
                "3",
                itemid="51237",
                specimen_id="22",
                valueuom="",
                value="1.2",
                valuenum="1.2",
            ),
        ]
        self.admissions = [
            dict(
                subject_id=str(p),
                hadm_id=str(a),
                admittime="2020-01-01 00:00:00",
                dischtime="2020-01-05 00:00:00",
                admission_type="URGENT",
                admission_location="ED",
            )
            for p, a in [(1, 10), (2, 20)]
        ]
        self.patients = [
            dict(subject_id="1", gender="F", anchor_age="50", anchor_year="2020"),
            dict(subject_id="2", gender="M", anchor_age="60", anchor_year="2020"),
        ]
        self.services = [
            dict(
                subject_id="1",
                hadm_id="10",
                transfertime="2020-01-02 11:00:00",
                curr_service="MED",
            ),
            dict(
                subject_id="1",
                hadm_id="10",
                transfertime="2020-01-02 14:00:00",
                curr_service="SURG",
            ),
        ]
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.write_sources()
        self.settings = read_json(PROJECT / "configs/data.template/lenient/ehr/features.json")
        self.settings.update(memory_limit="128MB", max_temp_directory_size="64MB", workspace_estimate_bytes=1024*1024)
        self.config = self.root / self.context.directory / "ehr/features.json"
        write_json(self.config, self.settings)
        with self.store.publish("step08", "freeze", data_files=("cohort.parquet", "cohort_audit.parquet")) as work:
            pq.write_table(self.cohort, work / "cohort.parquet")
            pq.write_table(self.evidence, work / "cohort_audit.parquet")
        with self.store.publish("step09", "split", inputs=[self.store.manifest_ref("step08", "freeze")], data_files=("patient_split.parquet",)) as work:
            pq.write_table(pa.table({"subject_id":[1,2], "split":["train","validation"]}), work / "patient_split.parquet")
        with self.store.publish("step10", "audit", inputs=[self.store.manifest_ref("step09", "split")], data_directories=("cache",)) as work:
            (work / "cache").mkdir()
            for name, source in [("labs_candidate_subset", "labs"), ("admissions", "admissions"), ("d_labitems", "dictionary")]:
                pq.write_table(pq.read_table(self.cache / f"{source}.parquet"), work / f"cache/{name}.parquet")
            write_json(work / "lab_audit.json", {"data_run":"v1_r1", "audit_checks_passed": True})
        with self.store.publish("step10", "freeze", inputs=[self.store.manifest_ref("step10", "audit")], data_files=("lab_dictionary.parquet",)) as work:
            pq.write_table(self.dictionary, work / "lab_dictionary.parquet")
            write_json(work / "lab_dictionary_freeze.json", {"data_run":"v1_r1", "dictionary_version":"v1_r1", "dictionary_frozen":True, "review_complete":True})
        # Prior-stage clinical replay has its own tests; retain real CSV and feature code here.
        guard = patch.object(pipeline.upstream, "verify_upstream", return_value=self.cohort)
        guard.start()
        self.addCleanup(guard.stop)

    def lab(self, identifier, **changes):
        row: dict[str, Any] = dict.fromkeys([*lab.LAB_REQUIRED, *lab.LAB_OPTIONAL], "")
        row.update(
            labevent_id=identifier,
            subject_id="1",
            hadm_id="10",
            itemid="51249",
            specimen_id="20",
            charttime="2020-01-02 11:00:00",
            storetime="2020-01-02 11:30:00",
            value="32",
            valuenum="32",
            valueuom="%",
        )
        row.update(changes)
        return row

    def write_sources(self):
        for name, rows in [
            ("patients", self.patients),
            ("admissions", self.admissions),
            ("services", self.services),
        ]:
            _write_csv(
                self.raw / f"hosp/{name}.csv",
                list(rows[0]) if rows else step.SOURCE_COLUMNS[name].split(),
                rows,
            )
        _write_csv(
            self.raw / "hosp/labevents.csv",
            [*lab.LAB_REQUIRED, *lab.LAB_OPTIONAL],
            self.labs,
        )
        _write_csv(
            self.raw / "hosp/d_labitems.csv",
            ["itemid", "label", "fluid", "category"],
            [],
        )
        rows = [
            {
                **r,
                "lab_record_key": lab.json_hash(r),
                "raw_row_count": 1,
                "comments_truncated": False,
            }
            for r in self.labs
        ]
        pq.write_table(pa.Table.from_pylist(rows), self.cache / "labs.parquet")
        pq.write_table(
            pa.Table.from_pylist(self.admissions), self.cache / "admissions.parquet"
        )
        pq.write_table(
            pa.table({"itemid": pa.array([], type=pa.string())}),
            self.cache / "dictionary.parquet",
        )

    def connection(self):
        connection = duckdb.connect(":memory:")
        self.addCleanup(connection.close)
        lab.load_cache(
            connection,
            {
                "labs": self.cache / "labs.parquet",
                "admissions": self.cache / "admissions.parquet",
                "dictionary": self.cache / "dictionary.parquet",
            },
        )
        lab.build_views(connection, self.cohort)
        return connection

    def test_boundaries_units_and_overlapping_windows(self):
        self.labs.extend(
            [
                self.lab(
                    "4",
                    charttime="2020-01-02 12:00:00",
                    storetime="2020-01-02 12:00:00",
                    specimen_id="24",
                    value="36",
                    valuenum="36",
                ),
                self.lab("5", specimen_id="25", storetime="2020-01-02 14:00:00"),
                self.lab("6", specimen_id="26", storetime=""),
                self.lab("7", specimen_id="27", hadm_id="20"),
                self.lab("8", specimen_id="28", charttime="invalid"),
            ]
        )
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        rows = step.aggregate_labs(connection, self.cohort, self.rules)
        self.assertEqual(rows[0]["mchc_count_24h"], 3)
        self.assertEqual(rows[0]["mchc_last_24h"], 36)
        self.assertEqual(rows[0]["mchc_min_24h"], 32)
        self.assertEqual(rows[0]["mchc_hours_since_last"], 0)
        self.assertEqual(rows[1]["mchc_count_24h"], 2)
        self.assertEqual(rows[2]["mchc_count_24h"], 0)
        self.assertIsNone(rows[2]["mchc_last_24h"])
        self.assertEqual(
            connection.execute(
                "SELECT count(*) FROM cleaned WHERE labevent_id='2'"
            ).fetchall()[0][0],
            2,
        )

    def test_revision_precedes_numerical_filter(self):
        self.labs = [
            self.lab("1"),
            self.lab("2", storetime="2020-01-02 12:30:00", value="<32", valuenum="32"),
        ]
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        rows = step.aggregate_labs(connection, self.cohort, self.rules)
        self.assertEqual(rows[0]["mchc_count_24h"], 1)
        self.assertEqual(rows[1]["mchc_count_24h"], 0)
        self.assertEqual(
            connection.execute(
                "SELECT count(*) FROM lab_trace WHERE disposition='superseded_revision'"
            ).fetchall()[0][0],
            1,
        )

    def test_nonnumeric_latest_extremes_and_ties(self):
        self.labs = [
            self.lab("1", value="1e20", valuenum="1e20"),
            self.lab("2", value="not numeric", valuenum=""),
            self.lab("3", specimen_id="21", value="33", valuenum="33"),
            self.lab("4", specimen_id="22", value="34", valuenum="34"),
        ]
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        rows = step.aggregate_labs(connection, self.cohort, self.rules)
        self.assertEqual(rows[0]["mchc_count_24h"], 2)
        self.assertEqual(rows[0]["mchc_last_24h"], 34)
        self.assertEqual(rows[0]["mchc_max_24h"], 34)
        self.labs = [self.lab("1", value="1e20", valuenum="1e20")]
        self.write_sources()
        connection = self.connection()
        step.clean_labs(connection, self.rules)
        self.assertEqual(
            step.aggregate_labs(connection, self.cohort, self.rules)[0]["mchc_max_24h"],
            1e20,
        )

    def test_lab_conflicts_and_dictionary_validation(self):
        for changes in [
            {"specimen_id": ""},
            {"value": "31"},
            {"labevent_id": "invalid"},
        ]:
            with self.subTest(changes=changes):
                self.labs = [self.lab("1", **changes)]
                self.write_sources()
                with self.assertRaises(ValueError):
                    step.clean_labs(self.connection(), self.rules)
        rows = self.dictionary.to_pylist()
        rows[0]["source_unit"] = "unsupported"
        with self.assertRaises(ValueError):
            step.validate_dictionary(
                pa.Table.from_pylist(rows, schema=review.DICTIONARY_SCHEMA),
                "v1_r1",
            )
        with self.assertRaises(ValueError):
            step.validate_dictionary(self.dictionary, "other_version")

    def test_static_future_missing_conflicts_and_age(self):
        connection = self.connection()
        for name, rows in [
            ("patients", self.patients),
            ("admissions", self.admissions),
            ("services", self.services),
        ]:
            connection.register(f"static_{name}", pa.Table.from_pylist(rows))
        static, issues = step.static_features(connection, self.cohort, self.evidence)
        self.assertEqual(issues, [])
        self.assertEqual(static["a"]["current_service"], "MED")
        self.assertEqual(static["a"]["hours_from_admission"], 36)
        self.assertIsNone(static["c"]["current_service"])
        self.services.append({**self.services[0], "curr_service": "SURG"})
        connection.unregister("static_services")
        connection.register("static_services", pa.Table.from_pylist(self.services))
        _, issues = step.static_features(connection, self.cohort, self.evidence)
        self.assertTrue(
            any(i["reason"] == "conflicting_latest_service" for i in issues)
        )
        self.patients[0]["anchor_age"] = "51"
        connection.unregister("static_patients")
        connection.register("static_patients", pa.Table.from_pylist(self.patients))
        _, issues = step.static_features(connection, self.cohort, self.evidence)
        self.assertTrue(any(i["field"] == "age_at_t0" for i in issues))

    def test_float32_schema_overflow_and_whitelist(self):
        schema = step.schema_for(self.rules, "ehr_features_v1")
        self.assertEqual(len(schema["numeric_features"]), 152)
        self.assertEqual(len(schema["categorical_features"]), 5)
        rows: list[dict[str, Any]] = [{"sample_id": k} for k in ["a", "b", "c"]]
        table = step.float32_table(rows, schema)
        step.validate_wide(table, self.cohort, schema)
        self.assertTrue(math.isnan(table["age_at_t0"][0].as_py()))
        self.assertEqual(table["sex"][0].as_py(), "__MISSING__")
        with self.assertRaises(ValueError):
            step.validate_wide(
                table.append_column("burden_score", pa.array([0, 0, 0])),
                self.cohort,
                schema,
            )
        rows[0]["age_at_t0"] = 1e100
        with self.assertRaises(ValueError):
            step.float32_table(rows, schema)

    def output_path(self, phase, name):
        manifest = self.store.validate(self.store.manifest_ref("step11", phase))
        return self.context.layout.resolve(self.store.output_ref(manifest, name))

    def run_mode(self, mode="audit"):
        return pipeline.run_step11(self.context, mode)

    def test_audit_build_separates_outputs_and_works_without_raw_sources(self):
        self.assertEqual(self.run_mode(), 0)
        record = read_json(self.output_path("audit", "ehr_audit.json"))
        self.assertEqual((record["studies"], record["numeric_features"], record["categorical_features"]), (3,152,5))
        self.assertFalse(record["ehr_features_ready"])
        candidate = pq.read_table(self.output_path("audit", "cache/ehr_features_candidate.parquet"))
        self.raw.rename(self.raw.with_name("offline_raw"))
        self.assertEqual(self.run_mode("build"), 0)
        table = pq.read_table(self.output_path("build", "ehr_features_raw.parquet"))
        self.assertEqual(table.schema, candidate.schema)
        for name in table.column_names:
            for left, right in zip(table[name].to_pylist(), candidate[name].to_pylist(), strict=True):
                self.assertTrue(left == right or isinstance(left, float) and math.isnan(left) and math.isnan(right))
        manifest = self.store.validate(self.store.manifest_ref("step11", "build"))
        self.assertEqual(manifest["inputs"], [{"manifest":self.store.manifest_ref("step11","audit"),"generation":1}])
        self.assertEqual({row["name"] for row in manifest["outputs"] if row["root"] == "processed"}, {"ehr_features_raw.parquet", "feature_schema.json"})

    def test_feature_coverage_uses_fold_indices(self):
        self.assertEqual(self.run_mode(), 0)
        candidate = pq.read_table(self.output_path("audit", "cache/ehr_features_candidate.parquet"))
        schema = read_json(self.output_path("audit", "feature_schema_candidate.json"))
        inputs = pipeline.load_inputs(self.context, self.store)
        connection = self.connection()
        step.clean_labs(connection, inputs["rules"])
        master = [{"sample_id": key} for key in candidate["sample_id"].to_pylist()]
        indices = {"master": master, "folds": [
            {"fold": 1, "train_indices": [0, 1], "validation_indices": [2]},
            {"fold": 2, "train_indices": [2], "validation_indices": [0, 1]},
        ]}
        summary, missingness = step.feature_reports(
            connection, candidate, self.cohort, inputs["patients"], schema, cv_indices=indices
        )
        for report in (summary, missingness):
            self.assertEqual(
                {r["split"]: r["studies"] for r in report if r["split"].startswith("fold_")},
                {"fold_1_train": 2, "fold_1_validation": 1,
                 "fold_2_train": 1, "fold_2_validation": 2},
            )

    def test_static_conflict_keeps_diagnostics_but_blocks_build(self):
        self.services.append({**self.services[0], "curr_service":"SURG"})
        self.write_sources()
        self.assertEqual(self.run_mode(), 2)
        self.assertGreater(read_json(self.output_path("audit","ehr_audit.json"))["blocking_conflicts"], 0)
        with self.assertRaisesRegex(PipelineError, "FEATURE_AUDIT_BLOCKED"):
            self.run_mode("build")
        self.services.pop()
        self.write_sources()
        self.assertEqual(self.run_mode(), 0)
        self.assertEqual(self.store.validate(self.store.manifest_ref("step11","audit"))["generation"], 2)

    def test_lab_conflict_publishes_only_static_cache_and_can_recover(self):
        with patch.object(pipeline.engine, "clean_labs", side_effect=ValueError("Synthetic missing specimen")):
            self.assertEqual(self.run_mode(), 2)
        record = read_json(self.output_path("audit","ehr_audit.json"))
        self.assertFalse(record["candidate_available"])
        self.assertTrue(self.output_path("audit","cache/patients.parquet").exists())
        with self.assertRaisesRegex(PipelineError, "FEATURE_AUDIT_BLOCKED"):
            self.run_mode("build")
        self.assertEqual(self.run_mode(), 0)

    def test_audit_rebuild_invalidates_published_build(self):
        self.run_mode()
        self.run_mode("build")
        self.run_mode()
        with self.assertRaises(PipelineError):
            self.store.validate(self.store.manifest_ref("step11", "build"))
        self.run_mode("build")
        self.assertEqual(self.store.validate(self.store.manifest_ref("step11", "build"))["generation"], 2)

    def test_invalid_feature_policy_and_cross_version_config_rejected(self):
        self.settings["lookback_hours"] = 12
        write_json(self.config, self.settings)
        with self.assertRaisesRegex(PipelineError, "UNSUPPORTED_FEATURE_POLICY"):
            self.run_mode()
        with self.assertRaises(PipelineError):
            pipeline.run_step11(self.context, "audit", "../v1_r2/ehr/features.json")

    def test_source_change_rolls_back_audit(self):
        original = pipeline.perform_audit
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            with (self.raw / "hosp/services.csv").open("a") as stream:
                stream.write("\n")
            return result
        with patch.object(pipeline, "perform_audit", side_effect=change):
            with self.assertRaisesRegex(PipelineError, "SOURCE_CHANGED"):
                self.run_mode()
        self.assertFalse(self.context.layout.resolve(self.store.manifest_ref("step11","audit")).exists())

    def test_missing_audit_and_schema_changes_reject_build(self):
        with self.assertRaises(PipelineError):
            self.run_mode("build")
        self.run_mode()
        path = self.output_path("audit", "feature_schema_candidate.json")
        schema = read_json(path)
        schema["lookback_hours"] = 12
        write_json(path, schema)
        with self.assertRaisesRegex(PipelineError, "FEATURE_SCHEMA_MISMATCH"):
            self.run_mode("build")

    def test_csv_middle_record_is_not_silently_discarded(self):
        path = self.raw / "hosp/services.csv"
        path.write_text("subject_id,hadm_id,transfertime,curr_service\n1,10,broken\n1,10,2020-01-02 11:00:00,MED\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_mode()
        self.assertFalse(self.context.layout.resolve(self.store.manifest_ref("step11","audit")).exists())
