"""Split configuration boundaries and atomic publication over a generated frozen cohort."""

import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.create_patient_split import (
    create_patient_split,
    load_settings,
    validate_settings,
)
from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.ehr.build_ehr_features import verify_upstream
from tests.cohort.test_patient_allocation import approved_settings, synthetic_cohort
from tests.common import test_foundation as foundation


class PatientSplitTests(unittest.TestCase):
    def setUp(self):
        self.fixture=foundation.FoundationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.context,self.store=self.fixture.ctx,self.fixture.store
        self.settings=approved_settings()
        self.config=self.context.layout.path("project",self.context.directory+"/cohort/patient_split.json")
        write_json(self.config,{k:v for k,v in self.settings.items() if k not in {"cohort_version","split_version"}})
        self.cohort=synthetic_cohort(n=12,high=3,phys=5)
        self.prior=self.store.manifest_ref("step08","freeze")
        self.ref=self.store.manifest_ref("step09","split")
        self.publish_cohort()
        self.generated_context={"cohort":self.cohort,"cohort_audit":self.cohort,
                                "source05":{"csv_tail_checks":{"status":"checked"}},
                                "source07":{"csv_tail_checks":{"status":"checked"}},
                                "clinical_snapshot":{"ehr":{"burden_v2":{"memory_limit":"128MB","max_temp_directory_size":"64MB"}}}}
        # Step08's clinical reconstruction is covered by its own end-to-end tests.
        # Here isolate publication, config identity and real source-leakage SQL.
        self.source_patch=patch("src.cohort.create_patient_split.load_frozen_cohort",return_value=self.generated_context)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.validation_patch=patch("src.cohort.create_patient_split.freeze_cohort.validate_cohort",side_effect=self.reconstruct)
        self.validation_patch.start()
        self.addCleanup(self.validation_patch.stop)

    def publish_cohort(self):
        with self.store.publish("step08","freeze",data_files=("cohort.parquet",)) as work:
            pq.write_table(self.cohort,work/"cohort.parquet")

    def reconstruct(self,connection,context):
        connection.register("common",self.cohort)
        connection.register("evidence",pa.Table.from_pylist([
            {"sample_id":r["sample_id"],"subject_id":r["subject_id"],"component":"in_hospital_death","source_record_key":str(r["subject_id"])}
            for r in self.cohort.to_pylist()]))
        return self.cohort,self.cohort

    def output(self,name):
        return self.context.layout.resolve(self.store.output_ref(self.store.validate(self.ref),name))

    def test_split_publication_counts_schema_repeat_and_directory_split(self):
        create_patient_split(self.context)
        manifest=self.store.validate(self.ref)
        self.assertEqual(manifest["inputs"],[{"manifest":self.prior,"generation":1}])
        self.assertEqual({r["name"] for r in manifest["outputs"] if r["root"]=="processed"},{"patient_split.parquet"})
        rows=pq.read_table(self.output("patient_split.parquet")).to_pylist()
        self.assertEqual(len(rows),12)
        self.assertEqual({r["split"] for r in rows},{"train","validation","test"})
        checks=read_json(self.output("split_checks.json"))
        self.assertTrue(checks["source_records_do_not_cross_splits"])
        self.assertEqual(checks["quotas"]["total"],[8,2,2])
        before=self.output("patient_split.parquet").read_bytes()
        create_patient_split(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"],2)
        self.assertEqual(self.output("patient_split.parquet").read_bytes(),before)
        self.assertEqual(read_json(self.output("run_metadata.json"))["csv_tail_checks"]["status"],"no_csv_inputs")

    def test_five_fold_publication_and_fixed_test_membership(self):
        with self.assertRaisesRegex(PipelineError, "SPLIT_METHOD_MISMATCH"):
            create_patient_split(self.context, split_method="five-fold")
        template = Path(__file__).resolve().parents[2] / "configs/data.template/v1_r2/cohort/patient_split.json"
        config = read_json(template)
        write_json(self.config, {
            **config, "confirmed": True, "reviewer": "synthetic",
            "reviewed_on": "2026-11-05", "rationale": "synthetic policy",
        })
        with self.assertRaisesRegex(PipelineError, "SPLIT_METHOD_MISMATCH"):
            create_patient_split(self.context)

        cohort = synthetic_cohort(n=40, high=12, phys=17)
        repeat = {**cohort.to_pylist()[0], "hadm_id": 9999, "study_id": 9999,
                  "sample_id": "p1000_h9999_s9999"}
        self.cohort = pa.concat_tables([cohort, pa.Table.from_pylist([repeat], schema=cohort.schema)])
        self.generated_context.update(cohort=self.cohort, cohort_audit=self.cohort)
        self.publish_cohort()
        create_patient_split(self.context, split_method="five-fold")

        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["details"]["split_method"], "five-fold")
        indices = read_json(self.output("cross_validation_indices.json"))
        validation_seen = set()
        names = {"cross_validation_indices.json"}
        for fold in indices["folds"]:
            name = f"fold_{fold['fold']}/patient_split.parquet"
            names.add(name)
            membership = {r["subject_id"]: r["split"] for r in pq.read_table(self.output(name)).to_pylist()}
            for part, ids in (
                ("train", fold["train_indices"]),
                ("validation", fold["validation_indices"]),
                ("test", indices["test_indices"]),
            ):
                self.assertEqual(
                    {i for i, row in enumerate(indices["master"]) if membership[row["subject_id"]] == part},
                    set(ids),
                )
            self.assertFalse(validation_seen.intersection(fold["validation_indices"]))
            validation_seen.update(fold["validation_indices"])
        self.assertEqual(validation_seen, set(indices["development_indices"]))
        self.assertEqual({r["name"] for r in manifest["outputs"] if r["root"] == "processed"}, names)
        with duckdb.connect() as connection:
            self.assertTrue(verify_upstream(self.context, self.store, connection).equals(self.cohort))
        # The secondary folds must be checked independently of the primary view.
        original = pq.read_table
        fold_2 = self.output("fold_2/patient_split.parquet")
        with patch("src.ehr.build_ehr_features.pq.read_table",
                   side_effect=lambda path: original(path).slice(1) if path == fold_2 else original(path)):
            with duckdb.connect() as connection, self.assertRaisesRegex(PipelineError, "fold 2"):
                verify_upstream(self.context, self.store, connection)

    def test_unconfirmed_unsupported_or_independent_versions_are_rejected(self):
        base=read_json(self.config)
        for update in [{"confirmed":False},{"seed":7},{"algorithm_version":"patient_grouped_study_balanced_cv_v1"},{"split_version":"other"}]:
            write_json(self.config,{**base,**update})
            with self.assertRaises(PipelineError):
                create_patient_split(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())
        with self.assertRaises(PipelineError):
            validate_settings({**self.settings,"cohort_version":"other"})

    def test_cross_version_config_and_cohort_rebuild_are_rejected(self):
        with self.assertRaises(PipelineError):
            load_settings(self.context,"../v1_r2/split.json")
        create_patient_split(self.context)
        self.publish_cohort()
        with self.assertRaises(PipelineError):
            self.store.validate(self.ref)

    def test_failed_rerun_rolls_back_and_successful_retry_increments_generation(self):
        create_patient_split(self.context)
        path=self.output("patient_split.parquet")
        before=path.read_bytes()
        with patch("src.cohort.create_patient_split.split_reports",side_effect=RuntimeError("synthetic report failure")),self.assertRaises(RuntimeError):
            create_patient_split(self.context)
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaises(PipelineError):
            self.store.validate(self.ref)
        create_patient_split(self.context)
        self.assertEqual(self.store.validate(self.ref)["generation"],2)

    def test_cohort_reconstruction_difference_blocks_publication(self):
        with patch("src.cohort.create_patient_split.freeze_cohort.validate_cohort",return_value=(self.cohort.slice(1),self.cohort)),self.assertRaises(PipelineError):
            create_patient_split(self.context)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())
