"""Generated training bundles, portable ZIPs and transactional export checks."""

import csv
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_io import COMPONENTS
from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.data import export_training_package as step
from src.data import package_contract as contract
from src.ehr import feature_engine as engine
from tests.common import test_foundation as foundation

PROJECT = Path(__file__).resolve().parents[2]


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = foundation.FoundationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.layout.project
        paths = self.root / "configs/environment/paths.local.json"
        roots = read_json(paths)
        roots["roots"]["mimic_cxr"] = "cxr"
        write_json(paths, roots)
        self.context = DataContext.load(Layout(self.root), 1, 1)
        self.store = ManifestStore(self.context.layout, data=self.context)
        self.settings = {"interface_version": "1.0", "workspace_estimate_bytes": 1024*1024}
        self.config = self.root / self.context.directory / "export/training_package.json"
        write_json(self.config, self.settings)
        cohort, patients = [], []
        for index, patient in enumerate([10000001, 10000001, 10000002, 10000003]):
            study, admission = 50000001+index, 20000000+patient
            row = dict.fromkeys(COMPONENTS, 0)
            row.update(sample_id=f"p{patient}_h{admission}_s{study}", subject_id=patient, hadm_id=admission,
                       study_id=study, dicom_id=f"synthetic_dicom_{index}", dicom_path=f"dataset/image_{index}.dcm",
                       view_position="AP", t0=datetime(2020,1,2,12), observation_end=datetime(2020,1,5,12),
                       observed_hours=72.0, early_discharge=0, burden_score=0.0, phys_death=0, high_burden=0,
                       processing_version="synthetic_pixels", processing_status="complete", qc_status="pass")
            row["npy_relpath"] = contract.expected_image_path(row)
            if index == 1:
                row.update({COMPONENTS[2]:1, COMPONENTS[5]:1, "burden_score":5.0, "phys_death":1, "high_burden":1})
            cohort.append(row)
            if patient not in {r["subject_id"] for r in patients}:
                patients.append({"subject_id":patient, "split":contract.SPLITS[len(patients)]})
        self.cohort = pa.Table.from_pylist(cohort)
        self.patients = pa.Table.from_pylist(patients)
        rules = [{"itemid":int(item), "feature_name":name, "canonical_unit":unit} for item,(name,unit) in engine.APPROVED_FEATURES.items()]
        self.schema = engine.schema_for(rules, self.context.key)
        ehr = engine.float32_table([{"sample_id":r["sample_id"], "age_at_t0":50.0, "sex":"F"} for r in cohort], self.schema)
        with self.store.publish("step05", "freeze") as work:
            write_json(work / "endpoint_decision.json", {"outcome_version":"synthetic", "target_formula":"max(icu,stepdown)+2*max(imv,hypoxemia)+2*vaso_inotrope+3*death"})
        with self.store.publish("step07", "prepare", inputs=[self.store.manifest_ref("step05","freeze")], data_directories=("images",), data_files=("image_qc_status.parquet",)) as work:
            pq.write_table(self.cohort, work / "image_qc_status.parquet")
            for index, row in enumerate(cohort):
                path = work / row["npy_relpath"]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.full((224,224), 0.1+index*0.1, dtype=np.float32), allow_pickle=False)
        with self.store.publish("step08", "freeze", inputs=[self.store.manifest_ref("step07","prepare")], data_files=("cohort.parquet",)) as work:
            pq.write_table(self.cohort, work / "cohort.parquet")
        with self.store.publish("step09", "split", inputs=[self.store.manifest_ref("step08","freeze")], data_files=("patient_split.parquet",)) as work:
            pq.write_table(self.patients, work / "patient_split.parquet")
        with self.store.publish("step11", "build", inputs=[self.store.manifest_ref("step09","split")], data_files=("ehr_features_raw.parquet","feature_schema.json")) as work:
            pq.write_table(ehr, work / "ehr_features_raw.parquet", compression="zstd")
            write_json(work / "feature_schema.json", self.schema)
        source = self.store.validate(self.store.manifest_ref("step11","build"))
        self.inputs = {"cohort":self.cohort, "patients":self.patients, "schema":self.schema, "ehr":ehr,
                       "ehr_path":self.context.layout.resolve(self.store.output_ref(source,"ehr_features_raw.parquet")),
                       "schema_path":self.context.layout.resolve(self.store.output_ref(source,"feature_schema.json")),
                       "qc_records":cohort, "image_manifest":self.store.validate(self.store.manifest_ref("step07","prepare")),
                       "image_settings":read_json(PROJECT/"configs/data.template/v1_r1/imaging/processing.json"),
                       "decision":{"outcome_version":"synthetic", "target_formula":"max(icu,stepdown)+2*max(imv,hypoxemia)+2*vaso_inotrope+3*death"},
                       "split_settings":{"algorithm_version":"synthetic", "seed":42, "weights":[70,15,15],
                                         "primary_stratum":"ever_phys_death", "require_high_burden_in_each_split":True}}
        # Clinical replay is covered upstream; retain real package, image and transaction code.
        guard = patch.object(step, "load_inputs", return_value=self.inputs)
        guard.start()
        self.addCleanup(guard.stop)

    def run_export(self):
        return step.run_export(self.context)

    def output(self, name):
        manifest = self.store.validate(self.store.manifest_ref("step12","export"))
        return self.context.layout.resolve(self.store.output_ref(manifest,name))

    def test_export_contract_and_relocated_archive(self):
        self.assertEqual(self.run_export(), 0)
        record = read_json(self.output("training_package_export.json"))
        self.assertEqual((record["studies"],record["images"],record["package_files"]), (4,4,10))
        self.assertEqual(record["split_counts"]["all"], {"studies":4,"admissions":3,"patients":3})
        self.assertFalse(record["independent_validation_completed"])
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            with zipfile.ZipFile(self.output("training_package.zip")) as archive:
                self.assertEqual(len(archive.namelist()),10)
                self.assertTrue(all(name.startswith("package/") for name in archive.namelist()))
                archive.extractall(temporary)
            package = Path(temporary)/"package"
            samples = pq.read_table(package/"samples.parquet")
            contract.validate_samples(samples)
            schema = read_json(package/"feature_schema.json")
            engine.validate_wide(pq.read_table(package/"ehr_features.parquet"), samples, schema)
            for row in samples.to_pylist():
                self.assertEqual(np.load(package/row["image_relpath"],allow_pickle=False).shape, (224,224))
            metadata = read_json(package/"dataset_metadata.json")
            self.assertEqual(metadata["release_id"],"v1_r1")
            self.assertEqual(len(metadata["files"]),10)
            self.assertFalse(any("sha256" in key for key in metadata))
        self.assertEqual({e["name"] for e in self.store.validate(self.store.manifest_ref("step12","export"))["outputs"] if e["root"]=="outputs"},
                         {"image_manifest.parquet","training_package_export.json","run_metadata.json","report.md"})

    def test_summary_has_distinct_patient_and_study_denominators(self):
        samples = contract.build_samples(self.cohort,self.patients)
        rows = contract.summary_rows(samples)
        all_rows = {r["metric"]:r for r in rows if r["split"]=="all"}
        self.assertEqual(all_rows["studies"]["value"],4)
        self.assertEqual(all_rows["patients"]["value"],3)
        self.assertEqual(all_rows["high_burden_studies"]["denominator"],4)
        self.assertEqual(all_rows["high_burden_patients"]["denominator"],3)

    def test_invalid_label_formula_time_and_path_are_rejected(self):
        for changes in [{"burden_score":2.0}, {"phys_death":1}, {"observed_hours":70.0},
                        {"observation_end":datetime(2020,1,6,12)}, {"npy_relpath":"../outside.npy"}]:
            rows = self.cohort.to_pylist()
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                contract.build_samples(pa.Table.from_pylist(rows),self.patients)

    def test_patient_assignments_must_be_complete_and_unique(self):
        for rows in [self.patients.to_pylist()[:-1], self.patients.to_pylist()*2,
                     [{"subject_id":10000001,"split":"invalid"},*self.patients.to_pylist()[1:]]]:
            with self.assertRaises(ValueError):
                contract.build_samples(self.cohort,pa.Table.from_pylist(rows))

    def test_qc_identity_conflict_aborts_without_export(self):
        rows = [r.copy() for r in self.inputs["qc_records"]]
        rows[0]["dicom_id"] = "different"
        self.inputs["qc_records"] = rows
        with self.assertRaisesRegex(PipelineError,"INVALID_IMAGE_QC"):
            self.run_export()
        self.assertFalse(self.context.layout.resolve(self.store.manifest_ref("step12","export")).exists())

    def test_feature_whitelist_violation_blocks_export(self):
        self.inputs["ehr"] = self.inputs["ehr"].append_column("burden_score", pa.array([0]*4))
        with self.assertRaises(ValueError):
            self.run_export()

    def test_pixels_are_checked_without_hash_bindings(self):
        source, target = self.root/"source.npy", self.root/"copy.npy"
        for values in [np.zeros((224,224),dtype=np.float64), np.zeros((10,10),dtype=np.float32),
                       np.full((224,224),np.nan,dtype=np.float32),np.full((224,224),2,dtype=np.float32)]:
            np.save(source,values,allow_pickle=False)
            with self.assertRaises(ValueError):
                step.copy_image(source,target,224)

    def test_source_change_during_image_copy_is_rejected(self):
        source, target = self.root/"source.npy", self.root/"copy.npy"
        np.save(source,np.zeros((224,224),dtype=np.float32),allow_pickle=False)
        with patch.object(step,"_file_version",side_effect=[{"size":1},{"size":2}]):
            with self.assertRaisesRegex(PipelineError,"SOURCE_CHANGED"):
                step.copy_image(source,target,224)

    def test_zip_is_deterministic_and_checks_actual_bytes(self):
        source=self.root/"value.txt"
        source.write_text("synthetic",encoding="utf-8")
        left,right=self.root/"left.zip",self.root/"right.zip"
        files={"value.txt":source}
        contract.archive_files(left,files,"package")
        contract.archive_files(right,files,"package")
        self.assertEqual(left.read_bytes(),right.read_bytes())
        step.verify_archive(left,files)
        source.write_text("different",encoding="utf-8")
        with self.assertRaisesRegex(PipelineError,"INVALID_PACKAGE_ZIP"):
            step.verify_archive(left,files)
        with self.assertRaises(ValueError):
            contract.archive_files(right,{"../escape":source},"package")

    def test_failed_reexport_rolls_back_both_destinations(self):
        self.run_export()
        before=self.output("package/samples.parquet").read_bytes()
        archive=self.output("training_package.zip")
        zip_bytes=archive.read_bytes()
        with patch.object(step,"verify_archive",side_effect=OSError("Synthetic interrupted export")):
            with self.assertRaises(OSError):
                self.run_export()
        self.assertEqual(archive.read_bytes(),zip_bytes)
        data=self.context.layout.path("processed","v1_r1/steps/step12/export/package/samples.parquet")
        self.assertEqual(data.read_bytes(),before)
        self.run_export()
        self.assertEqual(self.store.validate(self.store.manifest_ref("step12","export"))["generation"],2)

    def test_config_identity_and_version_boundaries(self):
        self.settings["release_id"]="independent_release"
        write_json(self.config,self.settings)
        with self.assertRaisesRegex(PipelineError,"INVALID_EXPORT_CONFIG"):
            self.run_export()
        with self.assertRaises(PipelineError):
            step.run_export(self.context,"../v1_r2/export/training_package.json")

    def test_published_image_references_are_relative(self):
        self.run_export()
        rows=pq.read_table(self.output("image_manifest.parquet")).to_pylist()
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r["dicom_root"]=="mimic_cxr" and r["npy_root"]=="processed" for r in rows))
        self.assertTrue(all(not Path(r["dicom_path"]).is_absolute() and not Path(r["npy_path"]).is_absolute() for r in rows))
        with self.output("package/summary.csv").open(encoding="utf-8",newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),156)
