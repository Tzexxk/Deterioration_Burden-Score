"""Independent reconstruction and release gating with generated data only."""

import json
import unittest
import zipfile
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.burden_support import write_csv
from src.cohort.raw_io import _file_version
from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.data import acceptance_checks as step
from src.data import package_contract as contract
from src.data import validate_training_package as pipeline
from tests.common import test_foundation as foundation


class AcceptanceTests(unittest.TestCase):
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
        self.settings = {"interface_version":"1.0", "expected_counts":{"studies":3,"admissions":3,"patients":3},
                         "expected_split_studies":dict.fromkeys(step.SPLITS,1), "expected_split_patients":dict.fromkeys(step.SPLITS,1),
                         "relocation_policy":"temporary_directory", "failure_detail_limit":10,"workspace_estimate_bytes":1024*1024}
        self.config = self.root / self.context.directory / "export/validation.json"
        write_json(self.config, self.settings)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.lab_path = self.cache / "labs.parquet"
        self.t0 = datetime(2020,1,2,12)
        cohort, samples, patients, static_patients, admissions, services = [], [], [], [], [], []
        for i,group in enumerate(step.SPLITS,1):
            patient, admission, study = 10000+i, 20000+i, 50000+i
            row = dict.fromkeys(step.EVENTS,0)
            row.update(sample_id=f"p{patient}_h{admission}_s{study}",subject_id=patient,hadm_id=admission,study_id=study,
                       dicom_id=f"synthetic_{i}",t0=self.t0,admittime=self.t0-timedelta(days=2),dischtime=self.t0+timedelta(days=4),
                       hospital_expire_flag="0",age_at_t0=50,careunit_at_t0="WARD",observation_end=self.t0+timedelta(hours=72),
                       observed_hours=72.0,early_discharge=0,burden_score=0.0,phys_death=0,high_burden=0,
                       npy_relpath=f"images/p10/p{patient}/s{study}.npy",dicom_path=f"dataset/{i}.dcm",
                       view_position="AP",qc_status="pass",processing_status="complete",processing_version="synthetic")
            for event in step.EVENTS:
                row[event+"_first_event_time"] = None
            if i == 2:
                row.update(burden_score=5.0,phys_death=1,high_burden=1,hospital_expire_flag="1")
                for event in (step.EVENTS[2],step.EVENTS[5]):
                    row[event] = 1
                    row[event+"_first_event_time"] = self.t0+timedelta(hours=3)
            dicom = self.context.layout.path("mimic_cxr", row["dicom_path"])
            dicom.parent.mkdir(parents=True,exist_ok=True)
            dicom.write_bytes(b"synthetic")
            row["source_version"] = json.dumps(_file_version(dicom))
            cohort.append(row)
            samples.append({key:group if key=="split" else row["npy_relpath"] if key=="image_relpath" else row[key] for key in step.SAMPLE_SCHEMA.names})
            patients.append({"subject_id":patient,"split":group})
            static_patients.append({"subject_id":str(patient),"gender":"M","anchor_age":"50","anchor_year":"2020"})
            admissions.append({"subject_id":str(patient),"hadm_id":str(admission),"admittime":str(row["admittime"]),"dischtime":str(row["dischtime"]),"admission_type":"EMERGENCY","admission_location":"ED"})
            services.append({"subject_id":str(patient),"hadm_id":str(admission),"transfertime":str(self.t0),"curr_service":"MED"})
        self.cohort_rows = cohort
        self.cohort = pa.Table.from_pylist(cohort)
        self.samples = pa.Table.from_pylist(samples,schema=step.SAMPLE_SCHEMA)
        self.split = pa.Table.from_pylist(patients)
        self.rules = [{"keep":True,"itemid":1,"feature_name":"synthetic_lab","source_unit":"mg/dL","canonical_unit":"mg/dL",
                       "conversion_multiplier":1.0,"conversion_offset":0.0,"valid_min":None,"valid_max":None,"valid_range_basis":"Synthetic bounds",
                       "specimen_rule":'{"kind":"item_identity"}',"duplicate_rule":'{"kind":"latest_available","conflicting_id":"error","missing_specimen":"error"}',
                       "numeric_result_rule":'{"kind":"valuenum_only","comparator":"exclude","text_disagreement":"error"}',"missing_unit_rule":"exclude"}]
        numeric = ["synthetic_lab_"+a for a in step.AGGREGATIONS]+["age_at_t0","hours_from_admission"]
        self.schema = {"feature_set_version":"v1_r1","numeric_features":numeric,"categorical_features":step.CATEGORIES,
                       "features":dict.fromkeys(numeric+step.CATEGORIES,{}),"key_column":"sample_id","lookback_hours":24,
                       "categorical_missing_value":"__MISSING__","preprocessing_applied":False}
        rows = [{"sample_id":row["sample_id"],"age_at_t0":50,"hours_from_admission":48,"sex":"M","admission_type":"EMERGENCY",
                 "admission_location":"ED","careunit_at_t0":"WARD","current_service":"MED",
                 **{"synthetic_lab_"+a:0 if a=="count_24h" else np.nan for a in step.AGGREGATIONS}} for row in cohort]
        self.ehr = pa.Table.from_pylist(rows,schema=pa.schema([("sample_id",pa.string()),*[(n,pa.float32()) for n in numeric],*[(n,pa.string()) for n in step.CATEGORIES]]))
        self.static = {"patients":pa.Table.from_pylist(static_patients),"admissions":pa.Table.from_pylist(admissions),"services":pa.Table.from_pylist(services)}
        self.write_labs([])
        with self.store.publish("step07","prepare",data_directories=("images",)) as work:
            for row in cohort:
                path = work/row["npy_relpath"]
                path.parent.mkdir(parents=True,exist_ok=True)
                np.save(path,np.full((224,224),0.25,dtype=np.float32),allow_pickle=False)
        image_manifest = self.store.validate(self.store.manifest_ref("step07","prepare"))
        images = []
        for row,sample in zip(cohort,samples,strict=True):
            ref = self.store.output_ref(image_manifest,row["npy_relpath"])
            images.append({**row,"dicom_root":"mimic_cxr","npy_root":ref["root"],"npy_path":ref["path"],"split":sample["split"]})
        self.image_settings = {"size":224,"processing_version":"synthetic"}
        self.split_settings = {"algorithm_version":"synthetic","seed":42,"weights":[70,15,15],"primary_stratum":"ever_phys_death","require_high_burden_in_each_split":True}
        self.decision = {"target_formula":"max(icu,stepdown)+2*max(imv,hypoxemia)+2*vaso_inotrope+3*death","outcome_version":"synthetic"}
        upstream = [{"manifest":self.store.manifest_ref("step07","prepare"),"generation":1}]
        with self.store.publish("step12","export",inputs=[upstream[0]["manifest"]],data_directories=("package",),data_files=("training_package.zip",)) as work:
            self.package = work/"package"
            self.package.mkdir()
            pq.write_table(self.samples,self.package/"samples.parquet")
            pq.write_table(self.ehr,self.package/"ehr_features.parquet")
            write_json(self.package/"feature_schema.json",self.schema)
            (self.package/"README.md").write_text("Synthetic package",encoding="utf-8")
            for row in cohort:
                target = self.package/row["npy_relpath"]
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(self.context.layout.resolve(self.store.output_ref(image_manifest,row["npy_relpath"])).read_bytes())
            summaries = step.expected_summary(self.samples)
            write_csv(self.package/"summary.csv",["split","metric","value","unit","denominator"],summaries)
            self.counts = {group:{r["metric"]:r["value"] for r in summaries if r["split"]==group and r["metric"] in {"studies","admissions","patients"}} for group in ("all",*step.SPLITS)}
            metadata = {name:"v1_r1" for name in ("release_id","data_run","cohort_version","split_version","dictionary_version","feature_set_version")}
            metadata.update(self.decision,interface_version="1.0",horizon_hours=72,ehr_lookback_hours=24,target_column="burden_score",target_range=[0,8],
                            split_column="split",split_values=list(step.SPLITS),split_unit="subject_id",ehr_preprocessing_applied=False,independent_validation_completed=False,
                            image={"format":"npy","dtype":"float32","shape":[224,224],"value_range":[0.0,1.0],"model_normalization_applied":False},image_config=self.image_settings,
                            hypoxemia_source="MIMIC-IV ICU chartevents only",live_discharge_rule="Retain events in (t0, dischtime] when alive discharge precedes t0+H; absent recorded events are operational zeros.",
                            split_rules=self.split_settings,feature_counts={"numeric":8,"categorical":5},split_counts=self.counts,upstream=upstream,
                            files=sorted(pipeline.PACKAGE_FILES|{r["npy_relpath"] for r in cohort}))
            write_json(self.package/"dataset_metadata.json",metadata)
            self.inventory = [{"name":p.relative_to(self.package).as_posix(),"bytes":p.stat().st_size} for p in sorted(self.package.rglob("*")) if p.is_file()]
            self.inventory.sort(key=lambda item:item["name"])
            self.archive = work/"training_package.zip"
            self.write_zip()
        exported = self.store.validate(self.store.manifest_ref("step12","export"))
        self.package = self.context.layout.resolve(self.store.output_ref(exported,"package/samples.parquet")).parent
        self.archive = self.context.layout.resolve(self.store.output_ref(exported,"training_package.zip"))
        self.source = {"manifests":{("step07","prepare"):image_manifest},"export_manifest":exported,
                       "export":{"package_inventory":self.inventory,"upstream":upstream,"split_counts":self.counts,"studies":3,"images":3,"package_files":9},
                       "package":self.package,"archive":self.archive,"cohort":self.cohort,"evidence":self.cohort,"split":self.split,"samples":self.samples,
                       "ehr":self.ehr,"schema":self.schema,"rules":self.rules,"labs":self.lab_path,"lab_admissions":self.static["admissions"],
                       "static":self.static,"images":pa.Table.from_pylist(images),"qc":self.cohort,"decision":self.decision,
                       "image_settings":self.image_settings,"split_settings":self.split_settings,"watched":{self.lab_path:_file_version(self.lab_path)}}

    def write_zip(self, extra=None):
        with zipfile.ZipFile(self.archive,"w",compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.package.rglob("*")):
                if path.is_file():
                    archive.write(path,"package/"+path.relative_to(self.package).as_posix())
            if extra:
                archive.writestr(extra,b"synthetic")

    def raw_lab(self, **changes):
        row = {"lab_record_key":"record1","labevent_id":"1","subject_id":str(self.cohort_rows[0]["subject_id"]),
               "hadm_id":str(self.cohort_rows[0]["hadm_id"]),"specimen_id":"1","itemid":"1","charttime":str(self.t0),
               "storetime":str(self.t0),"value":"7","valuenum":"7","valueuom":"mg/dL"}
        row.update(changes)
        return row

    def write_labs(self, rows):
        pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([(name,pa.string()) for name in self.raw_lab()])),self.lab_path)

    def labs(self, actual=None, admissions=None):
        return step.reconstruct_labs(self.lab_path,admissions or self.static["admissions"],self.cohort,self.rules,actual or self.ehr)

    def with_feature(self, column, value):
        rows = self.ehr.to_pylist()
        rows[0][column] = value
        return pa.Table.from_pylist(rows,schema=self.ehr.schema)

    def run_synthetic(self):
        # Real provenance loading gets a read-only preflight; isolate independent domain checks here.
        with patch.object(pipeline,"verify_provenance",return_value=self.source):
            return pipeline.run_validation(self.context)

    def report(self):
        manifest = self.store.validate(self.store.manifest_ref("step13","validate"))
        return read_json(self.context.layout.resolve(self.store.output_ref(manifest,"training_package_acceptance.json")))

    def test_success_checks_every_domain_then_freezes_run(self):
        self.assertEqual(self.run_synthetic(),0)
        self.assertTrue(self.report()["training_package_accepted"])
        release = read_json(self.store.control/"release.json")
        self.assertEqual(release["acceptance"],self.store.manifest_ref("step13","validate"))
        self.assertEqual(release["generation"],1)
        with self.assertRaisesRegex(PipelineError,"RELEASE_IMMUTABLE"):
            self.run_synthetic()

    def test_failed_reconstruction_keeps_report_without_freezing(self):
        self.source["ehr"] = self.with_feature("synthetic_lab_count_24h",1)
        self.assertEqual(self.run_synthetic(),1)
        self.assertEqual(self.report()["checks"]["laboratory_reconstruction"]["status"],"fail")
        self.assertFalse((self.store.control/"release.json").exists())
        self.source["ehr"] = self.ehr
        self.assertEqual(self.run_synthetic(),0)
        self.assertEqual(read_json(self.store.control/"release.json")["generation"],2)

    def test_incomplete_lineage_is_not_accepted(self):
        self.assertEqual(pipeline.run_validation(self.context),1)
        self.assertEqual(self.report()["checks"]["provenance"]["status"],"fail")
        self.assertFalse((self.store.control/"release.json").exists())

    def test_metadata_tampering_is_independently_rejected(self):
        path = self.package/"dataset_metadata.json"
        data = read_json(path)
        data["horizon_hours"] = 48
        write_json(path,data)
        with self.assertRaises(ValueError):
            pipeline.validate_metadata(self.context,self.source,self.settings)

    def test_zip_rejects_unknown_or_traversal_member(self):
        for name in ("package/unknown", "../escape", "/outside"):
            self.write_zip(extra=name)
            with self.assertRaises(ValueError):
                pipeline.relocate_archive(self.source,self.root)

    def test_source_state_change_is_rejected(self):
        self.write_labs([self.raw_lab()])
        with self.assertRaises(ValueError):
            pipeline.verify_stability(self.source)

    def test_missing_export_manifest_still_publishes_failed_diagnostics(self):
        path = self.context.layout.resolve(self.store.manifest_ref("step12","export"))
        saved = path.with_suffix(".saved")
        path.rename(saved)
        self.addCleanup(lambda:saved.rename(path))
        self.assertEqual(pipeline.run_validation(self.context),1)
        self.assertFalse(self.report()["training_package_accepted"])
        manifest = self.store.validate(self.store.manifest_ref("step13","validate"))
        self.assertEqual(manifest["inputs"],[])
        self.assertFalse((self.store.control/"release.json").exists())

    def test_zip_symlink_is_rejected_before_extraction(self):
        with zipfile.ZipFile(self.archive,"w") as archive:
            for path in self.package.rglob("*"):
                if not path.is_file():
                    continue
                name = "package/"+path.relative_to(self.package).as_posix()
                if path.name == "README.md":
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = 0o120777 << 16
                    archive.writestr(info,path.read_bytes())
                else:
                    archive.write(path,name)
        with self.assertRaisesRegex(ValueError,"Unsafe ZIP"):
            pipeline.relocate_archive(self.source,self.root)

    def test_configuration_requires_current_run_and_explicit_counts(self):
        self.settings["expected_counts"]["studies"] = None
        write_json(self.config,self.settings)
        with self.assertRaisesRegex(PipelineError,"INVALID_VALIDATION_CONFIG"):
            pipeline.run_validation(self.context)
        with self.assertRaises(PipelineError):
            pipeline.run_validation(self.context,"../v1_r2/export/validation.json")

    def test_mean_tolerance_is_one_float32_ulp_and_requires_matching_missingness(self):
        name = "synthetic_lab_mean_24h"
        want = np.float32(1/3)
        expected = {"synthetic":{name:float(want)}}
        one = np.nextafter(want,np.float32(np.inf))
        two = np.nextafter(one,np.float32(np.inf))
        for value,accepted in [(one,True),(two,False),(np.nan,False)]:
            table = pa.table({"sample_id":["synthetic"],name:pa.array([value],type=pa.float32())})
            if accepted:
                step.compare_features(table,expected,[name])
            else:
                with self.assertRaises(ValueError):
                    step.compare_features(table,expected,[name])

    def test_wrong_formula_even_when_cohort_agrees(self):
        rows = self.cohort.to_pylist()
        rows[1]["burden_score"] = 4
        samples = self.samples.to_pylist()
        samples[1]["burden_score"] = 4
        with self.assertRaisesRegex(ValueError, "formula"):
            step.validate_samples(
                pa.Table.from_pylist(samples, schema=step.SAMPLE_SCHEMA),
                pa.Table.from_pylist(rows),
                self.split,
                self.settings,
            )

    def test_secondary_fold_assignment_must_match_its_patient_table(self):
        rotated = self.split.set_column(1, "split", pa.array(["validation", "train", "test"]))
        remaining = self.split.set_column(1, "split", pa.array(["train", "train", "test"]))
        folds = [self.split, rotated, remaining, remaining, remaining]
        samples = contract.build_samples(self.cohort, self.split, fold_patients=folds)
        self.assertEqual(step.validate_samples(samples, self.cohort, self.split, self.settings, folds=folds),
                         self.settings["expected_counts"])
        column = "split_fold_2"
        wrong = samples.set_column(samples.schema.get_field_index(column), column, self.samples["split"])
        with self.assertRaisesRegex(ValueError, "Patient split changed"):
            step.validate_samples(wrong, self.cohort, self.split, self.settings, folds=folds)

    def test_changed_split_key_and_types(self):
        for field, value in [
            ("split", "test"),
            ("sample_id", "wrong"),
            ("image_relpath", "../image.npy"),
        ]:
            rows = self.samples.to_pylist()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                step.validate_samples(
                    pa.Table.from_pylist(rows, schema=step.SAMPLE_SCHEMA),
                    self.cohort,
                    self.split,
                    self.settings,
                )
        with self.assertRaisesRegex(ValueError, "types"):
            step.validate_samples(
                self.samples.set_column(
                    7, "burden_score", self.samples["burden_score"].cast(pa.float64())
                ),
                self.cohort,
                self.split,
                self.settings,
            )

    def test_future_event_time_and_early_discharge(self):
        rows = self.cohort.to_pylist()
        rows[1][step.EVENTS[2] + "_first_event_time"] = self.t0 + timedelta(hours=73)
        with self.assertRaisesRegex(ValueError, "outside"):
            step.validate_samples(
                self.samples, pa.Table.from_pylist(rows), self.split, self.settings
            )
        rows = self.cohort.to_pylist()
        samples = self.samples.to_pylist()
        rows[0].update(
            dischtime=self.t0 + timedelta(hours=2),
            observation_end=self.t0 + timedelta(hours=2),
            observed_hours=2,
            early_discharge=1,
        )
        samples[0].update(
            observation_end=rows[0]["observation_end"],
            observed_hours=2,
            early_discharge=1,
        )
        step.validate_samples(
            pa.Table.from_pylist(samples, schema=step.SAMPLE_SCHEMA),
            pa.Table.from_pylist(rows),
            self.split,
            self.settings,
        )

    def test_inclusive_labs_unavailable_and_numeric_revisions(self):
        self.write_labs(
            [
                self.raw_lab(valuenum="", value="pending"),
                self.raw_lab(
                    lab_record_key="older",
                    labevent_id="2",
                    storetime=str(self.t0 - timedelta(hours=1)),
                    charttime=str(self.t0 - timedelta(hours=2)),
                ),
            ]
        )
        actual = self.ehr.to_pylist()
        actual[0].update(
            {
                f"synthetic_lab_{a}": v
                for a, v in zip(step.AGGREGATIONS, [7, 7, 7, 7, 1, 2], strict=True)
            }
        )
        self.labs(pa.Table.from_pylist(actual, schema=self.ehr.schema))
        self.write_labs(
            [
                self.raw_lab(
                    charttime=str(self.t0 - timedelta(hours=24)),
                    storetime=str(self.t0),
                    valuenum="1e10",
                    value="1e10",
                )
            ]
        )
        actual[0].update(
            {
                f"synthetic_lab_{a}": v
                for a, v in zip(
                    step.AGGREGATIONS, [1e10, 1e10, 1e10, 1e10, 1, 24], strict=True
                )
            }
        )
        self.labs(pa.Table.from_pylist(actual, schema=self.ehr.schema))
        for changes in [
            dict(storetime=str(self.t0 + timedelta(seconds=1))),
            dict(charttime="invalid"),
            dict(storetime=""),
            dict(hadm_id="99999"),
            dict(charttime=str(self.t0 + timedelta(seconds=1))),
            dict(charttime=str(self.t0 - timedelta(hours=24, seconds=1))),
        ]:
            self.write_labs([self.raw_lab(**changes)])
            self.labs()

    def test_new_nonnumeric_revision_prevents_numeric_backfill(self):
        self.write_labs(
            [
                self.raw_lab(
                    storetime=str(self.t0 - timedelta(hours=1)),
                    charttime=str(self.t0 - timedelta(hours=2)),
                ),
                self.raw_lab(
                    lab_record_key="revision",
                    labevent_id="2",
                    charttime=str(self.t0 - timedelta(hours=2)),
                    value="pending",
                    valuenum="",
                ),
            ]
        )
        self.labs()
        with self.assertRaisesRegex(ValueError, "Reconstruction"):
            self.labs(self.with_feature("synthetic_lab_count_24h", 1))

    def test_specimen_units_comparator_and_numeric_disagreement(self):
        for changes in [dict(value="<7"), dict(valueuom=""), dict(valueuom="other")]:
            self.write_labs([self.raw_lab(**changes)])
            self.labs()
        for changes in [
            dict(specimen_id=""),
            dict(specimen_id="invalid"),
            dict(value="8"),
            dict(labevent_id=""),
        ]:
            self.write_labs([self.raw_lab(**changes)])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.labs()

    def test_admission_inference_uses_all_patient_admissions(self):
        self.write_labs([self.raw_lab(hadm_id="")])
        rows = self.static["admissions"].to_pylist()
        rows.append({**rows[0], "hadm_id": "99999"})
        self.labs(admissions=pa.Table.from_pylist(rows))
        rows[-1]["admittime"] = str(self.t0 + timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "Reconstruction"):
            self.labs(admissions=pa.Table.from_pylist(rows))

    def test_conflicting_ids_across_unretained_items(self):
        self.write_labs(
            [self.raw_lab(), self.raw_lab(lab_record_key="conflict", itemid="2")]
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.labs()

    def test_overlapping_windows_legitimately_reuse_record(self):
        rows = self.cohort.to_pylist()
        rows[1].update(
            subject_id=rows[0]["subject_id"],
            hadm_id=rows[0]["hadm_id"],
            t0=self.t0 + timedelta(hours=1),
        )
        self.write_labs([self.raw_lab()])
        actual = self.ehr.to_pylist()
        for i in (0, 1):
            actual[i].update(
                {
                    f"synthetic_lab_{a}": v
                    for a, v in zip(step.AGGREGATIONS, [7, 7, 7, 7, 1, i], strict=True)
                }
            )
        result = step.reconstruct_labs(
            self.lab_path,
            self.static["admissions"],
            pa.Table.from_pylist(rows),
            self.rules,
            pa.Table.from_pylist(actual, schema=self.ehr.schema),
        )
        self.assertEqual(result["retained_associations"], 2)

    def test_contract_rejects_label_leak_and_nonfinite(self):
        with self.assertRaises(ValueError):
            step.validate_ehr_contract(
                self.ehr.append_column("burden_score", self.samples["burden_score"]),
                self.samples,
                self.schema,
                self.rules,
            )
        with self.assertRaises(ValueError):
            step.validate_ehr_contract(
                self.with_feature("age_at_t0", np.inf),
                self.samples,
                self.schema,
                self.rules,
            )
        with self.assertRaises(ValueError):
            step.validate_ehr_contract(
                self.with_feature("sex", ""), self.samples, self.schema, self.rules
            )

    def test_future_services_and_conflicting_latest_service(self):
        services = self.static["services"].to_pylist()
        services.append(
            {
                **services[0],
                "transfertime": str(self.t0 + timedelta(seconds=1)),
                "curr_service": "SURG",
            }
        )
        step.reconstruct_static(
            self.cohort,
            self.cohort,
            self.static["patients"],
            self.static["admissions"],
            pa.Table.from_pylist(services),
            self.ehr,
        )
        services[-1]["transfertime"] = str(self.t0)
        with self.assertRaisesRegex(ValueError, "latest service"):
            step.reconstruct_static(
                self.cohort,
                self.cohort,
                self.static["patients"],
                self.static["admissions"],
                pa.Table.from_pylist(services),
                self.ehr,
            )

    def test_latest_missing_service_does_not_backfill(self):
        services = self.static["services"].to_pylist()
        services[0]["curr_service"] = ""
        actual = self.with_feature("current_service", "__MISSING__")
        step.reconstruct_static(
            self.cohort,
            self.cohort,
            self.static["patients"],
            self.static["admissions"],
            pa.Table.from_pylist(services),
            actual,
        )
        with self.assertRaises(ValueError):
            step.reconstruct_static(
                self.cohort,
                self.cohort,
                self.static["patients"],
                self.static["admissions"],
                pa.Table.from_pylist(services),
                self.ehr,
            )

    def test_selector_and_conversion_rejected(self):
        for changes in [
            dict(conversion_multiplier=np.inf),
            dict(specimen_rule='{"kind":"expression"}'),
            dict(valid_min=0),
            dict(source_unit="", missing_unit_rule=""),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                step.retained_rules(
                    pa.Table.from_pylist([{**self.rules[0], **changes}])
                )
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            step.retained_rules(pa.Table.from_pylist(self.rules * 2))
