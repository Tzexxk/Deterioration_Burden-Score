"""Exercise CatBoost training with a synthetic frozen release and five patient folds."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import create_patient_split as split
from src.cohort.burden_io import COMPONENTS, score
from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.data.package_contract import CV_SAMPLE_SCHEMA, FOLD_SPLIT_COLUMNS, SPLITS
from src.ehr.feature_engine import schema_for
from src.models.experiments import experiment_store
from src.models.train.ehr.catboost.training import run_training

CONFIG = "configs/models/train/ehr/catboost.json"
TEMPLATE = Path(__file__).resolve().parents[5] / "configs/models.template/train/ehr/catboost.json"


class CatBoostTrainingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "pyproject.toml").touch()
        write_json(root / "configs/environment/paths.local.json", {"roots": {}})
        write_json(root / "configs/version_run.json", {"version": 99, "run": 1})
        write_json(root / CONFIG, read_json(TEMPLATE))
        self.layout = Layout(root)
        self.store = ManifestStore(self.layout, data=DataContext(self.layout, 1, 3))

    def publish(self, *, wrong_role=False, wrong_acceptance=False):
        version = "v1_r2"
        schema = schema_for([], version)
        rows, indexed = [], []
        t0 = datetime(2020, 1, 1)
        for i in range(12):
            group = i // 2
            flags = [0, 0, 0, 0, 0, i % 2]
            row = dict(sample_id=f"sample-{i:02d}", subject_id=i, hadm_id=i, study_id=i, dicom_id=f"dicom-{i}",
                       image_relpath=f"images/{i}.npy", t0=t0, observation_end=t0 + timedelta(hours=72),
                       observed_hours=72.0, early_discharge=0, **dict(zip(COMPONENTS, flags, strict=True)), **score(flags))
            for fold, column in enumerate(FOLD_SPLIT_COLUMNS, 1):
                row[column] = "test" if group == 0 else "validation" if group == fold else "train"
            rows.append(row)
            keys = ("sample_id", "subject_id", "hadm_id", "burden_score", "phys_death", "high_burden")
            indexed.append(dict(study_index=i, fold_id=group, **{key: row[key] for key in keys}))
        if wrong_role:
            rows[0][FOLD_SPLIT_COLUMNS[3]] = "train"
        folds = [dict(fold=i, train_indices=[j for j in range(2, 12) if j//2 != i],
                      validation_indices=[i*2, i*2+1]) for i in range(1, 6)]
        cv = dict(index_base=0, n_folds=5, primary_view_fold=1, test_fold_id=0, ordering="sample_id ascending",
                  seed=42, algorithm_version=split.CV_ALGORITHM, allocation_order="holdout_then_development_folds",
                  cohort_version=version, split_version=version, master=indexed, folds=folds,
                  test_indices=[0, 1], development_indices=list(range(2, 12)))
        patients = pa.Table.from_pylist([
            dict(subject_id=i, split="test" if i<2 else "validation" if i<4 else "train", seed=42,
                 split_version=version, cohort_version=version, study_count=1, max_burden_score=row["burden_score"],
                 ever_phys_death=row["phys_death"], ever_high_burden=row["high_burden"])
            for i, row in enumerate(rows)], schema=split.SCHEMA)
        fold_files = [f"splits/fold_{i}/{split.TABLE_NAME}" for i in range(1, 6)]
        with self.store.publish("step12", "export", data_directories=("package",)) as work:
            package = work / "package"
            package.mkdir()
            pq.write_table(pa.Table.from_pylist(rows, schema=CV_SAMPLE_SCHEMA), package / "samples.parquet")
            features = {"sample_id": pa.array([row["sample_id"] for row in reversed(rows)])}
            for column in schema["numeric_features"]:
                features[column] = pa.array(range(11, -1, -1), type=pa.float32())
            for column in schema["categorical_features"]:
                features[column] = pa.array(["__MISSING__"] * 12)
            pq.write_table(pa.table(features), package / "ehr_features.parquet")
            write_json(package / "feature_schema.json", schema)
            write_json(package / split.CV_INDICES, cv)
            for fold, name in enumerate(fold_files, 1):
                path = package / name
                path.parent.mkdir(parents=True)
                pq.write_table(split.select_fold(patients, cv, fold), path)
            counts = {"numeric": len(schema["numeric_features"]), "categorical": len(schema["categorical_features"])}
            files = [path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()]
            write_json(package / "dataset_metadata.json", dict(
                release_id="v1_r3", data_run="v1_r3", cohort_version="v1_r3", parent_release_id=version,
                split_version=version, dictionary_version=version, feature_set_version=version, interface_version="1.0",
                target_column="burden_score", horizon_hours=72, ehr_lookback_hours=24, split_unit="subject_id",
                ehr_preprocessing_applied=False, feature_counts=counts, files=sorted([*files, "dataset_metadata.json"]),
                split_values=list(SPLITS), split_columns=list(FOLD_SPLIT_COLUMNS),
                cross_validation=dict(indices_file=split.CV_INDICES, n_folds=5, split_column_semantics="per_fold",
                                      patient_split_files=fold_files)))
        export = self.store.manifest_ref("step12", "export")
        with self.store.publish("step13", "validate", inputs=[export],
                                details={"checks_passed": True, "training_package_accepted": True}) as work:
            write_json(work / "training_package_acceptance.json", {
                "training_package_accepted": True, "source_export": {"manifest": export, "generation": 1},
                "checks": {
                    "samples": {"evidence": {"studies": 12, "admissions": 12, "patients": 12,
                                             "fixed_test_studies": 3 if wrong_acceptance else 2, "folds_checked": 5}},
                    "ehr_contract": {"evidence": {f"{key}_features": value for key, value in counts.items()}},
                }})

    def test_training_requires_a_frozen_release_and_valid_id(self):
        self.publish()
        with self.assertRaisesRegex(PipelineError, "SOURCE_NOT_RELEASED"):
            run_training(self.layout, CONFIG, "d2_l210")
        with self.assertRaisesRegex(PipelineError, "UNKNOWN_EXPERIMENT_ID"):
            run_training(self.layout, CONFIG, "pending")

    def test_training_checks_step13_before_creating_results(self):
        self.publish(wrong_acceptance=True)
        self.store.freeze(self.store.manifest_ref("step13", "validate"))
        with self.assertRaisesRegex(PipelineError, "INVALID_EHR_PACKAGE"):
            run_training(self.layout, CONFIG, "d2_l210")
        self.assertFalse(self.layout.path("outputs", "models/train/ehr/catboost").exists())

    def test_sample_role_must_agree_with_index(self):
        self.publish(wrong_role=True)
        self.store.freeze(self.store.manifest_ref("step13", "validate"))
        with self.assertRaisesRegex(PipelineError, "INVALID_EHR_FOLDS"):
            run_training(self.layout, CONFIG, "d2_l210")
        self.assertFalse(self.layout.path("outputs", "models/train/ehr/catboost/d2_l210").exists())

    def test_catboost_batch_checkpoints_and_id_reuse(self):
        self.publish()
        acceptance = self.store.manifest_ref("step13", "validate")
        self.store.freeze(acceptance)
        config_file = self.layout.path("project", CONFIG)
        config = read_json(config_file)
        config["defaults"]["model"].update(iterations=3, early_stopping_rounds=1)
        config["experiments"] = config["experiments"][:2]
        config["experiments"][1]["parameters"]["seed"] = 43
        write_json(config_file, config)
        result = run_training(self.layout, CONFIG)
        self.assertEqual(result["status"], "complete", result)
        ids = [entry["id"] for entry in result["experiments"]]
        self.assertEqual(ids, ["d2_l210", "d2_l230"])
        for entry in result["experiments"]:
            store = experiment_store(self.layout, "train", "ehr", "catboost", entry["id"])
            manifest = store.validate(entry["manifest"])
            self.assertEqual(manifest["inputs"], [{"manifest": acceptance, "generation": 1}])
            directory = self.layout.resolve(entry["manifest"]).parent
            self.assertEqual(len(list(directory.glob("fold_*/model.cbm"))), 5)
            summary = read_json(directory / "input_summary.json")
            self.assertEqual((summary["studies"], summary["fixed_test_studies"], summary["split_version"]), (12, 2, "v1_r2"))
            for interface in directory.glob("fold_*/interface.json"):
                rows = read_json(interface)
                used = set(rows["train_sample_ids"] + rows["validation_sample_ids"])
                self.assertNotIn("sample-00", used)
                self.assertNotIn("sample-01", used)
        index = self.layout.path("outputs", "models/train/ehr/catboost/INDEX.md").read_text(encoding="utf-8")
        self.assertIn("seed=43", index)
        self.assertIn("candidate.l2_leaf_reg=30", index)
        self.assertEqual(run_training(self.layout, CONFIG, ids[0])["experiments"][0]["status"], "unchanged")
        config["experiments"][0]["parameters"]["seed"] = 44
        write_json(config_file, config)
        failed = run_training(self.layout, CONFIG)
        self.assertEqual([entry["status"] for entry in failed["experiments"]], ["failed", "unchanged"])
        self.assertEqual(failed["experiments"][0]["error"], "EXPERIMENT_ID_CONFLICT")
