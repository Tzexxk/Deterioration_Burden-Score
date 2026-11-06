"""Exercise the Ridge checkpoint and OOF flow on generated data."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.ehr.ridge.assessment import run_assessment
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.train.ehr.ridge.training import run_training
from tests.models.train.ehr.catboost import test_training

ROOT = Path(__file__).resolve().parents[5]


class RidgeWorkflowTests(unittest.TestCase):
    def test_portable_transform_matches_sklearn_with_missing_and_unseen_values(self):
        train = pd.DataFrame({"a": np.array([1, np.nan, 4], dtype=np.float32),
                              "empty": np.full(3, np.nan, dtype=np.float32), "cat": ["a", "b", "a"]})
        validation = pd.DataFrame({"a": np.array([2, np.nan], dtype=np.float32),
                                   "empty": np.array([np.nan, 2], dtype=np.float32), "cat": ["b", "new"]})
        numeric = ["a", "empty"]
        categorical = ["cat"]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler().fit(imputer.fit_transform(train[numeric]))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[categorical])
        expected = np.concatenate([scaler.transform(imputer.transform(validation[numeric])),
                                   validation[numeric].isna().to_numpy(float), encoder.transform(validation[categorical])], axis=1)
        state = fit_preprocessing(train, {"numeric_features": numeric, "categorical_features": categorical})
        np.testing.assert_array_equal(transform_features(validation, state), expected)

    def test_ridge_round_trip_and_assessment(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish()
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        train_config, assess_config = ("configs/models/train/ehr/ridge.json", "configs/models/assessment/ehr/ridge.json")
        for config in (train_config, assess_config):
            value = read_json(ROOT / config.replace("configs/models/", "configs/models.template/"))
            value["data_release"] = "v1_r3"
            write_json(layout.path("project", config), value)
        trained = run_training(layout, train_config, "r01")["experiments"][0]
        self.assertEqual(trained["status"], "complete", trained)
        result = run_assessment(layout, assess_config, "r01")["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(read_json(folder / "manifest.json")["inputs"], [{"manifest": trained["manifest"], "generation": 1}])
        predictions = pd.read_parquet(folder / "oof_predictions.parquet")
        self.assertEqual(predictions.sample_id.tolist(), [f"sample-{i:02d}" for i in range(2, 12)])
        self.assertTrue(np.isfinite(predictions.predicted_score).all())
        self.assertIn("ridge", read_json(folder / "metrics.json")["pooled_oof"])
        self.assertFalse((folder / "training_history.svg").exists())
        for rule in ("sensitivity", "specificity"):
            self.assertTrue((folder / "threshold_based" / f"{rule}_80" / "confusion_matrices.svg").is_file())
