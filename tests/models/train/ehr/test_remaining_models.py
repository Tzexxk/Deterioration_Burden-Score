"""Exercise the six EHR workflows against a small frozen development package."""

import unittest
from importlib import import_module
from pathlib import Path

import pandas as pd

from src.common.manifest import write_json
from src.common.paths import read_json
from tests.models.train.ehr.catboost import test_training

ROOT = Path(__file__).resolve().parents[4]
MODELS = ("random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "mlp_sklearn", "mlp_torch")


class EHRModelWorkflowTests(unittest.TestCase):
    def test_saved_models_reproduce_fold_predictions_for_assessment(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish()
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        for model in MODELS:
            with self.subTest(model=model):
                train = f"configs/models/train/ehr/{model}.json"
                assessment = f"configs/models/assessment/ehr/{model}.json"
                train_cfg, assess_cfg = (
                    read_json(ROOT / name.replace("configs/models/", "configs/models.template/"))
                    for name in (train, assessment)
                )
                candidate = train_cfg["experiments"][0]
                key = candidate["id"]
                if model in {"random_forest", "extra_trees"}:
                    candidate["parameters"]["candidate"]["params"]["n_estimators"] = 3
                if model == "xgboost":
                    candidate["parameters"]["candidate"]["params"].update(n_estimators=8, early_stopping_rounds=2)
                if model == "mlp_torch":
                    train_cfg["defaults"]["torch_training"].update(max_epochs=5, minimum_epochs=3, patience=2)
                write_json(layout.path("project", train), train_cfg)
                write_json(layout.path("project", assessment), assess_cfg)
                train_model = import_module(f"src.models.train.ehr.{model}.training")
                assess_model = import_module(f"src.models.assessment.ehr.{model}.assessment")
                fitted = train_model.run_training(layout, train, key)["experiments"][0]
                self.assertEqual(fitted["status"], "complete", fitted)
                result = assess_model.run_assessment(layout, assessment, key)["experiments"][0]
                self.assertEqual(result["status"], "complete", result)
                folder = layout.resolve(result["manifest"]).parent
                self.assertEqual(read_json(folder / "manifest.json")["inputs"],
                                 [{"manifest": fitted["manifest"], "generation": 1}])
                scores = pd.read_parquet(folder / "oof_predictions.parquet")
                self.assertEqual(len(scores), 10)
                self.assertTrue(scores.sample_id.is_unique)
                self.assertEqual(set(read_json(folder / "operating_points.json")), {"sensitivity", "specificity"})
                if model in {"xgboost", "mlp_torch"}:
                    self.assertTrue((folder / "training_history.svg").is_file())
