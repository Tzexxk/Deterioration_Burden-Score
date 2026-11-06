"""Verify that Tweedie publishes reusable log-link checkpoints and OOF predictions."""

import unittest
from pathlib import Path

import pandas as pd

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.ehr.tweedie.assessment import run_assessment
from src.models.train.ehr.tweedie.training import run_training
from tests.models.train.ehr.catboost import test_training

ROOT = Path(__file__).resolve().parents[5]
TRAIN = "configs/models/train/ehr/tweedie.json"
ASSESS = "configs/models/assessment/ehr/tweedie.json"


class TweedieWorkflowTests(unittest.TestCase):
    def test_assessment_loads_log_link_checkpoints(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish()
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        for name in (TRAIN, ASSESS):
            value = read_json(ROOT / name.replace("configs/models/", "configs/models.template/"))
            value["data_release"] = "v1_r3"
            write_json(layout.path("project", name), value)
        trained = run_training(layout, TRAIN, "w01")["experiments"][0]
        self.assertEqual(trained["status"], "complete", trained)
        training_folder = layout.resolve(trained["manifest"]).parent
        for fold in range(1, 6):
            self.assertEqual(read_json(training_folder / f"fold_{fold}/model.json")["link"], "log")

        result = run_assessment(layout, ASSESS, "w01")["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(read_json(folder / "manifest.json")["inputs"], [{"manifest": trained["manifest"], "generation": 1}])
        predictions = pd.read_parquet(folder / "oof_predictions.parquet")
        self.assertEqual(len(predictions), 10)
        self.assertTrue(predictions.predicted_score.gt(0).all())
        self.assertEqual(set(read_json(folder / "operating_points.json")), {"sensitivity", "specificity"})
