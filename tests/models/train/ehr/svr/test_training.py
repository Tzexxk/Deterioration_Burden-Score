"""Verify SVR fold checkpoints can be used for OOF assessment."""

import unittest
from pathlib import Path

import pandas as pd

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.ehr.svr.assessment import run_assessment
from src.models.train.ehr.svr.training import run_training
from tests.models.train.ehr.catboost import test_training

ROOT = Path(__file__).resolve().parents[5]
TRAIN = "configs/models/train/ehr/svr.json"
ASSESS = "configs/models/assessment/ehr/svr.json"


class SVRWorkflowTests(unittest.TestCase):
    def test_fold_effective_gamma_and_oof_coverage(self):
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
        trained = run_training(layout, TRAIN, "s01")["experiments"][0]
        self.assertEqual(trained["status"], "complete", trained)
        folder = layout.resolve(trained["manifest"]).parent
        for fold in range(1, 6):
            checkpoint = read_json(folder / f"fold_{fold}/model.json")
            state = checkpoint["preprocessing"]
            width = 2 * len(state["numeric"]) + sum(map(len, state["categories"].values()))
            self.assertEqual(checkpoint["gamma"], 1 / width)
            support = checkpoint["support_indices"]
            self.assertTrue(all(0 <= index < 8 for index in support))
            self.assertEqual(len(checkpoint["dual_coefficients"]), len(support))

        result = run_assessment(layout, ASSESS, "s01")["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(read_json(folder / "manifest.json")["inputs"], [{"manifest": trained["manifest"], "generation": 1}])
        predictions = pd.read_parquet(folder / "oof_predictions.parquet")
        self.assertEqual(len(predictions), 10)
        self.assertTrue(predictions.sample_id.is_unique)
        self.assertEqual(set(read_json(folder / "operating_points.json")), {"sensitivity", "specificity"})
