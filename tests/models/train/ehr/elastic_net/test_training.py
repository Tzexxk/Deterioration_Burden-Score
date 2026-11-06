"""Check Elastic Net publication, checkpoint reuse and convergence failures."""

import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.ehr.elastic_net.assessment import run_assessment
from src.models.train.ehr.elastic_net.training import run_training
from tests.models.train.ehr.catboost import test_training

ROOT = Path(__file__).resolve().parents[5]
TRAIN = "configs/models/train/ehr/elastic_net.json"
ASSESS = "configs/models/assessment/ehr/elastic_net.json"


class ElasticNetWorkflowTests(unittest.TestCase):
    def test_nonconvergent_candidate_does_not_publish_and_can_retry(self):
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
        with patch("src.models.train.ehr.elastic_net.training.ElasticNet.fit",
                   side_effect=lambda *args: warnings.warn("fixture", ConvergenceWarning)):
            failed = run_training(layout, TRAIN, "e01")["experiments"][0]
        self.assertEqual(failed["error"], "OPTIMIZER_DID_NOT_CONVERGE")
        self.assertFalse(layout.path("outputs", "models/train/ehr/elastic_net/e01/manifest.json").exists())
        trained = run_training(layout, TRAIN, "e01")["experiments"][0]
        self.assertEqual(trained["status"], "complete", trained)
        result = run_assessment(layout, ASSESS, "e01")["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(read_json(folder / "manifest.json")["inputs"], [{"manifest": trained["manifest"], "generation": 1}])
        self.assertEqual(len(pd.read_parquet(folder / "oof_predictions.parquet")), 10)
        self.assertIn("elastic_net", read_json(folder / "metrics.json")["pooled_oof"])
        self.assertEqual(set(read_json(folder / "operating_points.json")), {"sensitivity", "specificity"})
