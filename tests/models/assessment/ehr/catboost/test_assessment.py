"""Check checkpoint reuse, development-only predictions and mean-baseline assessment."""

import shutil
import unittest

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.ehr.catboost.assessment import run_assessment
from src.models.assessment.ehr.comparison import run_comparison
from src.models.experiments import experiment_store
from src.models.train.ehr.catboost.training import run_training
from tests.models.train.ehr.catboost import test_training

CONFIG = "configs/models/assessment/ehr/catboost.json"


class CatBoostAssessmentTests(unittest.TestCase):
    def test_assessment_binds_train_and_publishes_only_development_predictions(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish()
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        train_config = read_json(layout.path("project", test_training.CONFIG))
        train_config["defaults"]["model"].update(iterations=3, early_stopping_rounds=1)
        write_json(layout.path("project", test_training.CONFIG), train_config)
        config = {"data_release": "v1_r3", "defaults": {"sensitivity_target": 0.8, "specificity_target": 0.8, "plot_training_history": True},
                  "experiments": [
                      {"id": "missing", "parameters": {"train_id": "missing"}},
                      {"id": "review", "parameters": {"train_id": "d2_l210"}},
                  ]}
        write_json(layout.path("project", CONFIG), config)
        train = run_training(layout, test_training.CONFIG, "d2_l210")["experiments"][0]
        result = run_assessment(layout, CONFIG)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["experiments"][0]["error"], "TRAINING_NOT_COMPLETE")
        entry = result["experiments"][1]
        self.assertEqual(entry["status"], "complete", result)
        store = experiment_store(layout, "assessment", "ehr", "catboost", "review")
        manifest = store.validate(entry["manifest"])
        self.assertEqual(manifest["inputs"], [{"manifest": train["manifest"], "generation": 1}])
        folder = layout.resolve(entry["manifest"]).parent
        predictions = pd.read_parquet(folder / "oof_predictions.parquet")
        self.assertEqual(predictions.sample_id.tolist(), [f"sample-{i:02d}" for i in range(2, 12)])
        self.assertEqual(predictions.fold.tolist(), [i // 2 for i in range(2, 12)])
        np.testing.assert_array_equal(predictions.baseline_mean_score, np.full(10, 1.5))
        metrics = read_json(folder / "metrics.json")
        self.assertEqual(metrics["pooled_oof"]["baseline_mean"]["raw"]["rmse"], 1.5)
        expected = np.sqrt(np.mean((predictions.predicted_score - predictions.burden_score) ** 2))
        self.assertAlmostEqual(metrics["pooled_oof"]["catboost"]["raw"]["rmse"], expected)
        self.assertEqual([item["train_label_mean"] for item in metrics["folds"]], [1.5] * 5)
        self.assertIn("<svg ", (folder / "training_history.svg").read_text(encoding="utf-8"))
        points = read_json(folder / "operating_points.json")
        self.assertGreaterEqual(points["sensitivity"]["model"]["high_burden"]["metrics"]["sensitivity"], 0.8)
        self.assertGreaterEqual(points["specificity"]["model"]["high_burden"]["metrics"]["specificity"], 0.8)
        for rule in ("sensitivity", "specificity"):
            graph = folder / "threshold_based" / f"{rule}_80" / "confusion_matrices.svg"
            self.assertIn("<svg ", graph.read_text(encoding="utf-8"))
        self.assertEqual(run_assessment(layout, CONFIG, "review")["experiments"][0]["status"], "unchanged")

        ridge_train = experiment_store(layout, "train", "ehr", "ridge", "review")
        release = fixture.store.manifest_ref("step13", "validate")
        with ridge_train.publish("train", "run", inputs=[release]) as work:
            write_json(work / "fixture.json", {"synthetic": True})
        ridge = experiment_store(layout, "assessment", "ehr", "ridge", "review")
        with ridge.publish("assessment", "run", inputs=[ridge_train.manifest_ref("train", "run")],
                           details={"data_release": "v1_r3", "train_id": "review"}) as work:
            for name in ("oof_predictions.parquet", "operating_points.json"):
                shutil.copyfile(folder / name, work / name)
        summary_store = experiment_store(layout, "assessment", "ehr", "comparison", "paired")
        with self.assertRaises(InterruptedError), summary_store.publish("comparison", "run") as work:
            write_json(work / "partial.json", {"temporary": True})
            raise InterruptedError("synthetic publication failure")
        self.assertFalse(layout.path("outputs", summary_store.prefix).exists())

        progress = []
        summary = run_comparison(layout, "paired", draws=12, progress=progress.append)
        self.assertEqual(summary["manifest"]["path"], "models/assessment/ehr/comparison/paired/manifest.json")
        self.assertTrue(any(line.startswith("Validating assessments: 2/2") for line in progress))
        self.assertTrue(any(line.startswith("Bootstrap (12 draws): 3/3") for line in progress))
        self.assertTrue(any(line.startswith("Writing SVGs: 18/18") for line in progress))
        self.assertIn("Rechecking assessments: 2/2", progress)
        summary_manifest = summary_store.validate(summary["manifest"])
        self.assertEqual(summary_manifest["inputs"], [{"manifest": ref, "generation": 1} for ref in (
            entry["manifest"], ridge.manifest_ref("assessment", "run"))])
        out = layout.resolve(summary["manifest"]).parent
        free = read_json(out / "threshold_free/metrics.json")
        self.assertEqual(free["baseline_mean"]["rmse_relative_pct"]["point"], 0)
        self.assertEqual(free["catboost/review"]["phys_death_ap"]["requested_draws"], 12)
        self.assertEqual(free["catboost/review"], free["ridge/review"])
        plots = list(out.rglob("*.svg"))
        self.assertEqual(len(plots), 18)
        for rule, metric in (("sensitivity", "ppv"), ("specificity", "npv")):
            directory = out / "threshold_based" / f"{rule}_80"
            self.assertEqual(len(list(directory.glob("*.svg"))), 6)
            plot = directory / f"high_burden_{metric}.svg"
            content = plot.read_text(encoding="utf-8")
            for label in ("<svg ", "Point estimate [95% CI]", "catboost/review", "ridge/review", "baseline_mean"):
                self.assertIn(label, content)
        self.assertEqual(len(list((out / "threshold_free").glob("*.svg"))), 6)
        self.assertEqual(run_comparison(layout, "paired", draws=12)["status"], "unchanged")
        config["experiments"][1]["parameters"]["sensitivity_target"] = 0.9
        write_json(layout.path("project", CONFIG), config)
        conflict = run_assessment(layout, CONFIG, "review")["experiments"][0]
        self.assertEqual(conflict["error"], "EXPERIMENT_ID_CONFLICT")
