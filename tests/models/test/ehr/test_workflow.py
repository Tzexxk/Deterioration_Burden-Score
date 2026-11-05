"""Verify refit and fixed-test publication for all selected EHR model families."""

import unittest
from importlib import import_module
from statistics import median

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.ehr_data import load_package
from src.models.experiments import experiment_store
from src.models.released_data import ReleasedData
from src.models.test.ehr.comparison import run_comparison
from src.models.test.ehr.workflow import run_test
from tests.models.train.ehr.catboost import test_training


class EHRFixedTestTests(unittest.TestCase):
    def test_selected_models_use_development_mean_and_frozen_thresholds(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(different_test_mean=True)
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        package = load_package(ReleasedData.load(layout, "v1_r3"))
        development = package.samples.iloc[package.cv["development_indices"]]
        test = package.samples.iloc[package.cv["test_indices"]]
        baseline = float(development.burden_score.to_numpy(float).mean())
        self.assertNotEqual(baseline, float(test.burden_score.mean()))

        evaluated = []
        for model, key in (("catboost", "d2_l210"), ("xgboost", "x03"), ("extra_trees", "t02")):
            with self.subTest(model=model):
                config = f"configs/models/test/ehr/{model}.json"
                write_json(layout.path("project", config), {"data_release": "v1_r3", "id": key, "assessment_id": key})
                train_module = import_module(f"src.models.train.ehr.{model}.training")
                assess_module = import_module(f"src.models.assessment.ehr.{model}.assessment")
                adapter = import_module(f"src.models.test.ehr.{model}.model")
                with self.assertRaisesRegex(PipelineError, "REFIT_NOT_COMPLETE"):
                    run_test(layout, config, model, "evaluate", adapter.fit, adapter.predict)

                if model == "catboost":
                    training = read_json(layout.path("project", test_training.CONFIG))
                    training["defaults"]["model"].update(iterations=4, early_stopping_rounds=2)
                else:
                    params = ({"objective": "reg:squarederror", "eval_metric": "rmse", "missing": "NaN", "device": "cpu",
                               "n_jobs": 1, "n_estimators": 4, "early_stopping_rounds": 2} if model == "xgboost"
                              else {"n_estimators": 4, "n_jobs": 1, "random_state": 42})
                    family = "xgb" if model == "xgboost" else "extra"
                    training = {"data_release": "v1_r3", "defaults": {}, "experiments": [
                        {"id": key, "parameters": {"candidate": {"name": key.upper(), "family": family, "params": params}}}]}
                train_path = f"configs/models/train/ehr/{model}.json"
                write_json(layout.path("project", train_path), training)
                trained = train_module.run_training(layout, train_path, key)["experiments"][0]
                self.assertEqual(trained["status"], "complete", trained)
                assess_path = f"configs/models/assessment/ehr/{model}.json"
                write_json(layout.path("project", assess_path), {"data_release": "v1_r3", "defaults": {
                    "sensitivity_target": 0.8, "specificity_target": 0.8, "plot_training_history": False},
                    "experiments": [{"id": key, "parameters": {"train_id": key}}]})
                assessed = assess_module.run_assessment(layout, assess_path, key)["experiments"][0]
                self.assertEqual(assessed["status"], "complete", assessed)

                refit = run_test(layout, config, model, "refit", adapter.fit, adapter.predict)
                refit_dir = layout.resolve(refit["manifest"]).parent
                summary = read_json(refit_dir / "fit_summary.json")
                self.assertEqual(summary["train_sample_ids"], development.sample_id.tolist())
                self.assertEqual(summary["baseline_mean"], baseline)
                self.assertFalse(summary["test_consumed_for_fitting"])
                folds = read_json(layout.resolve(trained["manifest"]).parent / "training_summary.json")["folds"]
                self.assertEqual(summary["refit_rounds"], None if model == "extra_trees" else median(f["best_round"] for f in folds))

                result = run_test(layout, config, model, "evaluate", adapter.fit, adapter.predict)
                evaluated.append(result["manifest"])
                store = experiment_store(layout, "test", "ehr", model, key)
                self.assertEqual(store.validate(result["manifest"])["inputs"], [{"manifest": refit["manifest"], "generation": 1}])
                self.assertEqual(store.validate(refit["manifest"])["inputs"], [{"manifest": assessed["manifest"], "generation": 1}])
                folder = layout.resolve(result["manifest"]).parent
                predictions = pd.read_parquet(folder / "test_predictions.parquet")
                self.assertEqual(predictions.sample_id.tolist(), test.sample_id.tolist())
                self.assertTrue(predictions.baseline_mean_score.eq(baseline).all())
                metrics = read_json(folder / "metrics.json")
                self.assertAlmostEqual(metrics["model"]["raw"]["rmse"],
                                       np.sqrt(np.mean((predictions.burden_score - predictions.predicted_score) ** 2)))
                self.assertIn("postprocessed", metrics["model"])
                points = read_json(folder / "operating_points.json")
                selected = read_json(layout.resolve(assessed["manifest"]).parent / "operating_points.json")
                for rule in ("sensitivity", "specificity"):
                    directory = folder / "threshold_based" / f"{rule}_80"
                    for endpoint, value in points[rule]["model"].items():
                        self.assertEqual(value["threshold"], selected[rule]["model"][endpoint]["threshold"])
                        self.assertEqual(sum(value["confusion_matrix"].values()), len(test))
                    self.assertTrue((directory / "confusion_matrices.svg").is_file())
                again = run_test(layout, config, model, "evaluate", adapter.fit, adapter.predict)
                self.assertEqual(again["status"], "unchanged")

        progress = []
        result = run_comparison(layout, "paired_test", draws=12, progress=progress.append)
        self.assertEqual(result["manifest"]["path"], "models/test/ehr/comparison/paired_test/manifest.json")
        store = experiment_store(layout, "test", "ehr", "comparison", "paired_test")
        manifest = store.validate(result["manifest"])
        self.assertCountEqual(manifest["inputs"], [{"manifest": ref, "generation": 1} for ref in evaluated])
        directory = layout.resolve(result["manifest"]).parent
        self.assertEqual(len(list(directory.rglob("*.svg"))), 18)
        free = read_json(directory / "threshold_free/metrics.json")
        self.assertEqual(set(free), {"baseline_mean", "catboost/d2_l210", "extra_trees/t02", "xgboost/x03"})
        self.assertEqual(free["xgboost/x03"]["rmse"]["valid_draws"], 12)
        self.assertTrue(any(line.startswith("Bootstrap (12 draws): 4/4") for line in progress))
        self.assertTrue(any(line.startswith("Writing SVGs: 18/18") for line in progress))
        self.assertIn("Rechecking test evaluations: 3/3", progress)
        self.assertEqual(run_comparison(layout, "paired_test", draws=12)["status"], "unchanged")
