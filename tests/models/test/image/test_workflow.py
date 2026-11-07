"""Verify that image refit and evaluation use their respective frozen splits."""

import unittest
from statistics import median
from unittest.mock import patch

import pandas as pd
import torch

from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.assessment.image.resnet50.assessment import run_assessment
from src.models.experiments import experiment_store
from src.models.image_data import ImagePackage
from src.models.test.image.comparison import run_comparison
from src.models.test.image.resnet50.workflow import run_test
from src.models.train.image.resnet50 import training
from tests.models.train.ehr.catboost import test_training
from tests.models.train.image.resnet50.test_training import small_resnet


class ImageFixedTestTests(unittest.TestCase):
    def test_refit_uses_development_pixels_and_evaluate_uses_frozen_thresholds(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36), different_test_mean=True)
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

        weights = layout.path("project", "imagenet.pth")
        torch.save(small_resnet(weights=None).state_dict(), weights)
        train_path = "configs/models/train/image/resnet50.json"
        assessment_path = "configs/models/assessment/image/resnet50.json"
        variants = {"b1": {"head_only_epochs": 1, "layer4_lr": 3e-5}, "c5": {"event_components": True},
                    "combo": {"event_components": True, "head_only_epochs": 1, "layer4_lr": 3e-5}}
        write_json(layout.path("project", train_path), {"data_release": "v1_r3", "defaults": {
            "image_source": {"kind": "package"}, "weights": layout.ref("project", "imagenet.pth"),
            "pretraining": training.IMAGENET, "device": "cpu", "seed": 42, "num_workers": 0,
            "batch_size": 4, "max_epochs": 2, "head_lr": 1e-3, "weight_decay": 1e-4},
            "experiments": [{"id": key, "parameters": params} for key, params in variants.items()]})
        write_json(layout.path("project", assessment_path), {"data_release": "v1_r3", "defaults": {
            "sensitivity_target": 0.8, "specificity_target": 0.8, "plot_training_history": False},
            "experiments": [{"id": key, "parameters": {"train_id": key}} for key in variants]})
        with patch.object(training, "resnet50", small_resnet):
            trained = training.run_training(layout, train_path)
            self.assertEqual(trained["status"], "complete", trained)
            assessed = run_assessment(layout, assessment_path)
            self.assertEqual(assessed["status"], "complete", assessed)
            for train, assessment in zip(trained["experiments"], assessed["experiments"], strict=True):
                key = train["id"]
                config_path = f"configs/models/test/image/resnet50_{key}.json"
                write_json(layout.path("project", config_path), {"id": key, "data_release": "v1_r3", "assessment_id": key})
                with self.assertRaisesRegex(PipelineError, "REFIT_NOT_COMPLETE"):
                    run_test(layout, config_path, "evaluate")
                read_image = ImagePackage.read_image
                calls = []

                def checked_read(package, index):
                    calls.append(index)
                    return read_image(package, index)

                with patch.object(ImagePackage, "read_image", checked_read):
                    refit = run_test(layout, config_path, "refit")
                    self.assertEqual(set(calls), set(range(2, 12)))
                    calls.clear()
                    evaluated = run_test(layout, config_path, "evaluate")
                    self.assertCountEqual(calls, [0, 1])
                store = experiment_store(layout, "test", "image", "resnet50", key)
                self.assertEqual(store.validate(refit["manifest"])["inputs"],
                                 [{"manifest": assessment["manifest"], "generation": 1}])
                self.assertEqual(store.validate(evaluated["manifest"])["inputs"],
                                 [{"manifest": refit["manifest"], "generation": 1}])
                summary = read_json(layout.resolve(refit["manifest"]).with_name("fit_summary.json"))
                folds = read_json(layout.resolve(train["manifest"]).with_name("training_summary.json"))["folds"]
                self.assertEqual(summary["refit_rounds"], median(f["best_epoch"] for f in folds))
                self.assertFalse(summary["test_consumed_for_fitting"])
                folder = layout.resolve(evaluated["manifest"]).parent
                predictions = pd.read_parquet(folder / "test_predictions.parquet")
                self.assertEqual(predictions.sample_id.tolist(), ["sample-00", "sample-01"])
                self.assertTrue(predictions.baseline_mean_score.eq(1.5).all())
                points = read_json(folder / "operating_points.json")
                selected = read_json(layout.resolve(assessment["manifest"]).with_name("operating_points.json"))
                for rule in ("sensitivity", "specificity"):
                    for endpoint, point in points[rule]["model"].items():
                        self.assertEqual(point["threshold"], selected[rule]["model"][endpoint]["threshold"])
                    self.assertTrue((folder / "threshold_based" / f"{rule}_80/confusion_matrices.svg").is_file())
                self.assertEqual(run_test(layout, config_path, "evaluate")["status"], "unchanged")

        result = run_comparison(layout, "paired_test", draws=12)
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(result["manifest"]["path"], "models/test/image/comparison/paired_test/manifest.json")
        self.assertEqual(len(list(folder.rglob("*.svg"))), 18)
        self.assertEqual(set(read_json(folder / "threshold_free/metrics.json")),
                         {"baseline_mean", "resnet50/b1", "resnet50/c5", "resnet50/combo"})
