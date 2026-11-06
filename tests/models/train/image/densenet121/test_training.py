"""Check the frozen backbone and standard DenseNet development artifacts."""

import unittest
from pathlib import Path

import torch
from torchvision.models import densenet121

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.image.comparison import run_comparison
from src.models.assessment.image.densenet121.assessment import run_assessment
from src.models.train.image.densenet121.training import run_training
from tests.models.train.ehr.catboost import test_training


class DenseNet121TrainingTests(unittest.TestCase):
    def test_frozen_backbone_publishes_standard_development_results(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

        key = "imagenet1k_frozen"
        templates = Path(__file__).resolve().parents[5] / "configs/models.template"
        train_path = "configs/models/train/image/densenet121.json"
        config = read_json(templates / "train/image/densenet121.json")
        config["data_release"] = "v1_r3"
        config["defaults"].update(device="cpu", batch_size=4, max_epochs=1)
        weight_path = layout.path("project", "local_inputs/densenet121.pth")
        weight_path.parent.mkdir()
        pretrained = densenet121(weights=None).state_dict()
        torch.save(pretrained, weight_path)
        config["experiments"][0]["parameters"]["weights"] = layout.ref("project", "local_inputs/densenet121.pth")
        write_json(layout.path("project", train_path), config)

        result = run_training(layout, train_path, key)["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        for fold in range(1, 6):
            checkpoint = torch.load(folder / f"fold_{fold}/best.pt", map_location="cpu", weights_only=True)
            for name, tensor in pretrained.items():
                if name.startswith("features."):
                    self.assertTrue(torch.equal(checkpoint[name], tensor), name)
            interface = read_json(folder / f"fold_{fold}/interface.json")
            self.assertTrue({"sample-00", "sample-01"}.isdisjoint(
                interface["train_sample_ids"] + interface["validation_sample_ids"]))
        self.assertEqual(run_training(layout, train_path, key)["experiments"][0]["status"], "unchanged")

        assess_path = "configs/models/assessment/image/densenet121.json"
        settings = read_json(templates / "assessment/image/densenet121.json")
        write_json(layout.path("project", assess_path), dict(settings, data_release="v1_r3"))
        result = run_assessment(layout, assess_path, key)["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(read_json(folder / "metrics.json")["pooled_oof"]["studies"], 10)
        self.assertTrue((folder / "training_history.svg").is_file())
        for rule in ("sensitivity", "specificity"):
            self.assertTrue((folder / "threshold_based" / f"{rule}_80/confusion_matrices.svg").is_file())

        result = run_comparison(layout, "paired_densenet", draws=12)
        self.assertEqual(result["status"], "complete", result)
        folder = layout.resolve(result["manifest"]).parent
        metrics = read_json(folder / "threshold_free/metrics.json")
        self.assertEqual(set(metrics), {"baseline_mean", f"densenet121/{key}"})
