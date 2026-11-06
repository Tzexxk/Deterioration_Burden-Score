"""Verify both ResNet50 checkpoint formats use the image assessment workflow."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.image.resnet50.assessment import run_assessment
from src.models.train.image.resnet50 import training
from tests.models.train.ehr.catboost import test_training


def small_resnet(*, weights):
    assert weights is None
    return ResNet(BasicBlock, [1, 1, 1, 1])


class ResNet50TrainingTests(unittest.TestCase):
    def test_imagenet_and_xray_weights_train_and_replay(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

        templates = Path(__file__).resolve().parents[5] / "configs/models.template"
        train_path = "configs/models/train/image/resnet50.json"
        assessment_path = "configs/models/assessment/image/resnet50.json"
        train = read_json(templates / "train/image/resnet50.json")
        assessment = read_json(templates / "assessment/image/resnet50.json")
        train["data_release"] = assessment["data_release"] = "v1_r3"
        train["defaults"].update(device="cpu", max_epochs=1, batch_size=4)

        local = layout.path("project", "local_inputs")
        local.mkdir()
        imagenet = small_resnet(weights=None)
        torch.save(imagenet.state_dict(), local / "imagenet.pth")
        train["defaults"]["weights"] = layout.ref("project", "local_inputs/imagenet.pth")
        xray = small_resnet(weights=None)
        xray.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        xray.fc = nn.Linear(xray.fc.in_features, 18)
        torch.save(xray.state_dict(), local / "xray.pt")
        train["experiments"].append({
            "id": "xrv_all_frozen",
            "parameters": {"pretraining": training.XRV, "weights": layout.ref("project", "local_inputs/xray.pt")},
        })
        assessment["experiments"].append({"id": "xrv_all_frozen", "parameters": {"train_id": "xrv_all_frozen"}})
        write_json(layout.path("project", train_path), train)
        write_json(layout.path("project", assessment_path), assessment)

        with patch.object(training, "resnet50", small_resnet):
            result = training.run_training(layout, train_path)
            self.assertEqual(result["status"], "complete", result)
            evaluated = run_assessment(layout, assessment_path)
            self.assertEqual(evaluated["status"], "complete", evaluated)
        for trained, assessed, pretrained in zip(result["experiments"], evaluated["experiments"], (imagenet, xray),
                                                  strict=True):
            folder = layout.resolve(trained["manifest"]).parent
            state = torch.load(folder / "fold_1/best.pt", map_location="cpu", weights_only=True)
            for name, tensor in pretrained.state_dict().items():
                if not name.startswith("fc."):
                    self.assertTrue(torch.equal(state[name], tensor), name)
            self.assertEqual(state["fc.weight"].shape[0], 1)
            metrics = layout.resolve(assessed["manifest"]).with_name("metrics.json")
            self.assertEqual(read_json(metrics)["pooled_oof"]["studies"], 10)
