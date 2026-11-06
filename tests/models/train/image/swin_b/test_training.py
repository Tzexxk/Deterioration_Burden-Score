"""Verify Swin-B uses the standard image train and assessment contract."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchvision.models.swin_transformer import SwinTransformer

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.image.swin_b.assessment import run_assessment
from src.models.train.image.swin_b.training import run_training
from tests.models.train.ehr.catboost import test_training


def small_swin(*, weights):
    assert weights is None
    return SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[1] * 4,
                           num_heads=[1, 2, 4, 8], window_size=[7, 7])


class SwinBTrainingTests(unittest.TestCase):
    def test_frozen_training_and_checkpoint_replay(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

        weights = small_swin(weights=None).state_dict()
        path = layout.path("project", "local_inputs/swin.pth")
        path.parent.mkdir()
        torch.save(weights, path)
        templates = Path(__file__).resolve().parents[5] / "configs/models.template"
        train = "configs/models/train/image/swin_b.json"
        config = read_json(templates / "train/image/swin_b.json")
        config["data_release"] = "v1_r3"
        config["defaults"].update(device="cpu", max_epochs=1, batch_size=4,
                                  weights=layout.ref("project", "local_inputs/swin.pth"))
        write_json(layout.path("project", train), config)
        assess = "configs/models/assessment/image/swin_b.json"
        config = read_json(templates / "assessment/image/swin_b.json")
        write_json(layout.path("project", assess), dict(config, data_release="v1_r3"))

        with patch("src.models.train.image.swin_b.training.swin_b", small_swin):
            trained = run_training(layout, train)["experiments"][0]
            self.assertEqual(trained["status"], "complete", trained)
            result = run_assessment(layout, assess)["experiments"][0]
            self.assertEqual(result["status"], "complete", result)
        root = layout.resolve(trained["manifest"]).parent
        state = torch.load(root / "fold_1/best.pt", map_location="cpu", weights_only=True)
        for name, value in weights.items():
            if not name.startswith("head."):
                self.assertTrue(torch.equal(state[name], value), name)
        metrics = layout.resolve(result["manifest"]).with_name("metrics.json")
        self.assertEqual(read_json(metrics)["pooled_oof"]["studies"], 10)
