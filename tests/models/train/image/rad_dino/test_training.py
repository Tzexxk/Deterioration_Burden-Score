"""Verify RAD-DINO geometry and standard checkpoint replay."""

import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.image.rad_dino.assessment import run_assessment
from src.models.train.image.rad_dino import training
from tests.models.train.ehr.catboost import test_training


class RadDinoTrainingTests(unittest.TestCase):
    def test_preprocessing_preserves_input_and_pads_to_patch_grid(self):
        gray = torch.rand(1, 518, 518)
        prepared = training.rad_dino_input(gray)
        self.assertEqual(tuple(prepared.shape), (3, 518, 518))
        torch.testing.assert_close(prepared * training.STD[0] + training.MEAN[0], gray.expand_as(prepared))
        gray = torch.rand(1, 40, 36)
        prepared = training.rad_dino_input(gray)
        self.assertEqual(tuple(prepared.shape), (3, 42, 42))
        original = prepared[:, 1:-1, 3:-3] * training.STD[0] + training.MEAN[0]
        torch.testing.assert_close(original, gray.expand_as(original))

    @unittest.skipUnless(find_spec("transformers") and find_spec("safetensors"),
                         "RAD-DINO optional runtime is not installed")
    def test_local_weights_token_grid_and_checkpoint_replay(self):
        from safetensors.torch import save_file
        from transformers import Dinov2Config, Dinov2Model

        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

        architecture = Dinov2Config(image_size=42, patch_size=14, hidden_size=32, num_hidden_layers=1,
                                    num_attention_heads=4, attn_implementation="sdpa")
        path = layout.path("project", "local_inputs/rad-dino/model.safetensors")
        path.parent.mkdir(parents=True)
        save_file(Dinov2Model(architecture).state_dict(), str(path))
        templates = Path(__file__).resolve().parents[5] / "configs/models.template"
        train = "configs/models/train/image/rad_dino.json"
        config = read_json(templates / "train/image/rad_dino.json")
        config["data_release"] = "v1_r3"
        config["defaults"].update(device="cpu", max_epochs=1, batch_size=4,
                                  weights=layout.ref("project", "local_inputs/rad-dino/model.safetensors"))
        write_json(layout.path("project", train), config)
        assess = "configs/models/assessment/image/rad_dino.json"
        config = read_json(templates / "assessment/image/rad_dino.json")
        write_json(layout.path("project", assess), dict(config, data_release="v1_r3"))

        with patch.object(training, "backbone_config", return_value=architecture):
            with torch.inference_mode():
                global_features, patches = training.build_model({}).eval().encode(torch.zeros(2, 3, 42, 42))
            self.assertEqual(tuple(global_features.shape), (2, 32))
            self.assertEqual(tuple(patches.shape), (2, 32, 3, 3))
            trained = training.run_training(layout, train)["experiments"][0]
            self.assertEqual(trained["status"], "complete", trained)
            result = run_assessment(layout, assess)["experiments"][0]
            self.assertEqual(result["status"], "complete", result)
        metrics = layout.resolve(result["manifest"]).with_name("metrics.json")
        self.assertEqual(read_json(metrics)["pooled_oof"]["studies"], 10)
