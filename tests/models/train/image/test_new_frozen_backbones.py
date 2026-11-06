"""Check local ImageNet backbones against the shared image workflow."""

import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch
from torchvision.models import efficientnet_b0

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.image.caformer_s18 import assessment as caformer_assessment
from src.models.assessment.image.efficientnet_b0 import assessment as efficientnet_assessment
from src.models.train.image.caformer_s18 import training as caformer
from src.models.train.image.convformer_s18 import training as convformer
from src.models.train.image.deit_base_patch16_224 import training as deit
from src.models.train.image.efficientnet_b0 import training as efficientnet
from tests.models.train.ehr.catboost import test_training

TEMPLATES = Path(__file__).resolve().parents[4] / "configs/models.template"
TIMM_READY = find_spec("timm") is not None and find_spec("safetensors") is not None


class FrozenImageBackboneTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def check_publication(self, train, assessment, weights, filename, save, head_prefix):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        weights_file = layout.path("project", f"local_inputs/{filename}")
        weights_file.parent.mkdir()
        save(weights, weights_file)

        name = f"image/{train.MODEL}.json"
        config = read_json(TEMPLATES / "train" / name)
        config["data_release"] = "v1_r3"
        config["defaults"].update(device="cpu", max_epochs=1, batch_size=4,
                                  weights=layout.ref("project", f"local_inputs/{filename}"))
        train_path = f"configs/models/train/{name}"
        write_json(layout.path("project", train_path), config)
        trained = train.run_training(layout, train_path)["experiments"][0]
        self.assertEqual(trained["status"], "complete", trained)
        folder = layout.resolve(trained["manifest"]).parent
        state = torch.load(folder / "fold_1/best.pt", map_location="cpu", weights_only=True)
        for key, tensor in state.items():
            if not key.startswith(head_prefix):
                original_key = key.removeprefix("encoder.")
                self.assertTrue(torch.equal(tensor, weights[original_key]), key)

        config = read_json(TEMPLATES / "assessment" / name)
        assess_path = f"configs/models/assessment/{name}"
        write_json(layout.path("project", assess_path), dict(config, data_release="v1_r3"))
        assessed = assessment.run_assessment(layout, assess_path)["experiments"][0]
        self.assertEqual(assessed["status"], "complete", assessed)
        metrics = layout.resolve(assessed["manifest"]).with_name("metrics.json")
        self.assertEqual(read_json(metrics)["pooled_oof"]["studies"], 10)

    def test_efficientnet_checkpoint_replay(self):
        self.check_publication(efficientnet, efficientnet_assessment, efficientnet_b0(weights=None).state_dict(),
                               "efficientnet.pth", torch.save, "classifier.")

    @unittest.skipUnless(TIMM_READY, "Optional timm runtime is not installed")
    def test_timm_checkpoint_replay(self):
        from safetensors.torch import save_file
        from timm.models.vision_transformer import VisionTransformer

        def small_encoder(_name, *, pretrained):
            self.assertFalse(pretrained)
            return VisionTransformer(img_size=32, patch_size=16, embed_dim=24, depth=1, num_heads=3,
                                     dynamic_img_size=True, dynamic_img_pad=True)

        with patch("timm.create_model", small_encoder):
            weights = small_encoder("synthetic", pretrained=False).state_dict()
            self.check_publication(caformer, caformer_assessment, weights, "model.safetensors",
                                   lambda state, path: save_file(state, str(path)), "head.")

    @unittest.skipUnless(TIMM_READY, "Optional timm runtime is not installed")
    def test_timm_backbones_use_one_linear_head(self):
        for adapter in (caformer, convformer, deit):
            with self.subTest(model=adapter.MODEL), torch.inference_mode():
                model = adapter.build_model({}).eval()
                self.assertIsInstance(model.head, torch.nn.Linear)
                image = torch.zeros(1, 3, 224, 224)
                self.assertEqual(tuple(model(image).shape), (1, 1))
                if adapter is deit:
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 40, 36)).shape), (1, 1))
                del model
