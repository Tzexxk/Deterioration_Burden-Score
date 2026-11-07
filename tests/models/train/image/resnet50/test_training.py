"""Verify ResNet50 training variants use the image assessment workflow."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.image.resnet50.assessment import run_assessment
from src.models.assessment.image.resnet50.calibration import fit_affine
from src.models.train.image import training as image_training
from src.models.train.image.resnet50 import training
from tests.models.train.ehr.catboost import test_training

LDS = {"bandwidth": 1, "power": 0.125, "max_ratio": 2}
ORDINAL_THRESHOLDS = [1, 2, 3, 5, 8]


def small_resnet(*, weights):
    assert weights is None
    return ResNet(BasicBlock, [1, 1, 1, 1])


class ResNet50TrainingTests(unittest.TestCase):
    def test_calibration_fits_positive_slope_and_falls_back_on_reverse_order(self):
        scores = torch.tensor([0., 1., 2.]).numpy()
        self.assertEqual(fit_affine(scores, 1 + 2 * scores),
                         {"intercept": 1., "slope": 2., "identity_fallback": False})
        self.assertEqual(fit_affine(scores, -scores),
                         {"intercept": 0., "slope": 1., "identity_fallback": True})

    def test_mixed_pool_preserves_channels_and_blends_spatial_summaries(self):
        features = torch.tensor([[[[0., 0.], [0., 4.]], [[2., 2.], [2., 2.]]]])
        torch.testing.assert_close(training.MixedGlobalPool(0.25)(features),
                                   torch.tensor([[[[1.75]], [[2.]]]]))

    def test_weighted_mse_doubles_high_score_gradients(self):
        errors = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        labels = torch.tensor([2.0, 3.0, 4.0])
        training.weighted_mse(errors, labels, threshold=3, weight=2).backward()
        torch.testing.assert_close(errors.grad, torch.tensor([2.0, 4.0, 4.0]) / 3)

    def test_lds_weights_smooth_cap_and_normalize(self):
        labels = torch.tensor([0] * 423 + [1] * 17 + [2] * 2 + [3] * 14 + [5] * 7 + [8])
        weights = training.lds_weights(labels, **LDS)
        self.assertAlmostEqual(weights[labels].mean().item(), 1.0, places=6)
        ratios = weights[[0, 1, 2, 3, 5, 8]] / weights[0]
        torch.testing.assert_close(ratios, torch.tensor([1.0, 1.058, 1.239, 1.442, 1.625, 2.0]),
                                   atol=0.001, rtol=0)

    def test_ordinal_head_orders_probabilities_and_decodes_grouped_scores(self):
        head = training.OrdinalHead(4, ORDINAL_THRESHOLDS)
        probabilities = head(torch.randn(3, 4)).sigmoid()
        self.assertTrue(bool((probabilities[:, :-1] >= probabilities[:, 1:]).all()))
        scores = torch.tensor([0, *ORDINAL_THRESHOLDS], dtype=torch.float32).unsqueeze(1)
        logits = torch.where(scores >= head.thresholds, torch.inf, -torch.inf)
        torch.testing.assert_close(head.score(logits), scores)

    def test_event_head_combines_probabilities_with_burden_weights(self):
        head = training.EventHead(4)
        logits = torch.where(torch.eye(4, dtype=torch.bool), torch.inf, -torch.inf)
        torch.testing.assert_close(head.score(logits).flatten(), torch.tensor([1., 2., 2., 3.]))

    def test_stage_unfreezes_only_layer4_and_keeps_batchnorm_in_eval(self):
        model = small_resnet(weights=None).eval().requires_grad_(False)
        model.fc.requires_grad_(True)
        for epoch, expected in ((5, {"fc"}), (6, {"fc", "layer4"})):
            training.set_stage(model, epoch, head_only_epochs=5)
            trainable = {name.split(".")[0] for name, value in model.named_parameters() if value.requires_grad}
            self.assertEqual(trainable, expected)
            self.assertFalse(model.layer4[0].bn1.training)

    def test_layer4_bn_updates_only_in_fine_tuning_batches(self):
        model = small_resnet(weights=None).eval()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        images = torch.zeros(2, 3, 32, 32)
        early_bn, late_bn = model.layer3[0].bn1, model.layer4[0].bn1
        with torch.no_grad():
            for epoch, updates in ((5, 0), (6, 1), (7, 2)):
                training.set_stage(model, epoch, head_only_epochs=5, update_layer4_bn=True)
                model(images)
                self.assertEqual(late_bn.num_batches_tracked.item(), updates)
                model.eval()(images)
                self.assertEqual(late_bn.num_batches_tracked.item(), updates)
                self.assertEqual(early_bn.num_batches_tracked.item(), 0)

    def test_frozen_early_stopping_waits_for_minimum_and_keeps_best_checkpoint(self):
        package = SimpleNamespace(samples=pd.DataFrame({"sample_id": ["train", "valid"], "burden_score": [0, 0]}))
        fold = {"fold": 1, "train_indices": [0], "validation_indices": [1]}
        config = {"seed": 42, "batch_size": 1, "num_workers": 0, "max_epochs": 7,
                  "early_stopping": {"min_epochs": 3, "patience": 1}}
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        validation = iter([2., 1., 1., 1., 1.])

        def epoch(_model, loader, _device, optimizer=None, **kwargs):
            rmse = 2. if optimizer is not None else next(validation)
            return {"rmse": rmse}, loader.dataset.ids, [rmse]

        with TemporaryDirectory() as temp, patch.object(image_training, "run_epoch", side_effect=epoch):
            result = image_training.fit_epochs(package, fold, config, model, optimizer, Path(temp) / "fold_1",
                                               label="early_stop", device=torch.device("cpu"))
        self.assertEqual(result, {"fold": 1, "best_epoch": 2, "best_rmse": 1., "epochs": 4})

    def test_frozen_weighted_and_fine_tuned_checkpoints_train_and_replay(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36), different_test_mean=True)
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
        for key, parameters in (
            ("c1", {"high_score_weight": {"threshold": 3, "weight": 2}}),
            ("b1", {"max_epochs": 2, "head_only_epochs": 1, "layer4_lr": 3e-5}),
            ("b2", {"max_epochs": 2, "head_only_epochs": 1, "layer4_lr": 1e-5}),
            ("c2", {"lds": LDS}),
            ("b4", {"mlp_hidden_size": 128}),
            ("c3", {"ordinal_thresholds": ORDINAL_THRESHOLDS}),
            ("b3", {"mean_output_init": True}),
            ("c4", {"event_components": True}),
            ("b5", {"max_epochs": 2, "head_only_epochs": 1, "layer4_lr": 3e-5, "update_layer4_bn": True}),
            ("d", {"max_pool_weight": 0.25}),
            ("a", {"train_affine": {"probability": 0.5, "degrees": 3, "translate_fraction": 0.015}}),
        ):
            train["experiments"].append({"id": key, "parameters": parameters})
            assessment["experiments"].append({"id": key, "parameters": {"train_id": key}})
        extensions = {"b4_long": "b4", "c3_long": "c3", "c5": "c4", "xrv_long": "xrv_all_frozen"}
        for key, source in extensions.items():
            parameters = next(item["parameters"] for item in train["experiments"] if item["id"] == source)
            train["experiments"].append({"id": key, "parameters": {
                **parameters, "max_epochs": 2, "early_stopping": {"min_epochs": 1, "patience": 1}}})
            assessment["experiments"].append({"id": key, "parameters": {"train_id": key}})
        combinations = {"event_layer4": ("c5", {"head_only_epochs": 1, "layer4_lr": 3e-5}),
                        "ordinal_mean": ("c3_long", {"mean_output_init": True})}
        for key, (source, options) in combinations.items():
            parameters = next(item["parameters"] for item in train["experiments"] if item["id"] == source)
            train["experiments"].append({"id": key, "parameters": {**parameters, **options}})
            assessment["experiments"].append({"id": key, "parameters": {"train_id": key}})
        write_json(layout.path("project", train_path), train)
        write_json(layout.path("project", assessment_path), assessment)

        fit_frozen_head = training.fit_frozen_head
        initialized_folds = []

        def check_before_fit(package, fold, config, model, head, folder, **kwargs):
            if config.get("mean_output_init"):
                initialized_folds.append(fold["fold"])
                ordinal_head = isinstance(head, training.OrdinalHead)
                projection = head.projection if ordinal_head else head
                self.assertEqual(torch.count_nonzero(projection.weight).item(), 0)
                features = torch.arange(projection.in_features, dtype=projection.weight.dtype).reshape(1, -1)
                output = head(features)
                if ordinal_head:
                    output = head.score(output)
                torch.testing.assert_close(output, torch.tensor([[1.5]]))
            return fit_frozen_head(package, fold, config, model, head, folder, **kwargs)

        with (patch.object(training, "resnet50", small_resnet),
              patch.object(training, "fit_frozen_head", side_effect=check_before_fit),
              patch.object(training, "weighted_mse", wraps=training.weighted_mse) as loss,
              patch.object(training, "lds_weights", wraps=training.lds_weights) as lds,
              patch.object(training, "lds_mse", wraps=training.lds_mse) as lds_loss,
              patch.object(training, "ordinal_step", wraps=training.ordinal_step) as ordinal,
              patch.object(training, "event_step", wraps=training.event_step) as event,
              patch.object(image_training, "apply_weak_affine", wraps=image_training.apply_weak_affine) as affine):
            result = training.run_training(layout, train_path)
            self.assertEqual(result["status"], "complete", result)
            evaluated = run_assessment(layout, assessment_path)
            self.assertEqual(evaluated["status"], "complete", evaluated)
            self.assertEqual(loss.call_count, 10)  # Two training batches per fold, no validation calls.
            self.assertEqual(lds.call_count, 5)
            for call in lds.call_args_list:
                self.assertEqual(call.args[0].tolist(), [0, 3] * 4)
            self.assertEqual(lds_loss.call_count, 10)
            self.assertEqual(ordinal.call_count, 50)
            self.assertEqual(initialized_folds, [1, 2, 3, 4, 5] * 2)
            self.assertEqual(event.call_count, 50)
            self.assertEqual(affine.call_count, 40)  # Eight training samples, one epoch, five folds.
            event_run = next(item for item in result["experiments"] if item["id"] == "c4")
            event_dir = layout.resolve(event_run["manifest"]).parent
            for fold in range(1, 6):
                train_ids = read_json(event_dir / f"fold_{fold}/interface.json")["train_sample_ids"]
                for call in event.call_args_list[2 * (fold - 1):2 * fold]:
                    targets = call.kwargs["targets_by_id"]
                    self.assertEqual(set(targets), set(train_ids))
                    for sample_id, label in zip(call.args[3], call.args[2].flatten(), strict=True):
                        self.assertEqual(targets[sample_id], [0, 0, 0, int(label.item() == 3)])
            assessment["experiments"].append({"id": "s", "parameters": {"train_id": "event_layer4",
                                                                          "calibration_epochs": 2}})
            write_json(layout.path("project", assessment_path), assessment)
            calibrated = run_assessment(layout, assessment_path, "s")
            self.assertEqual(calibrated["status"], "complete", calibrated)
        folders = {item["id"]: layout.resolve(item["manifest"]).parent for item in result["experiments"]}
        calibrated_folder = layout.resolve(calibrated["experiments"][0]["manifest"]).parent
        self.assertEqual(read_json(calibrated_folder / "metrics.json")["pooled_oof"]["studies"], 10)
        self.assertFalse(list(calibrated_folder.rglob("*.pt")))
        for fold in range(1, 6):
            interface = read_json(folders["event_layer4"] / f"fold_{fold}/interface.json")
            fit = read_json(calibrated_folder / f"calibration/fold_{fold}/calibration.json")
            self.assertGreater(fit["slope"], 0)
            self.assertEqual(len(fit["inner_folds"]), 4)
            predicted = []
            for inner in fit["inner_folds"]:
                train_ids, validation_ids = set(inner["train_sample_ids"]), inner["validation_sample_ids"]
                self.assertTrue(train_ids.isdisjoint(validation_ids))
                self.assertEqual(train_ids | set(validation_ids), set(interface["train_sample_ids"]))
                self.assertEqual(inner["best_epoch"], 2)
                predicted.extend(validation_ids)
            self.assertCountEqual(predicted, interface["train_sample_ids"])
        for fold in range(1, 6):
            history = f"fold_{fold}/history.json"
            self.assertEqual(read_json(folders["event_layer4"] / history)["epochs"][:1],
                             read_json(folders["c5"] / history)["epochs"][:1])
        for key, source in extensions.items():
            for fold in range(1, 6):
                history = f"fold_{fold}/history.json"
                self.assertEqual(read_json(folders[key] / history)["epochs"][:1],
                                 read_json(folders[source] / history)["epochs"])
        for trained, assessed in zip(result["experiments"], evaluated["experiments"], strict=True):
            key = trained["id"]
            variant = extensions.get(key, {"event_layer4": "c4", "ordinal_mean": "c3"}.get(key, key))
            pretrained = xray if variant == "xrv_all_frozen" else imagenet
            folder = layout.resolve(trained["manifest"]).parent
            state = torch.load(folder / "fold_1/best.pt", map_location="cpu", weights_only=True)
            fine_tuned = key in {"b1", "b2", "b5", "event_layer4"}
            for name, tensor in pretrained.state_dict().items():
                if name.startswith("fc."):
                    continue
                if fine_tuned and name.startswith("layer4.") and name in dict(pretrained.named_parameters()):
                    continue
                if key == "b5" and name.startswith("layer4."):
                    continue
                backbone_key = f"backbone.{name}" if variant in {"c3", "c4"} else name
                self.assertTrue(torch.equal(state[backbone_key], tensor), name)
            if fine_tuned:
                history = read_json(folder / "fold_1/history.json")["epochs"]
                self.assertEqual([entry["stage"] for entry in history], [1, 2])
            if variant == "b4":
                self.assertEqual(state["fc.0.weight"].shape[0], 128)
                self.assertEqual(state["fc.2.weight"].shape[0], 1)
            elif variant == "c3":
                self.assertEqual(state["fc.cutpoint_gaps"].numel(), len(ORDINAL_THRESHOLDS) - 1)
            elif variant == "c4":
                self.assertEqual(state["fc.weight"].shape[0], 4)
            else:
                self.assertEqual(state["fc.weight"].shape[0], 1)
            metrics = layout.resolve(assessed["manifest"]).with_name("metrics.json")
            self.assertEqual(read_json(metrics)["pooled_oof"]["studies"], 10)
