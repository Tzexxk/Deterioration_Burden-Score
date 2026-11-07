"""Small frozen synthetic sources and real CatBoost; no project data or weights."""

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.assessment.ehr.catboost.assessment import run_assessment as assess_ehr
from src.models.assessment.operating_points import operating_point
from src.models.ehr_data import load_package
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import experiment_store
from src.models.image_data import load_image_package
from src.models.released_data import ReleasedData
from src.models.train.ehr.catboost.training import run_training as train_ehr
from src.models.train.fusion.feature_fusion import training
from src.models.train.image.training import image_runtime_versions
from tests.models.train.ehr.catboost import test_training

CONFIG = "configs/models/train/fusion/feature_fusion.json"
ASSESSMENT_CONFIG = "configs/models/assessment/fusion/feature_fusion.json"
TEMPLATE_ROOT = test_training.TEMPLATE.parents[2]


class TinyEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(48, 2048), nn.BatchNorm1d(2048), nn.Dropout(.5))
        self.fc = nn.Linear(2048, 4)

    def forward(self, _pixels):
        raise AssertionError("F2 must extract before the event output head")


class FusionFixture:
    def __init__(self, case):
        fixture = test_training.CatBoostTrainingTests()
        temp_root = Path(__file__).resolve().parents[5] / "outputs/fusion_tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with patch("tempfile.tempdir", str(temp_root)):
            fixture.setUp()
        case.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(4, 4))
        self.layout = fixture.layout
        # Vary only synthetic pixels, before freezing; fixed-shape NPY byte sizes are unchanged.
        export = read_json(self.layout.resolve(fixture.store.manifest_ref("step12", "export")))
        for item in export["outputs"]:
            if item["name"].endswith(".npy"):
                number = int(Path(item["name"]).stem)
                np.save(self.layout.resolve({"root": item["root"], "path": item["path"]}),
                        np.random.default_rng(number).random((4, 4), dtype=np.float32), allow_pickle=False)
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        self.release = ReleasedData.load(self.layout, "v1_r3")
        self.package = load_package(self.release)
        ehr_config = read_json(self.layout.path("project", test_training.CONFIG))
        ehr_config["defaults"]["model"].update(iterations=3, early_stopping_rounds=1)
        write_json(self.layout.path("project", test_training.CONFIG), ehr_config)
        assert train_ehr(self.layout, test_training.CONFIG, "d2_l210")["status"] == "complete"
        config = {"data_release": "v1_r3", "defaults": {"sensitivity_target": .8, "specificity_target": .8,
                  "plot_training_history": False}, "experiments": [{"id": "d2_l210", "parameters": {"train_id": "d2_l210"}}]}
        write_json(self.layout.path("project", "configs/models/assessment/ehr/catboost.json"), config)
        assert assess_ehr(self.layout, "configs/models/assessment/ehr/catboost.json")["status"] == "complete"
        self.config = read_json(TEMPLATE_ROOT / "train/fusion/feature_fusion.json")
        self.config["data_release"] = "v1_r3"
        self.config["defaults"].update(pca_components=2, extraction={"device": "cpu", "batch_size": 3, "num_workers": 0})
        write_json(self.layout.path("project", CONFIG), self.config)
        config = read_json(TEMPLATE_ROOT / "assessment/fusion/feature_fusion.json")
        config["data_release"] = "v1_r3"
        write_json(self.layout.path("project", ASSESSMENT_CONFIG), config)
        self.publish_image_source()

    def publish_image_source(self):
        self.image_store = experiment_store(self.layout, "train", "image", "resnet50", "selected_image")
        images = load_image_package(self.release, {"kind": "package"})
        params = {"pretraining": "IMAGENET1K_V1", "image_source": {"kind": "package"}, "seed": 42,
                  "event_components": True, "layer4_lr": 3e-5}
        snapshot = {"id": "selected_image", "data_release": "v1_r3", "parameters": params,
                    "resolved_image_source": images.source, "image_shape": [4, 4],
                    "runtime_versions": image_runtime_versions(params)}
        with self.image_store.publish("train", "run", inputs=[self.release.acceptance["self"]],
                                       details={"data_release": "v1_r3", "image_shape": [4, 4]}) as work:
            write_json(work / "effective_configuration.json", snapshot)
            for fold in self.package.cv["folds"]:
                folder = work / f"fold_{fold['fold']}"
                folder.mkdir()
                torch.manual_seed(100 + fold["fold"])
                torch.save(TinyEventModel().state_dict(), folder / "best.pt")
                write_json(folder / "interface.json", {"fold": fold["fold"], "seed": 42, "best_epoch": 1,
                    "train_sample_ids": self.package.samples.iloc[fold["train_indices"]].sample_id.tolist(),
                    "validation_sample_ids": self.package.samples.iloc[fold["validation_indices"]].sample_id.tolist()})
                write_json(folder / "history.json", {"epochs": [{"epoch": 1, "validation": {"rmse": 1.0}}]})
        assessment = experiment_store(self.layout, "assessment", "image", "resnet50", "selected_image")
        frames = []
        for fold in self.package.cv["folds"]:
            rows = self.package.samples.iloc[fold["validation_indices"]][
                ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]].copy()
            frames.append(rows.assign(fold=fold["fold"], predicted_score=rows.burden_score * .4 + .2,
                baseline_mean_score=float(self.package.samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())))
        predictions = pd.concat(frames).sort_values("sample_id").reset_index(drop=True)
        points = {rule: {name: {ep: operating_point(predictions[ep].to_numpy(bool), predictions[column].to_numpy(float), rule, .8)
                  for ep in ENDPOINTS} for name, column in (("model", "predicted_score"), ("baseline_mean", "baseline_mean_score"))}
                  for rule in ("sensitivity", "specificity")}
        with assessment.publish("assessment", "run", inputs=[self.image_store.manifest_ref("train", "run")],
                                details={"data_release": "v1_r3", "train_id": "selected_image"}) as work:
            predictions.to_parquet(work / "oof_predictions.parquet", index=False)
            write_json(work / "operating_points.json", points)

    def train(self):
        with patch.object(training, "build_model", side_effect=lambda _config: TinyEventModel()):
            return training.run_training(self.layout, CONFIG, "f2_pca08")


class FeatureFusionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)

    def test_full_training_binds_sources_and_reuses_only_identical_id(self):
        f = FusionFixture(self)
        result = f.train()
        self.assertEqual(result["status"], "complete", result)
        store = experiment_store(f.layout, "train", "fusion", "feature_fusion", "f2_pca08")
        manifest = store.validate(result["experiments"][0]["manifest"])
        self.assertEqual(len(manifest["inputs"]), 3)
        records = read_json(training.output(store, manifest, "sources.json"))
        self.assertEqual(read_json(training.output(store, manifest, "effective_ehr_configuration.json")),
                         records["ehr"]["effective_training_configuration"])
        first_vectors = None
        for fold in f.package.cv["folds"]:
            augmented = training.load_fold_package(store, manifest, f.package, fold)
            pd.testing.assert_frame_equal(augmented.features.iloc[:, :-2], f.package.features)
            self.assertEqual(augmented.features.columns[-2:].tolist(), ["img_pc_01", "img_pc_02"])
            self.assertTrue(augmented.features.iloc[f.package.cv["test_indices"], -2:].isna().all().all())
            prefix = f"fold_{fold['fold']}"
            metadata = read_json(training.output(store, manifest, f"{prefix}/image_features.json"))
            self.assertEqual(metadata["checkpoint"], f.image_store.output_ref(
                read_json(f.layout.resolve(f.image_store.manifest_ref("train", "run"))), f"{prefix}/best.pt"))
            self.assertNotIn("sample-00", metadata["sample_ids"])
            self.assertNotIn("sample-01", metadata["sample_ids"])
            with np.load(training.output(store, manifest, f"{prefix}/image_features.npz"), allow_pickle=False) as saved:
                vectors = saved["vectors"]
            if first_vectors is None:
                first_vectors = vectors
            else:
                self.assertFalse(np.array_equal(vectors, first_vectors))
        with patch.object(training, "extract_vectors", side_effect=AssertionError("must reuse")):
            self.assertEqual(f.train()["experiments"][0]["status"], "unchanged")
        f.config["defaults"]["pca_components"] = 3
        write_json(f.layout.path("project", CONFIG), f.config)
        self.assertEqual(f.train()["experiments"][0]["error"], "EXPERIMENT_ID_CONFLICT")

    def test_extraction_uses_eval_before_head_and_never_reads_test_pixels(self):
        f = FusionFixture(self)
        sources, _, images = training.resolve_sources(f.layout, f.package, f.config["defaults"])
        fold = f.package.cv["folds"][0]
        original_read = images.read_image
        seen = []
        def read(index):
            self.assertNotIn(index, f.package.cv["test_indices"])
            seen.append(index)
            return original_read(index)
        images.read_image = read
        model = TinyEventModel()
        with patch.object(training, "build_model", return_value=model):
            vectors, metadata = training.extract_vectors(images, fold, sources["image"],
                f.config["defaults"]["extraction"], "synthetic")
        self.assertEqual(seen, f.package.cv["development_indices"])
        self.assertFalse(model.training)
        self.assertFalse(model.backbone[2].training)
        self.assertFalse(model.backbone[3].training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(vectors.shape, (10, 2048))
        with patch.object(training, "build_model", side_effect=lambda _config: TinyEventModel()):
            again, _ = training.extract_vectors(images, fold, sources["image"],
                f.config["defaults"]["extraction"], "synthetic")
        np.testing.assert_array_equal(vectors, again)
        self.assertFalse(metadata["augmentation"])

    def test_pca_fits_training_only_and_preserves_missing_and_categories(self):
        f = FusionFixture(self)
        fold = f.package.cv["folds"][0]
        indices = sorted(fold["train_indices"] + fold["validation_indices"])
        positions = {index: pos for pos, index in enumerate(indices)}
        train_rows = [positions[i] for i in fold["train_indices"]]
        train_ids = f.package.samples.iloc[fold["train_indices"]].sample_id.tolist()
        vectors = np.random.default_rng(42).normal(size=(10, 2048))
        metadata = {"sample_ids": f.package.samples.iloc[indices].sample_id.tolist(), "fold": fold["fold"],
                    "layer": training.LAYER, "dimension": 2048, "augmentation": False, "mode": "eval"}
        projection, arrays = training.fit_projection(vectors, train_rows, 2, train_ids)
        features = f.package.features.copy()
        features.iloc[2, 0] = np.nan
        features.iloc[3, -1] = "unseen_synthetic_category"
        package = replace(f.package, features=features)
        augmented = training.augmented_package(package, fold, vectors, metadata, projection, arrays)
        pd.testing.assert_frame_equal(augmented.features.iloc[:, :-2], features)
        changed = vectors.copy()
        changed[[positions[i] for i in fold["validation_indices"]]] += 10000
        _, other_arrays = training.fit_projection(changed, train_rows, 2, train_ids)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, other_arrays[key])
        with self.assertRaisesRegex(PipelineError, "INSUFFICIENT_IMAGE_RANK"):
            training.fit_projection(np.ones((10, 2048)), train_rows, 2, train_ids)
        with self.assertRaisesRegex(PipelineError, "INVALID_IMAGE_FEATURES"):
            training.fit_projection(vectors, train_rows, 8, train_ids)
        invalid = {**arrays, "scaler_mean": arrays["scaler_mean"] + 1}
        with self.assertRaisesRegex(PipelineError, "INVALID_FUSION_PROJECTION"):
            training.augmented_package(package, fold, vectors, metadata, projection, invalid)
        # The production dimension is also exercised; the tiny five-fold fixture has only 8 training rows.
        _, pca8 = training.fit_projection(np.random.default_rng(7).normal(size=(20, 2048)), list(range(16)),
                                          8, [f"synthetic-{i}" for i in range(16)])
        self.assertEqual(pca8["components"].shape, (8, 2048))

    def test_real_event_backbone_exposes_2048_features_without_pretrained_weights(self):
        from src.models.train.image.resnet50.training import build_model
        model = build_model({"pretraining": "IMAGENET1K_V1", "event_components": True}).eval()
        with torch.inference_mode():
            self.assertEqual(model.backbone(torch.zeros(1, 3, 32, 32)).shape, (1, 2048))

    def test_failure_rolls_back_and_stale_image_source_is_rejected(self):
        f = FusionFixture(self)
        with patch.object(training, "fit_fold", side_effect=RuntimeError("synthetic failure")):
            self.assertEqual(f.train()["experiments"][0]["status"], "failed")
        self.assertFalse(f.layout.path("outputs", "models/train/fusion/feature_fusion/f2_pca08").exists())
        self.assertEqual(f.train()["status"], "complete")
        manifest = read_json(f.layout.resolve(f.image_store.manifest_ref("train", "run")))
        state_path = f.layout.resolve(manifest["state"])
        state = read_json(state_path)
        write_json(state_path, {**state, "status": "failed"})
        self.assertEqual(f.train()["experiments"][0]["error"], "UPSTREAM_INCOMPLETE")

    def test_configuration_rejects_unsupported_sources_and_settings(self):
        defaults = read_json(TEMPLATE_ROOT / "train/fusion/feature_fusion.json")["defaults"]
        for key, value in (("pca_components", 0), ("pca_components", True),
                           ("ehr_source", {"model": "xgboost", "assessment_id": "x03"}),
                           ("extraction", {"device": "auto", "batch_size": 0, "num_workers": 0})):
            with self.assertRaises(PipelineError):
                training.validate_parameters({**defaults, key: value})
