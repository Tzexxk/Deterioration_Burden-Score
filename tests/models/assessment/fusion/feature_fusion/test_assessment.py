"""Reload F2 models and cached projections on synthetic development data."""

import unittest
from unittest.mock import patch

import pandas as pd
import torch

from src.common.manifest import write_json
from src.common.paths import read_json
from src.models.assessment.fusion.feature_fusion.assessment import run_assessment
from src.models.experiments import experiment_store
from src.models.train.fusion.feature_fusion import training
from tests.models.train.fusion.feature_fusion.test_training import (
    ASSESSMENT_CONFIG,
    FusionFixture,
)


class FeatureFusionAssessmentTests(unittest.TestCase):
    def test_assessment_reloads_without_pixels_binds_train_and_rejects_invalid_cache(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        f = FusionFixture(self)
        self.assertEqual(f.train()["status"], "complete")
        with patch.object(training, "extract_vectors", side_effect=AssertionError("no extraction during assessment")):
            result = run_assessment(f.layout, ASSESSMENT_CONFIG)
        self.assertEqual(result["status"], "complete", result)
        store = experiment_store(f.layout, "assessment", "fusion", "feature_fusion", "f2_pca08")
        manifest = store.validate(result["experiments"][0]["manifest"])
        self.assertEqual(manifest["inputs"], [{"manifest": f.layout.ref("outputs", "models/train/fusion/feature_fusion/f2_pca08/manifest.json"), "generation": 1}])
        folder = f.layout.resolve(manifest["self"]).parent
        rows = pd.read_parquet(folder / "oof_predictions.parquet")
        self.assertEqual(rows.sample_id.tolist(), [f"sample-{i:02d}" for i in range(2, 12)])
        self.assertEqual(len(read_json(folder / "metrics.json")["folds"]), 5)
        self.assertTrue((folder / "training_history.svg").is_file())
        self.assertTrue((folder / "threshold_based/sensitivity_80/confusion_matrices.svg").is_file())
        self.assertEqual(run_assessment(f.layout, ASSESSMENT_CONFIG)["experiments"][0]["status"], "unchanged")
        # Synthetic tampering with a size-updated manifest exercises semantic cache checks.
        train = experiment_store(f.layout, "train", "fusion", "feature_fusion", "f2_pca08")
        train_ref = train.manifest_ref("train", "run")
        tm = read_json(f.layout.resolve(train_ref))
        path = training.output(train, tm, "fold_1/image_features.json")
        metadata = read_json(path)
        metadata["checkpoint"]["path"] = metadata["checkpoint"]["path"].replace("fold_1", "fold_2")
        write_json(path, metadata)
        next(item for item in tm["outputs"] if item["name"] == "fold_1/image_features.json")["bytes"] = path.stat().st_size
        write_json(f.layout.resolve(train_ref), tm)
        config = read_json(f.layout.path("project", ASSESSMENT_CONFIG))
        config["experiments"][0]["id"] = "invalid_cache"
        write_json(f.layout.path("project", ASSESSMENT_CONFIG), config)
        self.assertEqual(run_assessment(f.layout, ASSESSMENT_CONFIG)["experiments"][0]["error"], "INVALID_FUSION_FEATURES")
