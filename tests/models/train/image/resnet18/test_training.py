"""Exercise package and variant image loading with arbitrary synthetic dimensions."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18

from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.assessment.image.comparison import run_comparison
from src.models.assessment.image.resnet18.assessment import run_assessment
from src.models.experiments import experiment_store
from src.models.image_data import load_image_package
from src.models.released_data import ReleasedData
from src.models.train.image.resnet18.training import fit_fold, run_training
from tests.models.train.ehr.catboost import test_training

CONFIG = "configs/models/train/image/resnet18.json"
CANDIDATE = "imagenet1k_frozen"


class ResNet18TrainingTests(unittest.TestCase):
    def test_uses_declared_dimensions_and_rejects_invalid_variant_pixels(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(9, 9))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        release = ReleasedData.load(layout, "v1_r3")
        master = {"kind": "package"}
        variant = {"kind": "variant", "batch": "tiny", "size": 5}
        package = load_image_package(release, master)
        self.assertEqual(package.shape, (9, 9))
        self.assertEqual(package.read_image(0).shape, (9, 9))
        root = layout.path("processed", "v1_r3/image_variants/batch-tiny")
        with self.assertRaisesRegex(PipelineError, "VARIANT_NOT_RELEASED"):
            load_image_package(release, variant)

        manifest = {"schema": 1, "status": "released", "batch": "tiny", "sizes": [5],
                    "image_dtype": "float32", "value_range": [0.0, 1.0],
                    "images_per_size": len(package.samples), "source": {
                        "data_run": "v1_r3", "acceptance": {"manifest": release.acceptance["self"], "generation": 1},
                        "export": release.acceptance["inputs"][0],
                        "samples": release.store.output_ref(release.export, "package/samples.parquet")}}
        manifest_path = root / "manifest.json"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(PipelineError, "VARIANT_SIZE_NOT_RELEASED"):
            load_image_package(release, {**variant, "size": 7})
        pixels = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)
        for name in package.samples.image_relpath:
            path = root / "5x5" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, pixels, allow_pickle=False)
        selected = load_image_package(release, variant)
        self.assertEqual(selected.shape, (5, 5))
        np.testing.assert_array_equal(selected.read_image(0), pixels)
        self.assertTrue(package.samples.equals(selected.samples))
        self.assertEqual(package.cv, selected.cv)

        first = root / "5x5" / package.samples.iloc[0].image_relpath
        np.save(first, pixels.astype(np.float64), allow_pickle=False)
        with self.assertRaisesRegex(PipelineError, "INVALID_IMAGE_PIXELS"):
            selected.read_image(0)

    def test_frozen_baseline_keeps_backbone_state_and_skips_augmentation(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        release = ReleasedData.load(fixture.layout, "v1_r3")
        package = load_image_package(release, {"kind": "package"})
        template = Path(__file__).resolve().parents[5] / "configs/models.template/train/image/resnet18.json"
        config = read_json(template)["defaults"]
        self.assertEqual(config["head_only_epochs"], config["max_epochs"])
        self.assertIs(config["augment"], False)
        config.update(device="cpu", batch_size=4, head_only_epochs=2, max_epochs=2)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        weights = resnet18(weights=None).state_dict()
        folder = fixture.layout.path("project", "fold")
        with patch("src.models.train.image.training.TF.affine", side_effect=AssertionError("Unexpected augmentation")):
            fit_fold(package, package.cv["folds"][0], config, weights, folder, label="frozen")
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        for name, value in weights.items():
            if not name.startswith("fc."):
                self.assertTrue(torch.equal(checkpoint[name], value), name)
        self.assertEqual([row["stage"] for row in read_json(folder / "history.json")["epochs"]], [1, 1])

    def test_cv_publishes_checkpoints_and_reads_development_pixels_only(self):
        fixture = test_training.CatBoostTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(40, 36))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        layout = fixture.layout
        release = ReleasedData.load(layout, "v1_r3")
        package = load_image_package(release, {"kind": "package"})
        test_file = package.image_root / package.samples.iloc[0].image_relpath
        np.save(test_file, np.full(package.shape, np.nan, np.float32), allow_pickle=False)
        with self.assertRaisesRegex(PipelineError, "INVALID_IMAGE_PIXELS"):
            package.read_image(0)

        weights = layout.path("project", "local_inputs/weights.pth")
        weights.parent.mkdir()
        torch.save(resnet18(weights=None).state_dict(), weights)
        template = Path(__file__).resolve().parents[5] / "configs/models.template/train/image/resnet18.json"
        config = read_json(template)
        config["data_release"] = "v1_r3"
        config["defaults"].pop("augment")
        config["defaults"].update(image_source={"kind": "package"}, device="cpu", batch_size=4,
                                  head_only_epochs=1, max_epochs=2,
                                  weights=layout.ref("project", "local_inputs/weights.pth"))
        write_json(layout.path("project", CONFIG), config)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        training_log = StringIO()
        with redirect_stdout(training_log):
            result = run_training(layout, CONFIG, CANDIDATE)["experiments"][0]
        self.assertIn(f"resnet18/{CANDIDATE}: checking released image source", training_log.getvalue())
        self.assertIn(f"resnet18/{CANDIDATE} fold=1 epoch=1/2", training_log.getvalue())
        self.assertEqual(result["status"], "complete", result)
        store = experiment_store(layout, "train", "image", "resnet18", CANDIDATE)
        manifest = store.validate(result["manifest"])
        self.assertEqual(manifest["inputs"], [{"manifest": release.acceptance["self"], "generation": 1}])
        folder = layout.resolve(result["manifest"]).parent
        self.assertEqual(read_json(folder / "input_summary.json")["image_shape"], [40, 36])
        for fold in range(1, 6):
            directory = folder / f"fold_{fold}"
            self.assertTrue((directory / "best.pt").is_file())
            self.assertTrue((directory / "validation_predictions.parquet").is_file())
            self.assertEqual([row["stage"] for row in read_json(directory / "history.json")["epochs"]], [1, 2])
            interface = read_json(directory / "interface.json")
            self.assertNotIn("sample-00", interface["train_sample_ids"] + interface["validation_sample_ids"])
        self.assertEqual(run_training(layout, CONFIG, CANDIDATE)["experiments"][0]["status"], "unchanged")

        config_path = "configs/models/assessment/image/resnet18.json"
        assessment_template = template.parent.parent.parent / "assessment/image/resnet18.json"
        write_json(layout.path("project", config_path), dict(read_json(assessment_template), data_release="v1_r3"))
        assessment = run_assessment(layout, config_path, CANDIDATE)["experiments"][0]
        self.assertEqual(assessment["status"], "complete", assessment)
        assessment_store = experiment_store(layout, "assessment", "image", "resnet18", CANDIDATE)
        assessed = assessment_store.validate(assessment["manifest"])
        self.assertEqual(assessed["inputs"], [{"manifest": result["manifest"], "generation": 1}])
        directory = layout.resolve(assessment["manifest"]).parent
        oof = pd.read_parquet(directory / "oof_predictions.parquet")
        expected = pd.concat([pd.read_parquet(folder / f"fold_{i}/validation_predictions.parquet")
                              for i in range(1, 6)]).sort_values("sample_id")
        self.assertEqual(oof.sample_id.tolist(), expected.sample_id.tolist())
        np.testing.assert_array_equal(oof.predicted_score, expected.predicted_score)
        np.testing.assert_array_equal(oof.baseline_mean_score, np.full(10, 1.5))
        self.assertTrue((directory / "training_history.svg").is_file())
        for rule in ("sensitivity", "specificity"):
            self.assertTrue((directory / "threshold_based" / f"{rule}_80" / "confusion_matrices.svg").is_file())

        progress = []
        comparison = run_comparison(layout, "paired_image", draws=12, progress=progress.append)
        self.assertEqual(comparison["status"], "complete", comparison)
        comparison_store = experiment_store(layout, "assessment", "image", "comparison", "paired_image")
        compared = comparison_store.validate(comparison["manifest"])
        self.assertEqual(compared["inputs"], [{"manifest": assessment["manifest"], "generation": 1}])
        directory = layout.resolve(comparison["manifest"]).parent
        self.assertEqual(directory, layout.path("outputs", "models/assessment/image/comparison/paired_image"))
        self.assertEqual(len(list(directory.rglob("*.svg"))), 18)
        metrics = read_json(directory / "threshold_free/metrics.json")
        self.assertEqual(set(metrics), {"baseline_mean", f"resnet18/{CANDIDATE}"})
        self.assertEqual(metrics[f"resnet18/{CANDIDATE}"]["rmse"]["valid_draws"], 12)
        self.assertTrue(any(message.startswith("Validating assessments: 1/1") for message in progress))
        self.assertTrue(any(message.startswith("Bootstrap (12 draws): 2/2") for message in progress))
        self.assertTrue(any(message.startswith("Writing SVGs: 18/18") for message in progress))
        self.assertEqual(run_comparison(layout, "paired_image", draws=12)["status"], "unchanged")

        config_file = layout.path("project", config_path)
        config = read_json(config_file)
        config["experiments"].append({"id": "second", "parameters": {"train_id": CANDIDATE}})
        write_json(config_file, config)
        assessment_log = StringIO()
        with redirect_stdout(assessment_log):
            second = run_assessment(layout, config_path, "second")["experiments"][0]
        self.assertIn("resnet18/second fold=1: reloading epoch", assessment_log.getvalue())
        self.assertEqual(second["status"], "complete", second)
        write_json(layout.path("outputs", "models/assessment/image/other/incomplete/manifest.json"),
                   {"status": "running"})

        primary, extra = f"resnet18/{CANDIDATE}", "resnet18/second"
        selected = run_comparison(layout, "selected_image", assessment_ids=[extra, primary], draws=12)
        self.assertEqual(selected["status"], "complete", selected)
        store = experiment_store(layout, "assessment", "image", "comparison", "selected_image")
        manifest = store.validate(selected["manifest"])
        self.assertEqual(manifest["inputs"], [
            {"manifest": entry["manifest"], "generation": 1} for entry in (assessment, second)
        ])
        root = layout.resolve(selected["manifest"]).parent
        snapshot = read_json(root / "effective_configuration.json")
        self.assertEqual(snapshot["parameters"]["assessment_ids"], [primary, extra])
        metrics = read_json(root / "threshold_free/metrics.json")
        self.assertEqual(set(metrics), {"baseline_mean", primary, extra})
        self.assertEqual(run_comparison(layout, "selected_image", assessment_ids=[primary, extra],
                                        draws=12)["status"], "unchanged")
        with self.assertRaisesRegex(PipelineError, "EXPERIMENT_ID_CONFLICT"):
            run_comparison(layout, "selected_image", assessment_ids=[primary], draws=12)
        for requested, error in [
            ([primary, primary], "DUPLICATE_ASSESSMENT"),
            (["resnet18/missing"], "ASSESSMENT_NOT_FOUND"),
            (["other/incomplete"], "UPSTREAM_INCOMPLETE"),
            ([CANDIDATE], "INVALID_ASSESSMENT_SELECTION"),
        ]:
            with self.subTest(requested=requested), self.assertRaisesRegex(PipelineError, error):
                run_comparison(layout, "bad_selection", assessment_ids=requested, draws=12)
        self.assertFalse(layout.path("outputs", "models/assessment/image/comparison/bad_selection").exists())
