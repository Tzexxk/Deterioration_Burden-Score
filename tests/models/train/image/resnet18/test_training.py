"""Exercise package and variant image loading with arbitrary synthetic dimensions."""

import unittest
from pathlib import Path

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
from src.models.train.image.resnet18.training import run_training
from tests.models.train.ehr.catboost import test_training

CONFIG = "configs/models/train/image/resnet18.json"


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
        config["defaults"].update(image_source={"kind": "package"}, device="cpu", batch_size=4,
                                  head_only_epochs=1, max_epochs=2,
                                  weights=layout.ref("project", "local_inputs/weights.pth"))
        write_json(layout.path("project", CONFIG), config)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        result = run_training(layout, CONFIG, "baseline")["experiments"][0]
        self.assertEqual(result["status"], "complete", result)
        store = experiment_store(layout, "train", "image", "resnet18", "baseline")
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
        self.assertEqual(run_training(layout, CONFIG, "baseline")["experiments"][0]["status"], "unchanged")

        config_path = "configs/models/assessment/image/resnet18.json"
        assessment_template = template.parent.parent.parent / "assessment/image/resnet18.json"
        write_json(layout.path("project", config_path), dict(read_json(assessment_template), data_release="v1_r3"))
        assessment = run_assessment(layout, config_path, "baseline")["experiments"][0]
        self.assertEqual(assessment["status"], "complete", assessment)
        assessment_store = experiment_store(layout, "assessment", "image", "resnet18", "baseline")
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
        self.assertEqual(set(metrics), {"baseline_mean", "resnet18/baseline"})
        self.assertEqual(metrics["resnet18/baseline"]["rmse"]["valid_draws"], 12)
        self.assertTrue(any(message.startswith("Validating assessments: 1/1") for message in progress))
        self.assertTrue(any(message.startswith("Bootstrap (12 draws): 2/2") for message in progress))
        self.assertTrue(any(message.startswith("Writing SVGs: 18/18") for message in progress))
        self.assertEqual(run_comparison(layout, "paired_image", draws=12)["status"], "unchanged")
