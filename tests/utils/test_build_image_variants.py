"""Exercise atomic batch publication and independent pixel validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.utils.build_image_variants import run_variants

CONFIG = "configs/utils/image_variants.json"
NAMES = ("images/a.npy", "images/b.npy")


class ImageVariantTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name).resolve()
        (self.root / "pyproject.toml").write_text("", encoding="utf-8")
        write_json(self.root / "configs/environment/paths.local.json", {"roots": {}})
        write_json(self.root / CONFIG, {"batch": "test", "sizes": [32, 16], "workers": 2})
        self.layout = Layout(self.root)
        self.context = DataContext(self.layout, 1, 3)
        self.store = ManifestStore(self.layout, data=self.context)
        self.export = self.store.manifest_ref("step12", "export")
        self.acceptance = self.store.manifest_ref("step13", "validate")
        self.master = self.root / "data/processed/v1_r3/steps/step12/export/package"
        self.variant = self.root / "data/processed/v1_r3/image_variants/batch-test"

    def publish_master(self, *, constant=False):
        with self.store.publish("step12", "export", data_directories=("package",)) as work:
            package = work / "package"
            (package / "images").mkdir(parents=True)
            pq.write_table(pa.table({"sample_id": ["a", "b"], "image_relpath": list(NAMES)}),
                           package / "samples.parquet")
            write_json(package / "dataset_metadata.json", {"data_run": "v1_r3", "release_id": "v1_r3",
                       "image": {"format": "npy", "dtype": "float32", "shape": [1024, 1024],
                                 "value_range": [0.0, 1.0], "model_normalization_applied": False}})
            gradient = np.broadcast_to(np.linspace(0, 1, 1024, dtype=np.float32), (1024, 1024))
            np.save(package / NAMES[0], gradient)
            np.save(package / NAMES[1], np.zeros_like(gradient) if constant else gradient.T)
        with self.store.publish("step13", "validate", inputs=[self.export],
                                details={"checks_passed": True, "training_package_accepted": True}) as work:
            (work / "report.md").write_text("Synthetic acceptance", encoding="utf-8")

    def test_batch_freeze_and_independent_tamper_detection(self):
        self.publish_master()
        with self.assertRaisesRegex(PipelineError, "MASTER_NOT_RELEASED"):
            run_variants(self.context, CONFIG)
        self.store.freeze(self.acceptance)
        release = self.store.control / "release.json"
        before = release.read_bytes()
        result = run_variants(self.context, CONFIG)
        self.assertEqual(result["sizes"], [16, 32])
        self.assertEqual(result["status"], "released")
        manifest = read_json(self.layout.resolve(result["manifest"]))
        self.assertEqual(manifest["source"]["acceptance"], {"manifest": self.acceptance, "generation": 1})
        self.assertEqual(manifest["source"]["export"], {"manifest": self.export, "generation": 1})
        self.assertEqual(manifest["images_per_size"], 2)
        self.assertEqual(np.load(self.variant / "16x16" / NAMES[0]).shape, (16, 16))
        self.assertEqual(np.load(self.variant / "32x32" / NAMES[0]).shape, (32, 32))
        self.assertEqual(run_variants(self.context, CONFIG, validate_only=True)["status"], "validated")
        with self.assertRaisesRegex(PipelineError, "VARIANT_IMMUTABLE"):
            run_variants(self.context, CONFIG)
        a, b = (self.variant / "16x16" / name for name in NAMES)
        a.write_bytes(b.read_bytes())
        with self.assertRaisesRegex(PipelineError, "VARIANT_PIXELS_MISMATCH"):
            run_variants(self.context, CONFIG, validate_only=True)
        self.assertEqual(release.read_bytes(), before)
        self.assertEqual(np.load(self.master / NAMES[0]).shape, (1024, 1024))

    def test_failed_size_does_not_publish_any_batch(self):
        self.publish_master(constant=True)
        self.store.freeze(self.acceptance)
        staging = self.variant.parent / ".batch-test.staging"
        staging.mkdir(parents=True)
        (staging / "interrupted").touch()
        with self.assertRaisesRegex(PipelineError, "CONSTANT_VARIANT_IMAGE"):
            run_variants(self.context, CONFIG)
        self.assertFalse(self.variant.exists())
        self.assertFalse(staging.exists())
