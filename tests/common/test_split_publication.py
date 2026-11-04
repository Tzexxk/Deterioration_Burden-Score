"""Two-root publication, crash recovery, inventory and independent copy checks."""

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError
from tests.common import test_foundation as foundation


class SplitPublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = foundation.FoundationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store, self.layout = self.fixture.store, self.fixture.layout
        self.ref = self.store.manifest_ref("step02", "build")
        self.data = self.layout.path("processed", "v1_r1/steps/step02/build")
        self.reports = self.layout.resolve(self.ref).parent

    def publish(self, value="old"):
        with self.store.publish("step02", "build", data_files=("table.bin",)) as work:
            (work / "table.bin").write_bytes(value.encode())
            write_json(work / "summary.json", {"value": value})
        return self.store.validate(self.ref)

    def assert_restored(self, generation=1):
        self.assertEqual((self.data / "table.bin").read_bytes(), b"old")
        self.assertEqual(json.loads((self.reports / "summary.json").read_text())["value"], "old")
        self.assertEqual(json.loads(self.layout.resolve(self.ref).read_text())["generation"], generation)
        self.fixture.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(self.ref))
        for root in ("processed", "outputs"):
            self.assertFalse(any(self.layout.roots[root].rglob("temp")))

    def test_inventory_locations_and_missing_data_rejected(self):
        manifest = self.publish()
        self.assertEqual(self.ref["root"], "outputs")
        self.assertEqual({p.name for p in self.data.iterdir()}, {"table.bin"})
        self.assertEqual({p.name for p in self.reports.iterdir()}, {"summary.json", "manifest.json"})
        self.assertEqual(self.store.output_ref(manifest, "table.bin")["root"], "processed")
        self.assertEqual(self.store.output_ref(manifest, "summary.json")["root"], "outputs")
        (self.data / "table.bin").unlink()
        self.fixture.error("OUTPUT_MISSING_OR_CHANGED", lambda: self.store.validate(self.ref))

    def test_failure_between_data_and_report_swap_rolls_back_both(self):
        self.publish()
        rename = Path.rename
        def fail(path, target):
            if path.parent.name == "temp" and path.is_relative_to(self.layout.roots["outputs"]) and ".backup" not in path.name:
                self.assertEqual((self.data / "table.bin").read_bytes(), b"new")
                self.fixture.error("MISSING_FILE", lambda: self.store.validate(self.ref))
                raise OSError("synthetic report swap failure")
            return rename(path, target)
        with patch.object(Path, "rename", fail), self.assertRaises(OSError):
            self.publish("new")
        self.assert_restored()
        self.assertEqual(self.publish("retry")["generation"], 2)

    def test_first_publication_failure_leaves_neither_destination(self):
        rename = Path.rename
        def fail(path, target):
            if path.parent.name == "temp" and path.is_relative_to(self.layout.roots["outputs"]):
                raise OSError("synthetic report swap failure")
            return rename(path, target)
        with patch.object(Path, "rename", fail), self.assertRaises(OSError):
            self.publish()
        self.assertFalse(self.data.exists())
        self.assertFalse(self.reports.exists())
        self.assertEqual(self.publish()["generation"], 1)

    def test_failure_after_both_swaps_restores_both(self):
        self.publish()
        original = write_json
        def fail(path, value):
            if path.name == "step02.build.state.json" and value["status"] == "complete":
                raise OSError("synthetic completion write failure")
            return original(path, value)
        with patch("src.common.manifest.write_json", fail), self.assertRaises(OSError):
            self.publish("new")
        self.assert_restored()

    def test_data_copy_failure_preserves_previous_publication(self):
        self.publish()
        with patch("src.common.manifest.shutil.copyfile", side_effect=OSError("synthetic volume failure")), self.assertRaises(OSError):
            self.publish("new")
        self.assert_restored()

    def test_missing_declared_data_is_not_published(self):
        with self.assertRaises(PipelineError) as caught:
            with self.store.publish("step02", "build", data_files=("missing.bin",)) as work:
                (work / "report.txt").write_text("report")
        self.assertEqual(caught.exception.code, "MISSING_DATA_OUTPUT")
        self.assertFalse(self.data.exists())
        self.assertFalse(self.reports.exists())

    def test_inventory_cannot_point_to_another_stage(self):
        manifest = self.publish()
        manifest["outputs"][0]["path"] = "v1_r1/steps/step03/select/table.bin"
        write_json(self.layout.resolve(self.ref), manifest)
        self.fixture.error("INVALID_MANIFEST", lambda: self.store.validate(self.ref))

    def test_unmanaged_data_directory_is_not_overwritten(self):
        self.data.mkdir(parents=True)
        (self.data / "keep.txt").write_text("keep")
        self.fixture.error("UNMANAGED_DESTINATION", self.publish)
        self.assertEqual((self.data / "keep.txt").read_text(), "keep")

    def test_process_exit_after_data_swap_is_recoverable(self):
        self.publish()
        code = '''
import os, sys
from pathlib import Path
from src.common.paths import Layout
from src.common.config import DataContext
from src.common.manifest import ManifestStore
layout = Layout(sys.argv[1])
store = ManifestStore(layout, data=DataContext.load(layout, 1, 1))
rename = Path.rename
def stop(path, target):
    if path.parent.name == "temp" and path.is_relative_to(layout.roots["outputs"]):
        os._exit(17)
    return rename(path, target)
Path.rename = stop
with store.publish("step02", "build", data_files=("table.bin",)) as work:
    (work / "table.bin").write_bytes(b"new")
    (work / "summary.json").write_text('{}')
'''
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.fixture.root)], cwd=Path(__file__).resolve().parents[2], capture_output=True)
        self.assertEqual(result.returncode, 17, result.stderr.decode(errors="replace"))
        self.fixture.error("MISSING_FILE", lambda: self.store.validate(self.ref))
        self.store.recover()
        self.assert_restored()

    def test_split_copy_is_independent_after_source_removed(self):
        self.publish()
        with self.store.publish("step13", "validate", inputs=[self.ref], details={"checks_passed": True}) as work:
            write_json(work / "summary.json", {"checks_passed": True})
        self.store.freeze(self.store.manifest_ref("step13", "validate"))
        target = ManifestStore(self.layout, data=DataContext(self.layout, 1, 2))
        copied = target.copy("step02", "build", source_store=self.store, source=self.ref)
        manifest = target.validate(copied)
        data = self.layout.resolve(target.output_ref(manifest, "table.bin"))
        self.assertFalse(os.path.samefile(data, self.data / "table.bin"))
        for root, path in [("processed", "v1_r1"), ("outputs", "data/v1_r1")]:
            source = self.layout.path(root, path)
            self.assertTrue(source.is_relative_to(self.fixture.root))
            shutil.rmtree(source)
        self.assertEqual(target.validate(copied)["action"], "copied")
        self.assertEqual(data.read_bytes(), b"old")

    def test_split_project_relocation_preserves_references(self):
        self.publish()
        moved = self.fixture.root.parent / "relocated"
        shutil.copytree(self.fixture.root, moved)
        layout = Layout(moved)
        store = ManifestStore(layout, data=DataContext.load(layout, 1, 1))
        self.assertEqual(store.validate(self.ref)["generation"], 1)

    def test_disk_preflight_checks_both_destinations_before_running(self):
        self.publish()
        self.data.parent.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.fixture.root)
        with patch("src.common.manifest.shutil.disk_usage", return_value=usage) as check:
            self.publish("new")
        checked = [call.args[0] for call in check.call_args_list]
        self.assertTrue(any(path.is_relative_to(self.layout.roots["processed"]) for path in checked))
        self.assertTrue(any(path.is_relative_to(self.layout.roots["outputs"]) for path in checked))


if __name__ == "__main__":
    unittest.main()
