"""Small synthetic checks only; no patient data and no source datasets."""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.common.config import DataContext, add_data_arguments, load_model_config
from src.common.manifest import ManifestStore, _rename, file_lock, write_json
from src.common.paths import Layout, PipelineError, portable


class FoundationTests(unittest.TestCase):
    def test_raw_slash_number_results_preserve_text_but_are_not_file_paths(self):
        for value in ("/14.", "/16.", "/18.", "/40."):
            self.assertEqual(portable(value), value)
            self.assertEqual(portable({"value": value}), {"value": value})
            self.assertEqual(portable(json.dumps({"value": value})), json.dumps({"value": value}))
            with self.assertRaises(PipelineError):
                self.layout.path("outputs", value)
        slash = chr(47)
        for value in (slash + "18./file", "File " + slash + "18./file", slash + "outside/file"):
            with self.assertRaises(PipelineError):
                portable(value)

    def test_slash_number_diagnostic_csv_can_be_published(self):
        with self.store.publish("step10", "audit") as work:
            (work / "examples.csv").write_text("record,value\nsynthetic,/18.\n", encoding="utf-8")
        manifest = self.store.validate(self.store.manifest_ref("step10", "audit"))
        self.assertEqual(manifest["generation"], 1)

    def test_lab_count_units_and_plus_minus_are_portable(self):
        values = {"units": ["#/hpf", "#/lpf", "#/uL", "+/-"]}
        self.assertEqual(portable(values), values)
        self.assertEqual(portable(json.dumps(values)), json.dumps(values))
        with self.assertRaises(PipelineError):
            portable(chr(47) + "outside/file")

    def test_compound_medication_names_do_not_mask_absolute_paths(self):
        value = {"medication": "compound 2%/ingredient 0.5%"}
        self.assertEqual(portable(value), value)
        self.assertEqual(portable(json.dumps(value)), json.dumps(value))
        outside = chr(47) + "outside/file"
        for unsafe in (outside, "File " + outside, "prefix%" + chr(67) + chr(58) + outside):
            with self.assertRaises(PipelineError):
                portable(unsafe)

    def test_windows_rename_denial_retries_then_succeeds(self):
        busy = PermissionError("synthetic sharing denial")
        busy.winerror = 5
        with patch.object(Path, "rename", side_effect=[busy, None]) as rename, patch("src.common.manifest.time.sleep") as sleep:
            _rename(Path("source"), Path("destination"))
        self.assertEqual(rename.call_count, 2)
        sleep.assert_called_once_with(0.05)

    def test_windows_rename_denial_has_a_fixed_retry_limit(self):
        busy = PermissionError("synthetic persistent denial")
        busy.winerror = 32
        with patch.object(Path, "rename", side_effect=busy) as rename, patch("src.common.manifest.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                _rename(Path("source"), Path("destination"))
        self.assertEqual(rename.call_count, 5)
        self.assertEqual(sleep.call_count, 4)

    def test_unrelated_rename_permission_errors_are_not_retried(self):
        error = PermissionError("synthetic non-sharing error")
        with patch.object(Path, "rename", side_effect=error), patch("src.common.manifest.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                _rename(Path("source"), Path("destination"))
        sleep.assert_not_called()

    def setUp(self):
        temp_base = Path(__file__).resolve().parents[2] / "outputs/foundation_tests"
        temp_base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        (self.root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
        write_json(self.root / "configs/environment/paths.local.json", {"roots": {}})
        write_json(self.root / "configs/version_run.json", {"version": 1, "run": 1})
        for run in (1, 2, 3):
            write_json(self.root / f"configs/data/v1_r{run}/cohort/cohort.json", {"horizon_hours": 72, "lookback_hours": 48})
        write_json(self.root / "configs/models/ehr/catboost.json", {"family": "catboost"})
        self.layout = Layout(self.root)
        self.ctx = DataContext.load(self.layout, 1, 1)
        self.store = ManifestStore(self.layout, data=self.ctx)

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.root), str(caught.exception))

    def publish(self, stage="step01", phase="audit", inputs=(), details=None):
        with self.store.publish(stage, phase, inputs=inputs, details=details) as work:
            (work / "result.txt").write_text("synthetic", encoding="utf-8")
        return self.store.manifest_ref(stage, phase)

    def test_version_selection_and_no_fallback(self):
        self.assertEqual(self.ctx.read("cohort/cohort.json")["horizon_hours"], 72)
        self.error("VERSION_MISMATCH", lambda: DataContext.load(self.layout, 1, 2))
        self.error("MISSING_FILE", lambda: self.ctx.read("cohort/missing.json"))
        self.error("MISSING_CONFIG_FIELD", lambda: self.ctx.read("cohort/cohort.json", required=("missing",)))
        ref = self.layout.ref("project", "configs/data/v1_r2/cohort/cohort.json")
        self.error("CROSS_VERSION_CONFIG", lambda: self.ctx.check_config_ref(ref))

    def test_required_arguments_and_positive_values(self):
        parser = argparse.ArgumentParser()
        add_data_arguments(parser)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])
        for value in (0, -1, True, "1"):
            self.error("INVALID_VERSION", lambda: DataContext.load(self.layout, value, 1))

    def test_single_outcome_and_independent_lookback(self):
        path = self.root / "configs/data/v1_r1/cohort/cohort.json"
        write_json(path, {"horizon_hours": 48})
        self.error("UNSUPPORTED_OUTCOME_WINDOW", lambda: self.ctx.read("cohort/cohort.json"))
        write_json(path, {"audit_horizons_hours": [48, 72]})
        self.error("UNSUPPORTED_OUTCOME_WINDOW", lambda: self.ctx.read("cohort/cohort.json"))
        write_json(path, {"horizon_hours": 72, "lookback_hours": 48})
        self.assertEqual(self.ctx.read("cohort/cohort.json")["lookback_hours"], 48)

    def test_model_config_ignores_active_data_selection(self):
        write_json(self.root / "configs/version_run.json", {"version": 9, "run": 9})
        self.assertEqual(load_model_config(self.layout, "configs/models/ehr/catboost.json")["family"], "catboost")
        model = ManifestStore(self.layout, family="ehr", experiment="synthetic01")
        with model.publish("audit", "run") as work:
            (work / "summary.json").write_text('{}', encoding="utf-8")
        ref = model.manifest_ref("audit", "run")
        self.assertEqual(model.validate(ref)["scope"]["kind"], "model")
        self.error("EXPERIMENT_EXISTS", lambda: self._publish_model_again(model))

    def test_model_dependency_requires_frozen_data_and_ignores_active_pointer(self):
        a = self.publish()
        model = ManifestStore(self.layout, family="image", experiment="synthetic01")
        self.error("SOURCE_NOT_RELEASED", lambda: model.validate(a))
        acceptance = self.publish("step13", "validate", inputs=[a], details={"checks_passed": True})
        self.store.freeze(acceptance)
        write_json(self.root / "configs/version_run.json", {"version": 9, "run": 9})
        with model.publish("audit", "run", inputs=[a]) as work:
            (work / "summary.json").write_text('{}', encoding="utf-8")
        self.assertEqual(model.validate(model.manifest_ref("audit", "run"))["inputs"][0]["generation"], 1)

    def _publish_model_again(self, model):
        with model.publish("audit", "run"):
            pass

    def test_path_traversal_absolute_and_serialization(self):
        for value in ("../outside", self.root.as_posix(), "bad\\name"):
            self.error("INVALID_PATH", lambda: self.layout.path("project", value))
        self.error("ABSOLUTE_PATH_FORBIDDEN", lambda: portable({"note": "File " + str(self.root)}))
        self.error("ABSOLUTE_PATH_FORBIDDEN", lambda: write_json(self.root / "leak.json", {"root": str(self.root)}))
        self.assertFalse((self.root / "leak.json").exists())
        self.assertEqual(portable("https://example.org/source/file"), "https://example.org/source/file")

    def test_config_cannot_serialize_absolute_paths(self):
        config = self.root / "configs/data/v1_r1/cohort/cohort.json"
        config.write_text(json.dumps({"path": str(self.root)}), encoding="utf-8")
        self.error("ABSOLUTE_PATH_FORBIDDEN", lambda: self.ctx.read("cohort/cohort.json"))

    def test_writable_input_overlap_rejected(self):
        local = self.root / "configs/environment/paths.local.json"
        local.write_text(json.dumps({"roots": {"processed": "data", "raw": "data/raw"}}), encoding="utf-8")
        self.error("INVALID_PATH_CONFIG", lambda: Layout(self.root))

    def test_generation_is_per_stage_and_stale_is_transitive(self):
        a = self.publish()
        b = self.publish("step02", inputs=[a])
        c = self.publish("step03", inputs=[b])
        self.assertEqual(self.store.validate(c)["generation"], 1)
        self.publish()
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(b))
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(c))
        self.publish("step02", inputs=[a])
        self.publish("step03", inputs=[b])
        self.assertEqual(self.store.validate(c)["generation"], 2)

    def test_failure_preserves_output_and_does_not_increment_generation(self):
        a = self.publish()
        original = self.layout.resolve(a).read_bytes()
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with self.store.publish("step01", "audit") as work:
                (work / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("synthetic failure")
        self.assertEqual(self.layout.resolve(a).read_bytes(), original)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(a))
        self.assertEqual(self.store.validate(self.publish())["generation"], 2)
        self.assertFalse(any(self.layout.roots["processed"].rglob("temp")))

    def test_failed_first_run_can_retry_at_generation_one(self):
        with self.assertRaises(RuntimeError):
            with self.store.publish("step01", "audit"):
                raise RuntimeError()
        self.assertEqual(self.store.validate(self.publish())["generation"], 1)

    def test_interrupted_rename_restores_previous_publication(self):
        a = self.publish()
        manifest = self.layout.resolve(a).read_bytes()
        original_rename = Path.rename

        def interrupt(path, target):
            if path.parent.name == "temp" and path.name.startswith("audit."):
                raise OSError("synthetic interruption")
            return original_rename(path, target)

        with self.assertRaises(OSError), patch.object(Path, "rename", interrupt):
            # Restoration rename must succeed; fail only the new staging directory.
            with self.store.publish("step01", "audit") as work:
                (work / "new.txt").write_text("new", encoding="utf-8")
        # The recovery journal survives even if restoration was interrupted too.
        self.store.recover()
        self.assertEqual(self.layout.resolve(a).read_bytes(), manifest)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(a))
        self.assertFalse(any(self.layout.roots["processed"].rglob("temp")))

    def test_os_lock_blocks_second_writer(self):
        with file_lock(self.store.control / "writer.lock"):
            self.error("CONCURRENT_RUN", self.store.recover)

    def test_process_crash_releases_lock_and_can_recover(self):
        a = self.publish()
        before = self.layout.resolve(a).read_bytes()
        code = """
import os, sys
from src.common.paths import Layout
from src.common.config import DataContext
from src.common.manifest import ManifestStore
layout = Layout(sys.argv[1])
store = ManifestStore(layout, data=DataContext.load(layout, 1, 1))
with store.publish('step01', 'audit') as work:
    (work / 'partial.txt').write_text('partial', encoding='utf-8')
    os._exit(17)
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.root)],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True)
        self.assertEqual(result.returncode, 17, result.stderr.decode(errors="replace"))
        self.error("UPSTREAM_INCOMPLETE", lambda: self.store.validate(a))
        self.store.recover()
        self.assertEqual(self.layout.resolve(a).read_bytes(), before)
        self.assertEqual(self.store.validate(self.publish())["generation"], 2)
        self.assertFalse(any(self.layout.roots["processed"].rglob("temp")))

    def test_cross_run_requires_explicit_copy_and_newer_run_rejected(self):
        a = self.publish()
        second = ManifestStore(self.layout, data=DataContext(self.layout, 1, 2))
        self.error("UNDECLARED_REUSE", lambda: second.validate(a))
        old = self.store
        self.store = second
        newer = self.publish()
        self.error("UPSTREAM_FROM_NEWER_RUN", lambda: old.validate(newer))

    def test_physical_copy_is_independent_and_release_immutable(self):
        a = self.publish()
        accept = self.publish("step13", "validate", inputs=[a], details={"checks_passed": True})
        self.store.freeze(accept)
        self.error("RELEASE_IMMUTABLE", lambda: self.publish())
        target = ManifestStore(self.layout, data=DataContext(self.layout, 1, 3))
        copied = target.copy("step01", "audit", source_store=self.store, source=a)
        target_path = self.layout.resolve(copied).parent / "result.txt"
        source_path = self.layout.resolve(a).parent / "result.txt"
        self.assertFalse(os.path.samefile(source_path, target_path))
        source_path.write_text("changed", encoding="utf-8")
        self.assertEqual(target_path.read_text(encoding="utf-8"), "synthetic")
        # Remove source only inside this owned temporary test project.
        source_root = self.layout.path("outputs", "data/v1_r1")
        self.assertTrue(source_root.is_relative_to(self.root))
        shutil.rmtree(source_root)
        self.assertEqual(target.validate(copied)["action"], "copied")

    def test_copy_requires_frozen_source(self):
        a = self.publish()
        target = ManifestStore(self.layout, data=DataContext(self.layout, 1, 2))
        self.error("SOURCE_NOT_RELEASED", lambda: target.copy("step01", "audit", source_store=self.store, source=a))

    def test_modes_are_separate_and_no_letter_steps(self):
        audit = self.publish("step05", "audit")
        frozen = self.publish("step05", "freeze", inputs=[audit])
        self.publish("step05", "audit")
        self.error("UPSTREAM_REBUILT", lambda: self.store.validate(frozen))
        self.error("INVALID_STAGE", lambda: self.store.manifest_ref("step10a", "audit"))

    def test_move_project_preserves_references(self):
        a = self.publish()
        moved = self.root.parent / "moved"
        shutil.copytree(self.root, moved)
        relocated = Layout(moved)
        store = ManifestStore(relocated, data=DataContext.load(relocated, 1, 1))
        self.assertEqual(store.validate(a)["generation"], 1)

    def test_missing_payload_rejected(self):
        a = self.publish()
        (self.layout.resolve(a).parent / "result.txt").unlink()
        self.error("OUTPUT_MISSING_OR_CHANGED", lambda: self.store.validate(a))

    def test_text_output_absolute_path_is_not_published(self):
        def operation():
            with self.store.publish("step01", "audit") as work:
                (work / "log.txt").write_text("Path: " + str(self.root), encoding="utf-8")
        self.error("ABSOLUTE_PATH_FORBIDDEN", operation)
        self.assertFalse(self.layout.resolve(self.store.manifest_ref("step01", "audit")).exists())

    def test_disk_estimate_failure_preserves_existing_stage(self):
        a = self.publish()
        def operation():
            with self.store.publish("step01", "audit", required_bytes=sys.maxsize):
                pass
        self.error("INSUFFICIENT_DISK_SPACE", operation)
        self.assertEqual(self.store.validate(a)["generation"], 1)

    def test_cycle_and_empty_publication_rejected(self):
        a = self.store.manifest_ref("step01", "audit")
        def operation():
            with self.store.publish("step01", "audit", inputs=[a]):
                pass
        self.error("DEPENDENCY_CYCLE", operation)
        def empty():
            with self.store.publish("step01", "audit"):
                pass
        self.error("EMPTY_PUBLICATION", empty)


if __name__ == "__main__":
    unittest.main()
