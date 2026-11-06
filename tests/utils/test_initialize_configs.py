"""Initialization uses only tiny templates in independent temporary projects."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.paths import PipelineError
from src.utils.initialize_configs import initialize_configs

PROJECT = Path(__file__).resolve().parents[2]


class InitializeConfigTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT / "outputs/config_setup_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
        self.configs = self.root / "configs"
        for name in ("data.template/strict/ehr", "environment.template"):
            (self.configs / name).mkdir(parents=True)
        self.data = self.configs / "data.template/strict/ehr/raw_audit.json"
        self.data.write_text('{"raw_audit": {}}\n', encoding="utf-8")
        (self.configs / "environment.template/paths.local.json").write_text('{"roots": {}}\n', encoding="utf-8")
        (self.configs / "version_run.template.json").write_text('{"version": 1, "run": 1}\n', encoding="utf-8")

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.root), str(caught.exception))

    def test_discovers_directory_and_root_file_templates_and_preserves_them(self):
        created = initialize_configs(self.root)
        self.assertEqual(set(created), {"configs/data/v1_r1", "configs/environment", "configs/version_run.json"})
        self.assertEqual((self.configs / "data/v1_r1/ehr/raw_audit.json").read_bytes(), self.data.read_bytes())
        self.assertEqual(json.loads((self.configs / "version_run.json").read_text()), {"version": 1, "run": 1})
        self.assertTrue(self.data.exists())
        self.assertTrue((self.configs / "version_run.template.json").exists())
        self.assertFalse((self.configs / "temp").exists())

    def test_preserves_shared_configuration_and_updates_selection(self):
        actual = self.configs / "environment"
        actual.mkdir()
        (actual / "keep.txt").write_text("keep", encoding="utf-8")
        initialize_configs(self.root)
        initialize_configs(self.root, version=2, run=1)
        initialize_configs(self.root, version=1, run=2)
        self.assertEqual((actual / "keep.txt").read_text(), "keep")
        self.assertEqual(json.loads((self.configs / "version_run.json").read_text()), {"version": 1, "run": 2})
        self.error("CONFIG_ALREADY_EXISTS", lambda: initialize_configs(self.root))
        self.assertFalse((self.configs / "temp").exists())

    def test_rejects_skipped_version_or_run(self):
        self.error("NONCONSECUTIVE_DATA_RUN", lambda: initialize_configs(self.root, run=2))
        initialize_configs(self.root)
        for version, run in ((1, 3), (2, 2), (3, 1)):
            self.error("NONCONSECUTIVE_DATA_RUN", lambda: initialize_configs(self.root, version=version, run=run))
        self.assertEqual(json.loads((self.configs / "version_run.json").read_text()), {"version": 1, "run": 1})

    def test_master_initialization_records_source_and_keeps_shared_plan(self):
        template = self.configs / "data.template/multiple_size/imaging"
        template.mkdir(parents=True)
        (template / "processing.json").write_text(
            '{"processing_version":"master","reviews_path":"imaging/reviews/image_qc.csv"}', encoding="utf-8")
        shared = self.configs / "utils.template"
        shared.mkdir()
        (shared / "reuse_master_inputs.json").write_text('{"source_run":2}', encoding="utf-8")
        initialize_configs(self.root)
        self.error("INVALID_REUSE", lambda: initialize_configs(self.root, template="multiple_size", run=2))
        with patch("src.utils.initialize_configs._master_settings",
                   return_value=({"size": 1024, "parent_run": 1}, {"source_run": 1})):
            initialize_configs(self.root, template="multiple_size", run=2, source_run=1)
        self.assertEqual(json.loads((self.configs / "utils/v1_r2/reuse_master_inputs.json").read_text()),
                         {"source_run": 1})
        self.assertEqual(json.loads((self.configs / "utils/reuse_master_inputs.json").read_text()),
                         {"source_run": 2})
        self.assertEqual(json.loads((self.configs / "data/v1_r2/imaging/processing.json").read_text())["parent_run"], 1)

    def test_discovers_future_templates_without_a_hardcoded_list(self):
        extra = self.configs / "extra.template"
        extra.mkdir()
        (extra / "settings.json").write_text('{}', encoding="utf-8")
        (self.configs / "other.template.json").write_text('{}', encoding="utf-8")
        created = initialize_configs(self.root)
        self.assertIn("configs/extra", created)
        self.assertIn("configs/other.json", created)

    def test_publish_failure_rolls_back_all_owned_targets(self):
        from src.utils.initialize_configs import _publish
        calls = 0
        def fail_second(staged, target, owned):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic publication failure")
            _publish(staged, target, owned)
        with patch("src.utils.initialize_configs._publish", fail_second), self.assertRaises(OSError):
            initialize_configs(self.root)
        for name in ("data", "environment", "version_run.json", "temp"):
            self.assertFalse((self.configs / name).exists())
        self.assertTrue(self.data.exists())

    def test_staging_failure_publishes_nothing(self):
        with patch("src.utils.initialize_configs.shutil.copytree", side_effect=OSError("synthetic copy failure")), self.assertRaises(OSError):
            initialize_configs(self.root)
        self.assertFalse((self.configs / "data").exists())
        self.assertFalse((self.configs / "version_run.json").exists())
        self.assertFalse((self.configs / "temp").exists())

    def test_target_race_does_not_overwrite_existing_file(self):
        from src.utils.initialize_configs import _publish
        def racing(staged, target, owned):
            if target.name == "version_run.json":
                target.write_text('{"keep": true}', encoding="utf-8")
            _publish(staged, target, owned)
        with patch("src.utils.initialize_configs._publish", racing), self.assertRaises(FileExistsError):
            initialize_configs(self.root)
        self.assertFalse((self.configs / "data").exists())
        self.assertFalse((self.configs / "environment").exists())
        self.assertEqual(json.loads((self.configs / "version_run.json").read_text()), {"keep": True})

    def test_bad_json_or_target_collision_is_rejected_before_publication(self):
        self.data.write_text('{broken', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            initialize_configs(self.root)
        self.assertFalse((self.configs / "data").exists())
        self.data.write_text('{}', encoding="utf-8")
        (self.configs / "version_run.json.template").mkdir()
        self.error("TEMPLATE_TARGET_COLLISION", lambda: initialize_configs(self.root))

    def test_absent_templates_are_rejected(self):
        self.data.unlink()
        for path in sorted(self.configs.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.error("NO_TEMPLATES", lambda: initialize_configs(self.root))

    def test_cli_works_without_existing_local_configuration_from_other_cwd(self):
        shutil.copytree(PROJECT / "src", self.root / "src")
        scripts = self.root / "scripts/utils"
        scripts.mkdir(parents=True)
        script = scripts / "initialize_configs.py"
        shutil.copyfile(PROJECT / "scripts/utils/initialize_configs.py", script)
        result = subprocess.run([sys.executable, "-B", str(script)], cwd=self.root.parent,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "initialized")
        self.assertNotIn(str(self.root), result.stdout + result.stderr)
