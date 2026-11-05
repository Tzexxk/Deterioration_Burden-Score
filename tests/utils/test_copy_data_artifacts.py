"""Generated-file checks for configurable cross-run artifact copies."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.config import DataContext
from src.common.manifest import ManifestStore, write_json
from src.common.paths import Layout, PipelineError, read_json
from src.utils.copy_data_artifacts import copy_data_artifacts

BASELINE_OUTPUTS = (
    ("step05", "audit", "excluded_studies.csv", "outputs"),
    ("step08", "freeze", "cohort.parquet", "processed"),
    ("step09", "split", "patient_split.parquet", "processed"),
    ("step12", "export", "image_manifest.parquet", "processed"),
    ("step13", "validate", "training_package_acceptance.json", "outputs"),
)


def default_plan(source_run=1):
    return {"source_run": source_run, "target": {"stage": "step05", "phase": "baseline"},
            "items": [{"source": {"stage": stage, "phase": phase, "name": name},
                       "target": {"name": name, "root": root}} for stage, phase, name, root in BASELINE_OUTPUTS]}


def prepare_clinical_baseline(context, source_run):
    write_json(context.layout.path("project", "configs/utils/copy_data_artifacts.json"), default_plan(source_run))
    return copy_data_artifacts(context, "configs/utils/copy_data_artifacts.json")


class CopyArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text("# fixture", encoding="utf-8")
        write_json(self.root / "configs/environment/paths.local.json", {"roots": {}})
        write_json(self.root / "configs/version_run.json", {"version": 1, "run": 2})
        (self.root / "configs/data/v1_r2").mkdir(parents=True)
        self.layout = Layout(self.root)
        self.context = DataContext.load(self.layout, 1, 2)
        self.source = ManifestStore(self.layout, data=DataContext(self.layout, 1, 1))
        self.target = ManifestStore(self.layout, data=self.context)

    def release(self, omit_from_lineage=None):
        refs = []
        for stage, phase, name, root in BASELINE_OUTPUTS:
            inputs = refs if stage == "step13" else []
            if omit_from_lineage:
                inputs = [ref for ref in inputs if omit_from_lineage not in ref["path"]]
            with self.source.publish(stage, phase, inputs=inputs,
                                     data_files=[name] if root == "processed" else [],
                                     details={"checks_passed": True}) as work:
                if name.endswith(".json"):
                    write_json(work / name, {"synthetic": True, "data_run": "v1_r1"})
                else:
                    (work / name).write_bytes(("synthetic:" + name).encode())
            refs.append(self.source.manifest_ref(stage, phase))
        self.source.freeze(refs[-1])

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.root), str(caught.exception))

    def baseline(self):
        return prepare_clinical_baseline(self.context, 1)

    def test_physical_copy_routing_origin_and_historical_identity(self):
        self.release()
        result = self.baseline()
        self.assertEqual(result["copied_files"], 5)
        manifest = self.target.validate(result["manifest"])
        self.assertEqual(manifest["action"], "copied")
        self.assertEqual(manifest["scope"]["run"], 2)
        self.assertEqual(manifest["origin"]["scope"]["run"], 1)
        self.assertEqual(len(manifest["details"]["sources"]), 5)
        for stage, phase, name, root in BASELINE_OUTPUTS:
            old = self.source.validate(self.source.manifest_ref(stage, phase))
            src = self.layout.resolve(self.source.output_ref(old, name))
            ref = self.target.output_ref(manifest, name)
            dest = self.layout.resolve(ref)
            self.assertEqual(ref["root"], root)
            self.assertEqual(src.read_bytes(), dest.read_bytes())
            self.assertNotEqual(src.stat().st_ino, dest.stat().st_ino)
            self.assertFalse(dest.is_symlink())
        acceptance = self.layout.resolve(self.target.output_ref(manifest, "training_package_acceptance.json"))
        self.assertEqual(read_json(acceptance)["data_run"], "v1_r1")
        # Changing the copy cannot affect the released origin.
        original = self.layout.resolve(self.source.output_ref(old, name))
        content = original.read_bytes()
        acceptance.write_text("copy changed", encoding="utf-8")
        self.assertEqual(original.read_bytes(), content)

    def test_source_must_be_released_and_older(self):
        self.error("SOURCE_NOT_RELEASED", self.baseline)
        for run in (2, 3):
            self.error("INVALID_REUSE", lambda: prepare_clinical_baseline(self.context, run))
        self.error("INVALID_VERSION", lambda: prepare_clinical_baseline(self.context, 0))

    def test_release_identity_and_generation_are_checked(self):
        self.release()
        path = self.source.control / "release.json"
        release = read_json(path)
        write_json(path, {**release, "generation": 99})
        self.error("UPSTREAM_REBUILT", self.baseline)
        write_json(path, {**release, "scope": {**release["scope"], "run": 2}})
        self.error("SOURCE_NOT_RELEASED", self.baseline)

    def test_existing_stage_outside_accepted_graph_is_rejected(self):
        self.release(omit_from_lineage="step09")
        self.error("SOURCE_NOT_ACCEPTED", self.baseline)

    def test_missing_origin_output_is_rejected_before_publication(self):
        self.release()
        ref = self.source.manifest_ref("step09", "split")
        manifest = self.source.validate(ref)
        self.layout.resolve(self.source.output_ref(manifest, "patient_split.parquet")).unlink()
        self.error("OUTPUT_MISSING_OR_CHANGED", self.baseline)
        self.assertFalse(self.layout.resolve(self.target.manifest_ref("step05", "baseline")).exists())

    def test_failed_copy_rolls_back_and_retry_increments_only_on_success(self):
        self.release()
        result = self.baseline()
        ref = result["manifest"]
        previous = self.layout.resolve(ref).read_bytes()
        real_copy = shutil.copyfile
        copied = 0

        def fail_second(src, dest):
            nonlocal copied
            copied += 1
            if copied == 2:
                raise OSError("synthetic failure")
            return real_copy(src, dest)

        with patch("src.utils.copy_data_artifacts.shutil.copyfile", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.baseline()
        self.assertEqual(self.layout.resolve(ref).read_bytes(), previous)
        self.error("UPSTREAM_INCOMPLETE", lambda: self.target.validate(ref))
        self.assertFalse(any((self.root / "outputs/data/v1_r2/steps/step05").glob("temp/*")))
        self.assertFalse(any((self.root / "data/processed/v1_r2/steps/step05").glob("temp/*")))
        self.assertEqual(self.baseline()["generation"], 2)
        self.source.validate(self.source.manifest_ref("step13", "validate"))

    def test_source_lock_and_target_release_protect_writes(self):
        self.release()
        from src.common.manifest import file_lock
        with file_lock(self.source.control / "writer.lock"):
            self.error("CONCURRENT_RUN", self.baseline)
        with self.target.publish("step13", "validate", details={"checks_passed": True}) as work:
            (work / "accepted.txt").write_text("synthetic", encoding="utf-8")
        self.target.freeze(self.target.manifest_ref("step13", "validate"))
        self.error("RELEASE_IMMUTABLE", self.baseline)


    def run_plan(self, plan):
        write_json(self.root / "configs/utils/copy_data_artifacts.json", plan)
        return copy_data_artifacts(self.context, "configs/utils/copy_data_artifacts.json")

    def test_arbitrary_selection_nested_renaming_and_destination_phase(self):
        self.release()
        plan = default_plan()
        plan["items"] = [plan["items"][2]]
        plan["items"][0]["target"] = {"name": "copied/nested/custom.bin", "root": "processed"}
        plan["target"] = {"stage": "step11", "phase": "reuse"}
        result = self.run_plan(plan)
        manifest = self.target.validate(result["manifest"])
        self.assertEqual(result["copied_files"], 1)
        self.assertEqual(manifest["stage"], "step11")
        ref = self.target.output_ref(manifest, "copied/nested/custom.bin")
        self.assertEqual(self.layout.resolve(ref).read_bytes(), b"synthetic:patient_split.parquet")
        snapshot = self.layout.resolve(self.target.output_ref(manifest, "copy_plan.json"))
        self.assertEqual(read_json(snapshot), plan)
        self.assertEqual(manifest["configs"], [self.layout.ref("project", "configs/utils/copy_data_artifacts.json")])

    def test_same_source_can_have_two_independent_destinations(self):
        self.release()
        plan = default_plan()
        item = plan["items"][0]
        plan["items"] = [{"source": item["source"], "target": {"name": f"{n}.csv", "root": "outputs"}} for n in ("first", "second")]
        result = self.run_plan(plan)
        manifest = self.target.validate(result["manifest"])
        a, b = [self.layout.resolve(self.target.output_ref(manifest, f"{n}.csv")) for n in ("first", "second")]
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertNotEqual(a.stat().st_ino, b.stat().st_ino)

    def test_empty_reserved_duplicate_and_overlapping_destinations_rejected(self):
        for names, code in [([], "INVALID_COPY_PLAN"), (["manifest.json"], "INVALID_COPY_PLAN"),
                            (["copy_plan.json"], "INVALID_COPY_PLAN"), (["X", "x"], "COPY_TARGET_COLLISION"),
                            (["a", "a/b"], "COPY_TARGET_COLLISION"), (["a/b", "a"], "COPY_TARGET_COLLISION")]:
            plan = default_plan()
            source = plan["items"][0]["source"]
            plan["items"] = [{"source": source, "target": {"name": name, "root": "outputs"}} for name in names]
            self.error(code, lambda: self.run_plan(plan))
        self.assertFalse(self.layout.resolve(self.target.manifest_ref("step05", "baseline")).exists())

    def test_path_escape_and_invalid_root_rejected_before_copy(self):
        for name in ("../escape.bin", "../v1_r1/escape.bin"):
            plan = default_plan()
            plan["items"][0]["target"]["name"] = name
            self.error("ABSOLUTE_PATH_FORBIDDEN", lambda: self.run_plan(plan))
        plan = default_plan()
        plan["items"][0]["target"]["root"] = "project"
        self.error("INVALID_COPY_PLAN", lambda: self.run_plan(plan))

    def test_unregistered_file_and_missing_current_config_are_rejected(self):
        self.release()
        plan = default_plan()
        plan["items"][0]["source"]["name"] = "unregistered.csv"
        self.error("MISSING_UPSTREAM_OUTPUT", lambda: self.run_plan(plan))
        self.error("MISSING_FILE", lambda: copy_data_artifacts(self.context, "configs/utils/missing.json"))
        self.error("INVALID_PATH", lambda: copy_data_artifacts(self.context, "../utils/copy_data_artifacts.json"))

    def test_wrong_active_run_and_missing_cli_arguments_rejected(self):
        write_json(self.root / "configs/version_run.json", {"version": 1, "run": 1})
        self.error("VERSION_MISMATCH", lambda: DataContext.load(self.layout, 1, 2))
        import subprocess
        import sys
        entry = Path(__file__).resolve().parents[2] / "scripts/utils/copy_data_artifacts.py"
        process = subprocess.run([sys.executable, "-B", str(entry), "--version", "1", "--run", "2"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn("--config", process.stderr)


    def test_shared_tool_config_is_reusable_across_target_runs(self):
        self.release()
        result = self.baseline()
        previous = self.layout.resolve(result["manifest"]).read_bytes()
        config = self.root / "configs/utils/copy_data_artifacts.json"
        original = config.read_bytes()
        write_json(self.root / "configs/version_run.json", {"version": 1, "run": 3})
        (self.root / "configs/data/v1_r3").mkdir(parents=True)
        context = DataContext.load(self.layout, 1, 3)
        later = copy_data_artifacts(context, "configs/utils/copy_data_artifacts.json")
        store = ManifestStore(self.layout, data=context)
        self.assertEqual(store.validate(later["manifest"])["scope"]["run"], 3)
        self.assertEqual(config.read_bytes(), original)
        self.assertEqual(self.layout.resolve(result["manifest"]).read_bytes(), previous)

    def test_data_and_template_paths_cannot_be_used_as_tool_configs(self):
        for path in ("configs/data/v1_r2/copy.json", "configs/utils.template/copy.json", "configs/models/copy.json"):
            write_json(self.root / path, default_plan())
            self.error("INVALID_UTILITY_CONFIG", lambda: copy_data_artifacts(self.context, path))
        self.error("INVALID_UTILITY_CONFIG", lambda: copy_data_artifacts(self.context, "configs/utils/copy.txt"))

    def test_manifest_tool_config_route_does_not_relax_data_config_isolation(self):
        config = "configs/utils/copy_data_artifacts.json"
        write_json(self.root / config, default_plan())
        ref = self.layout.ref("project", config)

        def publish(configs=(), utility_configs=()):
            with self.target.publish("step01", "audit", configs=configs, utility_configs=utility_configs) as work:
                (work / "test.txt").write_text("synthetic", encoding="utf-8")

        self.error("CROSS_VERSION_CONFIG", lambda: publish(configs=[ref]))
        other = self.context.config_ref("cohort/test.json")
        write_json(self.layout.resolve(other), {"synthetic": True})
        self.error("INVALID_UTILITY_CONFIG", lambda: publish(utility_configs=[other]))

    def test_utils_template_is_discovered_and_preserved(self):
        from src.utils.initialize_configs import initialize_configs
        project = self.root / "fresh"
        template = project / "configs/utils.template/copy_data_artifacts.json"
        write_json(template, default_plan())
        (project / "pyproject.toml").write_text("# synthetic project", encoding="utf-8")
        initialize_configs(project)
        actual = project / "configs/utils/copy_data_artifacts.json"
        self.assertEqual(actual.read_bytes(), template.read_bytes())
        self.assertTrue(template.is_file())


if __name__ == "__main__":
    unittest.main()
