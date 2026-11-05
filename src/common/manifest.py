"""Generation-bound publication, rollback and explicit physical reuse."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import load_utility_config
from .csv_io import validate_output_text
from .paths import (
    PipelineError,
    identifier,
    portable,
    positive,
    read_json,
    relative,
    require,
)


def write_json(path, value):
    portable(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _rename(source, destination):
    """Retry bounded Windows sharing/access denials without changing the target."""
    for attempt in range(5):
        try:
            source.rename(destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


@contextmanager
def file_lock(path):
    """OS releases this lock after a crash; the harmless lock file remains."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise PipelineError("CONCURRENT_RUN", "Another writer owns this data run or model experiment") from None
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class ManifestStore:
    def __init__(self, layout, *, data=None, family=None, experiment=None, model_stage=None, model=None):
        self.layout = layout
        self.data = data
        if data is not None:
            require(data.layout.project == layout.project and all(value is None for value in (family, experiment, model_stage, model)),
                    "INVALID_SCOPE", "Data and model scopes must be separate")
            self.scope = {"kind": "data", "version": data.version, "run": data.run}
            self.prefix = f"data/{data.key}"
        else:
            require(family in ("ehr", "image"), "INVALID_SCOPE", "Select ehr or image model family")
            self.scope = {"kind": "model", "family": family, "experiment": identifier(experiment)}
            require((model_stage is None) == (model is None), "INVALID_SCOPE", "Model stage and model must be supplied together")
            if model_stage is not None:
                self.scope.update(model_stage=model_stage, model=model)
            self.prefix = self._model_prefix(self.scope)
        self.control = layout.path("outputs", f"{self.prefix}/.control")

    @staticmethod
    def _model_prefix(scope):
        family = scope.get("family")
        require(family in ("ehr", "image"), "INVALID_SCOPE", "Unknown model modality")
        experiment = identifier(scope.get("experiment"))
        if "model_stage" not in scope and "model" not in scope:
            return f"models/{family}/{experiment}"
        stage = scope.get("model_stage")
        require(stage in ("train", "assessment", "test"), "INVALID_SCOPE", "Unknown model stage")
        return f"models/{stage}/{family}/{identifier(scope.get('model'))}/{experiment}"

    def _ref(self, path):
        return self.layout.ref("outputs", path.relative_to(self.layout.roots["outputs"]).as_posix())

    def _owned(self, ref):
        target = self.layout.resolve(ref)
        if self.data:
            allowed = {"processed": self.data.key, "outputs": self.prefix}
        else:
            allowed = {"outputs": self.prefix}
        require(ref["root"] in allowed, "INVALID_DESTINATION", "Stage destination must use its writable root")
        base = self.layout.path(ref["root"], allowed[ref["root"]])
        require(target != base and target.is_relative_to(base), "INVALID_DESTINATION", "Stage destination must be inside its run")
        return target

    def _remove_temp(self, ref):
        target = self._owned(ref)
        require(target.parent.name == "temp", "INVALID_RECOVERY", "Only an owned temporary directory may be removed")
        if target.exists():
            shutil.rmtree(target)
        if target.parent.exists() and not any(target.parent.iterdir()):
            target.parent.rmdir()

    def _recover_unlocked(self):
        for path in sorted(self.control.glob("*.state.json")):
            state = portable(read_json(path))
            require(state["scope"] == self.scope, "INVALID_RECOVERY", "Recovery scope mismatch")
            if state["status"] == "running":
                for target in reversed(state["targets"]):
                    dest = self._owned(target["destination"])
                    backup = self._owned(target["backup"])
                    if backup.exists():
                        if dest.exists():
                            shutil.rmtree(dest)
                        _rename(backup, dest)
                    elif not target["existed"] and dest.exists():
                        shutil.rmtree(dest)
                    elif target["existed"]:
                        require(dest.is_dir(), "INVALID_RECOVERY", "Previous output is missing")
                state["status"] = "failed"
                state["error_type"] = "InterruptedPublication"
                write_json(path, state)
            for target in state.get("targets", []):
                for key in ("temp", "backup"):
                    self._remove_temp(target[key])

    def recover(self):
        with file_lock(self.control / "writer.lock"):
            self._recover_unlocked()

    def manifest_ref(self, stage, phase, *, area=None, destination=None):
        identifier(stage)
        identifier(phase)
        if self.data:
            require(stage in {f"step{i:02d}" for i in range(1, 14)}, "INVALID_STAGE", "Data stages are step01 through step13")
        area = area or "outputs"
        require(area == "outputs", "INVALID_DESTINATION", "Manifests belong to outputs")
        base = self.data.key if area == "processed" and self.data else self.prefix
        directory = destination or f"{base}/steps/{stage}/{phase}"
        ref = self.layout.ref(area, f"{relative(directory)}/manifest.json")
        self._owned(ref)
        return ref

    def validate(self, ref, *, seen=None):
        path = self.layout.resolve(ref)
        manifest = portable(read_json(path))
        require(manifest.get("schema") == 2 and manifest.get("status") == "complete",
                "UPSTREAM_INCOMPLETE", "Manifest is not a completed publication")
        require(manifest.get("self") == ref, "INVALID_MANIFEST", "Manifest location does not match its reference")
        require(ref["root"] == "outputs", "INVALID_MANIFEST", "Manifests belong to outputs")
        positive(manifest.get("generation"))
        visited = set() if seen is None else set(seen)
        key = (ref["root"], ref["path"])
        require(key not in visited, "DEPENDENCY_CYCLE", "Dependency cycle detected")
        visited.add(key)
        scope = manifest["scope"]
        if self.data:
            require(scope.get("kind") == "data" and scope.get("version") == self.data.version,
                    "VERSION_MISMATCH", "Upstream raw-data version differs")
            require(scope.get("run", 0) <= self.data.run, "UPSTREAM_FROM_NEWER_RUN", "Upstream is from a newer run")
            require(scope.get("run") == self.data.run, "UNDECLARED_REUSE", "Cross-run inputs require explicit copying")
        state_path = self.layout.resolve(manifest["state"])
        identifier(manifest["stage"])
        identifier(manifest["phase"])
        if scope.get("kind") == "data":
            positive(scope.get("version"))
            positive(scope.get("run"))
            prefix = f"data/v{scope['version']}_r{scope['run']}"
        else:
            require(scope.get("kind") == "model", "INVALID_SCOPE", "Unknown upstream scope")
            prefix = self._model_prefix(scope)
            if "model_stage" in scope:
                require(ref["path"].startswith(prefix + "/"), "INVALID_MANIFEST", "Model manifest is outside its experiment")
        expected_control = self.layout.path("outputs", f"{prefix}/.control")
        expected_state = expected_control / f"{manifest['stage']}.{manifest['phase']}.state.json"
        require(state_path == expected_state, "INVALID_MANIFEST", "Stage state location mismatch")
        if self.data is None and scope["kind"] == "data":
            require((expected_control / "release.json").is_file(), "SOURCE_NOT_RELEASED", "Model input must come from a frozen data release")
            release = portable(read_json(expected_control / "release.json"))
            require(release.get("scope") == scope and release.get("status") == "released", "SOURCE_NOT_RELEASED", "Model input must come from a frozen data release")
        state = portable(read_json(state_path))
        require(state.get("scope") == scope and state.get("status") == "complete",
                "UPSTREAM_INCOMPLETE", "Upstream is running, failed or awaiting recovery")
        require(state.get("generation") == manifest["generation"], "UPSTREAM_REBUILT", "Upstream publication has changed")
        names = set()
        for item in manifest["outputs"]:
            require(item["name"] not in names, "INVALID_MANIFEST", "Output names must be unique")
            names.add(item["name"])
            output = self._output_path(manifest, item)
            require(output.is_file() and output.stat().st_size == item["bytes"], "OUTPUT_MISSING_OR_CHANGED", "Published output is missing or its size changed")
        for item in manifest["inputs"]:
            upstream = self.validate(item["manifest"], seen=visited)
            require(upstream["generation"] == item["generation"], "UPSTREAM_REBUILT", "Input generation differs from its recorded generation")
        return manifest

    def _output_path(self, manifest, item):
        name = relative(item["name"])
        root = item["root"]
        if root == "processed":
            scope = manifest["scope"]
            require(scope["kind"] == "data", "INVALID_MANIFEST", "Only data publications use processed")
            base = f"v{scope['version']}_r{scope['run']}/steps/{manifest['stage']}/{manifest['phase']}"
            expected = self.layout.path(root, f"{base}/{name}")
        else:
            require(root == "outputs", "INVALID_MANIFEST", "Output root must be processed or outputs")
            expected = self.layout.resolve(manifest["self"]).parent / name
        output = self.layout.resolve({"root": root, "path": item["path"]})
        require(output == expected, "INVALID_MANIFEST", "Output location does not match its stage and name")
        return output

    def output_ref(self, manifest, name):
        """Locate a declared output by logical name, independently of its root."""
        found = [item for item in manifest["outputs"] if item["name"] == relative(name)]
        require(len(found) == 1, "MISSING_UPSTREAM_OUTPUT", "Required output is not uniquely registered")
        self._output_path(manifest, found[0])
        return {key: found[0][key] for key in ("root", "path")}

    def _target(self, root, directory, token):
        dest = self.layout.path(root, directory)
        def ref(path):
            value = self.layout.ref(root, path.relative_to(self.layout.roots[root]).as_posix())
            self._owned(value)
            return value
        return {"destination": ref(dest),
                "temp": ref(dest.parent / "temp" / (dest.name + "." + token)),
                "backup": ref(dest.parent / "temp" / (dest.name + "." + token + ".backup")),
                "existed": dest.exists()}

    @contextmanager
    def publish(self, stage, phase, *, inputs=(), configs=(), area=None,
                destination=None, action="generated", origin=None, details=None, required_bytes=0,
                data_files=(), data_directories=(), utility_configs=()):
        ref = self.manifest_ref(stage, phase, area=area, destination=destination)
        dest = self.layout.resolve(ref).parent
        state_path = self.control / f"{stage}.{phase}.state.json"
        data_files = tuple(relative(name) for name in data_files)
        data_directories = tuple(relative(name) for name in data_directories)
        require(len(set(data_directories)) == len(data_directories) and "manifest.json" not in data_directories,
                "INVALID_OUTPUT", "Data directories must be unique and cannot be manifest.json")
        require(len(set(data_files)) == len(data_files) and "manifest.json" not in data_files,
                "INVALID_OUTPUT", "Data output names must be unique and cannot be manifest.json")
        require(not (data_files or data_directories) or self.data is not None, "INVALID_SCOPE", "Model outputs belong to outputs")
        with file_lock(self.control / "writer.lock"):
            self._recover_unlocked()
            require(not (self.control / "release.json").exists(), "RELEASE_IMMUTABLE", "Released data run cannot be overwritten")
            require(type(required_bytes) is int and required_bytes >= 0, "INVALID_SIZE_ESTIMATE", "Expected a nonnegative byte estimate")
            require(action in ("generated", "copied") and (action == "copied") == (origin is not None),
                    "INVALID_ACTION", "Copied publication requires explicit origin")
            bound = []
            for item in inputs:
                require(item != ref, "DEPENDENCY_CYCLE", "A stage cannot consume itself")
                source = self.validate(item)
                bound.append({"manifest": item, "generation": source["generation"]})
            for item in configs:
                if self.data:
                    self.data.check_config_ref(item)
                else:
                    config_path = self.layout.resolve(item)
                    require(item["root"] == "project" and config_path.is_relative_to(self.layout.path("project", "configs/models")),
                            "INVALID_MODEL_CONFIG", "Model configuration must belong to configs/models")
                    portable(config_path.read_text(encoding="utf-8-sig"))
            for item in utility_configs:
                self.layout.resolve(item)
                require(item["root"] == "project", "INVALID_UTILITY_CONFIG", "Utility configuration must use the project root")
                load_utility_config(self.layout, item["path"])
            previous = read_json(dest / "manifest.json") if (dest / "manifest.json").exists() else {}
            require(not dest.exists() or bool(previous), "UNMANAGED_DESTINATION", "Refusing to overwrite an unmanaged directory")
            if previous:
                require(previous.get("scope") == self.scope and previous.get("stage") == stage and previous.get("phase") == phase,
                        "INVALID_DESTINATION", "Destination belongs to another stage")
                require(previous.get("schema") == 2, "INVALID_MANIFEST", "Output layout must be migrated before execution")
                require(self.data is not None, "EXPERIMENT_EXISTS", "Model publication already exists; use its explicit resume interface")
            generation = previous.get("generation", 0) + 1
            token = uuid.uuid4().hex
            report_target = self._target("outputs", dest.relative_to(self.layout.roots["outputs"]).as_posix(), token)
            targets = []
            if data_files or data_directories:
                target = self._target("processed", f"{self.data.key}/steps/{stage}/{phase}", token)
                prior_data = [item for item in previous.get("outputs", []) if item["root"] == "processed"]
                require(not target["existed"] or bool(prior_data), "UNMANAGED_DESTINATION", "Data directory is not managed by this stage")
                targets.append(target)
            else:
                require(not any(item["root"] == "processed" for item in previous.get("outputs", [])),
                        "INVALID_OUTPUT", "Cannot silently remove the data destination")
            targets.append(report_target)
            # Check both volumes conservatively; outputs also stages data before routing.
            for target in targets:
                existing = self._owned(target["destination"]).parent
                while not existing.exists():
                    existing = existing.parent
                require(shutil.disk_usage(existing).free >= required_bytes, "INSUFFICIENT_DISK_SPACE", "Insufficient space for staged output; no automatic deletion or relocation")
            journal = {"scope": self.scope, "stage": stage, "phase": phase, "status": "running",
                       "previous_generation": generation - 1, "targets": targets}
            write_json(state_path, journal)
            committed = False
            try:
                for target in targets:
                    self._owned(target["temp"]).mkdir(parents=True)
                temp = self._owned(report_target["temp"])
                yield temp
                require(not (temp / "manifest.json").exists(), "RESERVED_FILENAME", "manifest.json belongs to the publisher")
                entries = []
                for output in sorted(temp.rglob("*")):
                    require(not output.is_symlink(), "INVALID_OUTPUT", "Symbolic links are not published")
                    if output.is_file():
                        if output.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md"):
                            validate_output_text(output)
                        name = output.relative_to(temp).as_posix()
                        is_data = name in data_files or any(name.startswith(folder + "/") for folder in data_directories)
                        target = targets[0] if is_data else report_target
                        final = self._owned(target["destination"]) / name
                        root = target["destination"]["root"]
                        entries.append({"name": name, "root": root,
                                        "path": final.relative_to(self.layout.roots[root]).as_posix(),
                                        "bytes": output.stat().st_size})
                require(bool(entries), "EMPTY_PUBLICATION", "A stage must publish at least one output")
                require(set(data_files) <= {item["name"] for item in entries}, "MISSING_DATA_OUTPUT", "A declared data output was not produced")
                require(all((temp / folder).is_dir() and not (temp / folder).is_symlink() for folder in data_directories),
                        "MISSING_DATA_OUTPUT", "A declared data directory was not produced")
                for name in (item["name"] for item in entries if item["root"] == "processed"):
                    target = self._owned(targets[0]["temp"]) / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(temp / name, target)
                    (temp / name).unlink()
                for item in bound:
                    require(self.validate(item["manifest"])["generation"] == item["generation"], "UPSTREAM_REBUILT", "Input changed during execution")
                manifest = {"schema": 2, "self": ref, "scope": self.scope, "stage": stage, "phase": phase,
                            "generation": generation, "status": "complete", "state": self._ref(state_path),
                            "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": bound,
                            "configs": list(configs) + list(utility_configs), "outputs": entries, "action": action,
                            "origin": origin, "details": details or {}}
                write_json(temp / "manifest.json", manifest)
                # Data first, manifest last; consumers require the shared complete state.
                for target in targets:
                    final = self._owned(target["destination"])
                    if final.exists():
                        _rename(final, self._owned(target["backup"]))
                    _rename(self._owned(target["temp"]), final)
                journal.update(status="complete", generation=generation)
                write_json(state_path, journal)
                committed = True
            except BaseException:
                self._recover_unlocked()
                raise
            finally:
                if committed:
                    for target in targets:
                        for key in ("temp", "backup"):
                            self._remove_temp(target[key])

    def freeze(self, acceptance_ref):
        require(self.data is not None, "INVALID_SCOPE", "Only data runs have a release freeze")
        with file_lock(self.control / "writer.lock"):
            self._recover_unlocked()
            manifest = self.validate(acceptance_ref)
            require(manifest["stage"] == "step13" and manifest["details"].get("checks_passed") is True,
                    "ACCEPTANCE_REQUIRED", "A successful Step13 acceptance is required")
            release = self.control / "release.json"
            require(not release.exists(), "RELEASE_IMMUTABLE", "Data run is already released")
            write_json(release, {"scope": self.scope, "status": "released", "acceptance": acceptance_ref,
                                 "generation": manifest["generation"]})

    def copy(self, stage, phase, *, source_store, source, inputs=(), configs=(), destination=None):
        require(self.data is not None and source_store.data is not None, "INVALID_REUSE", "Copy reuse requires data scopes")
        require(self.data.version == source_store.data.version, "VERSION_MISMATCH", "Reuse requires the same raw-data version")
        require(source_store.data.run < self.data.run, "INVALID_REUSE", "Copy source must be an older run")
        with file_lock(source_store.control / "writer.lock"):
            source_store._recover_unlocked()
            require((source_store.control / "release.json").is_file(), "SOURCE_NOT_RELEASED", "Copy source must be a frozen release")
            old = source_store.validate(source)
            origin = {"manifest": source, "scope": old["scope"], "generation": old["generation"]}
            data_files = [item["name"] for item in old["outputs"] if item["root"] == "processed"]
            with self.publish(stage, phase, inputs=inputs, configs=configs, destination=destination, action="copied", origin=origin,
                              data_files=data_files, required_bytes=sum(item["bytes"] for item in old["outputs"]) + 8192) as work:
                for item in old["outputs"]:
                    src = source_store._output_path(old, item)
                    target = work / relative(item["name"])
                    require(not src.is_symlink(), "INVALID_REUSE", "Copy source cannot be a symbolic link")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, target)
        return self.manifest_ref(stage, phase, destination=destination)
