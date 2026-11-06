"""Create local configuration from templates without consuming the templates."""

from __future__ import annotations

import json
import re
import shutil
import stat
from pathlib import Path
from tempfile import mkdtemp

from src.common.manifest import file_lock, write_json
from src.common.paths import Layout, portable, positive, read_json, require
from src.models.released_data import ReleasedData


def _linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _discover(project):
    configs = project / "configs"
    require(configs.is_dir() and not _linked(configs), "INVALID_TEMPLATE_ROOT", "Expected a real configs directory")
    entries = []
    for source in sorted(configs.iterdir()):
        name = source.name
        if name.endswith(".template"):
            require(source.is_dir(), "INVALID_TEMPLATE", "A .template entry must be a directory")
            target_name = name.removesuffix(".template")
        elif name.endswith(".template.json"):
            require(source.is_file(), "INVALID_TEMPLATE", "A .template.json entry must be a file")
            target_name = name.removesuffix(".template.json") + ".json"
        else:
            continue
        require(target_name and not target_name.startswith(".") and target_name != ".json",
                "INVALID_TEMPLATE", "Template has no usable target name")
        require(not _linked(source), "INVALID_TEMPLATE", "Linked templates are not supported")
        targets = [source, *source.rglob("*")] if source.is_dir() else [source]
        for path in targets:
            require(not _linked(path) and path.resolve().is_relative_to(configs.resolve()),
                    "INVALID_TEMPLATE", "Template entry escaped configs or contains a link")
            if path.is_file():
                require(path.suffix.lower() in (".json", ".csv", ".tsv", ".md", ".txt"),
                        "INVALID_TEMPLATE", "Templates must be text configuration or documentation")
                text = path.read_text(encoding="utf-8-sig")
                portable(text)
                if path.suffix.lower() == ".json":
                    json.loads(text)
        entries.append((source, configs / target_name))
    require(bool(entries), "NO_TEMPLATES", "No configuration templates found")
    names = [target.name.casefold() for _, target in entries]
    require(len(names) == len(set(names)), "TEMPLATE_TARGET_COLLISION", "Templates map to the same target")
    return entries


def _preflight(project, entries):
    conflicts = [target.relative_to(project).as_posix() for _, target in entries
                 if target.exists() or target.is_symlink()]
    require(not conflicts, "CONFIG_ALREADY_EXISTS", "Configuration targets already exist: " + ", ".join(conflicts))


def _publish(staged, target, owned):
    # Exclusive creation also protects against a non-tool writer appearing after preflight.
    if staged.is_dir():
        target.mkdir()
        owned.append(target)
        shutil.copytree(staged, target, dirs_exist_ok=True)
    else:
        with target.open("xb") as destination:
            owned.append(target)
            with staged.open("rb") as source:
                shutil.copyfileobj(source, destination)


def _remove_owned(path, configs):
    resolved = path.resolve()
    require(resolved != configs and resolved.is_relative_to(configs) and not _linked(path),
            "INVALID_CLEANUP", "Refusing to clean a path outside this initialization")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


TEMPLATES = ("lenient", "strict", "multiple_size")


def _run_target(configs, version, run):
    data = configs / "data"
    if data.exists() or data.is_symlink():
        require(data.is_dir() and not _linked(data), "INVALID_CONFIG_ROOT", "Expected a real configs/data directory")
    target = data / f"v{version}_r{run}"
    require(not target.exists() and not target.is_symlink(), "CONFIG_ALREADY_EXISTS",
            f"Configuration already exists for {target.name}")
    versions = {}
    if data.is_dir():
        for folder in data.iterdir():
            match = re.fullmatch(r"v([1-9][0-9]*)_r([1-9][0-9]*)", folder.name)
            if match:
                require(folder.is_dir() and not _linked(folder), "INVALID_CONFIG_ROOT",
                        "Expected a real data run directory")
                v, r = map(int, match.groups())
                versions.setdefault(v, set()).add(r)
    if version in versions:
        valid = run == max(versions[version]) + 1 and len(versions[version]) == run - 1
    else:
        valid = version == len(versions) + 1 == max(versions, default=0) + 1 and run == 1
    require(valid, "NONCONSECUTIVE_DATA_RUN", "Choose the next run in an existing version or run 1 in the next version")
    return target


def _master_settings(project, configs, version, run, source_run):
    positive(source_run)
    require(source_run < run, "INVALID_REUSE", "Select an older source run of the same data version")
    release = ReleasedData.load(Layout(project), f"v{version}_r{source_run}")
    metadata = read_json(release.package_file("dataset_metadata.json"))
    require(metadata.get("image", {}).get("shape") == [224, 224]
            and metadata.get("cross_validation", {}).get("n_folds") == 5,
            "INVALID_MASTER_SOURCE", "The master source must be a frozen 224-pixel five-fold package")
    parent = release.store.validate(release.store.manifest_ref("step07", "prepare"))
    snapshot = read_json(release.store.layout.resolve(release.store.output_ref(parent, "effective_configuration.json")))
    settings = dict(snapshot["processing"], size=1024, parent_run=source_run)
    plan = read_json(configs / "utils.template/reuse_master_inputs.json")
    plan["source_run"] = source_run
    return settings, plan


def initialize_configs(project=None, *, template="strict", version=1, run=1, source_run=None):
    project = Path(project or Path(__file__).resolve().parents[2]).resolve()
    require((project / "pyproject.toml").is_file(), "PROJECT_NOT_FOUND", "Project marker missing")
    require(template in TEMPLATES, "INVALID_TEMPLATE", "Choose lenient, strict or multiple_size")
    positive(version)
    positive(run)
    require((source_run is not None) == (template == "multiple_size"), "INVALID_REUSE",
            "Specify --source-run only with multiple_size")
    configs = project / "configs"
    require(configs.is_dir() and not _linked(configs), "INVALID_TEMPLATE_ROOT", "Expected a real configs directory")
    with file_lock(configs / ".initialize.lock"):
        discovered = _discover(project)
        target = _run_target(configs, version, run)
        selection = configs / "version_run.json"
        require(not selection.is_symlink() and (not selection.exists() or
                (selection.is_file() and not _linked(selection))), "INVALID_VERSION_CONFIG",
                "Expected a real version_run.json file")
        previous_selection = selection.read_bytes() if selection.is_file() else None
        source = configs / "data.template" / template
        require(source.is_dir(), "INVALID_TEMPLATE", "Selected data template is missing")
        common = []
        for template_path, local in discovered:
            if local.name in ("data", "version_run.json"):
                continue
            if local.exists() or local.is_symlink():
                require(not _linked(local) and local.is_dir() == template_path.is_dir(),
                        "INVALID_CONFIG_ROOT", "Existing shared configuration has the wrong type or is linked")
            else:
                common.append((template_path, local))
        entries = [*common, (source, target)]
        needed = sum(path.stat().st_size for source, _ in entries
                     for path in ([source] if source.is_file() else source.rglob("*")) if path.is_file())
        require(shutil.disk_usage(configs).free >= needed * 2 + 65536,
                "INSUFFICIENT_DISK_SPACE", "Insufficient space to stage and initialize configurations")
        parent = configs / "temp"
        require(not parent.is_symlink() and (not parent.exists() or not _linked(parent)),
                "INVALID_TEMPLATE_ROOT", "Configuration temp directory must not be linked")
        master = _master_settings(project, configs, version, run, source_run) if source_run is not None else None
        parent.mkdir(exist_ok=True)
        temporary = Path(mkdtemp(prefix="initialize_", dir=parent)).resolve()
        require(temporary.is_relative_to(configs), "PATH_ESCAPE", "Initialization temp escaped configs")
        owned = []
        try:
            staged_entries = []
            for index, (source, destination) in enumerate(entries):
                staged = temporary / str(index)
                if source.is_dir():
                    shutil.copytree(source, staged)
                else:
                    shutil.copyfile(source, staged)
                if destination == target:
                    endpoint = staged / "cohort/reviews/endpoint_review.json"
                    if endpoint.is_file():
                        write_json(endpoint, dict(read_json(endpoint), version=version, run=run))
                    if master:
                        settings, plan = master
                        template_settings = read_json(staged / "imaging/processing.json")
                        settings.update(processing_version=template_settings["processing_version"],
                                        reviews_path=template_settings["reviews_path"])
                        write_json(staged / "imaging/processing.json", settings)
                staged_entries.append((staged, destination))
            if master:
                plan_file = temporary / "reuse_master_inputs.json"
                write_json(plan_file, plan)
                staged_entries.append((plan_file, configs / "utils" / target.name / plan_file.name))
            selector = temporary / selection.name
            write_json(selector, {"version": version, "run": run})
            _preflight(project, staged_entries)
            for staged, destination in staged_entries:
                if not destination.parent.exists():
                    destination.parent.mkdir()
                    owned.append(destination.parent)
                require(not _linked(destination.parent), "INVALID_CONFIG_ROOT", "Local configuration parent is linked")
                _publish(staged, destination, owned)
            if previous_selection is not None:
                require(selection.is_file() and not _linked(selection) and selection.read_bytes() == previous_selection,
                        "VERSION_CONFIG_CHANGED", "version_run.json changed during initialization")
                selector.replace(selection)
            else:
                _publish(selector, selection, owned)
        except BaseException:
            for path in reversed(owned):
                _remove_owned(path, configs)
            raise
        finally:
            _remove_owned(temporary, configs)
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    return [destination.relative_to(project).as_posix() for _, destination in staged_entries] + ["configs/version_run.json"]
