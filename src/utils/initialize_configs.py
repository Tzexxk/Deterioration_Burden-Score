"""Create local configuration from templates without consuming the templates."""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path
from tempfile import mkdtemp

from src.common.manifest import file_lock
from src.common.paths import portable, require


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


def initialize_configs(project=None):
    project = Path(project or Path(__file__).resolve().parents[2]).resolve()
    require((project / "pyproject.toml").is_file(), "PROJECT_NOT_FOUND", "Project marker missing")
    entries = _discover(project)
    _preflight(project, entries)
    configs = (project / "configs").resolve()
    with file_lock(configs / ".initialize.lock"):
        entries = _discover(project)
        _preflight(project, entries)
        needed = sum(path.stat().st_size for source, _ in entries
                     for path in ([source] if source.is_file() else source.rglob("*")) if path.is_file())
        require(shutil.disk_usage(configs).free >= needed * 2 + 65536,
                "INSUFFICIENT_DISK_SPACE", "Insufficient space to stage and initialize configurations")
        parent = configs / "temp"
        require(not parent.is_symlink() and (not parent.exists() or not _linked(parent)),
                "INVALID_TEMPLATE_ROOT", "Configuration temp directory must not be linked")
        parent.mkdir(exist_ok=True)
        temporary = Path(mkdtemp(prefix="initialize_", dir=parent)).resolve()
        require(temporary.is_relative_to(configs), "PATH_ESCAPE", "Initialization temp escaped configs")
        owned = []
        try:
            for source, target in entries:
                staged = temporary / target.name
                if source.is_dir():
                    shutil.copytree(source, staged)
                else:
                    shutil.copyfile(source, staged)
            _preflight(project, entries)
            for _, target in entries:
                _publish(temporary / target.name, target, owned)
        except BaseException:
            for path in reversed(owned):
                _remove_owned(path, configs)
            raise
        finally:
            _remove_owned(temporary, configs)
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    return [target.relative_to(project).as_posix() for _, target in entries]
