"""All machine-specific roots enter through paths.local.json."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def require(condition, code, message):
    if not condition:
        raise PipelineError(code, message)


def positive(value):
    require(type(value) is int and value > 0, "INVALID_VERSION", "Expected a positive integer")
    return value


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value),
            "INVALID_IDENTIFIER", "Expected letters, digits, underscores or hyphens")
    return value


def relative(value):
    require(isinstance(value, str) and bool(value) and "\\" not in value,
            "INVALID_PATH", "Use a nonempty relative POSIX path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and not PureWindowsPath(value).drive
            and ".." not in path.parts and ":" not in value,
            "INVALID_PATH", "Absolute paths and parent traversal are forbidden")
    return path.as_posix()


def portable(value):
    """Reject serialized absolute paths, including paths embedded in prose."""
    if isinstance(value, dict):
        for key, item in value.items():
            portable(key)
            portable(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            portable(item)
    elif isinstance(value, str):
        # Percent/slash is also used inside compound medication names.
        absolute = re.search(
            r"(?<![A-Za-z0-9/:])(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+\\)"
            r"|(?<![A-Za-z0-9/:%])/(?!/)[A-Za-z0-9_.-]+", value)
        require(not absolute and not value.startswith(("/", "\\")),
                "ABSOLUTE_PATH_FORBIDDEN", "Serialized values must use portable paths")
    return value


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise PipelineError("MISSING_FILE", "Required relative file does not exist") from None
    except (ValueError, UnicodeError):
        raise PipelineError("INVALID_JSON", "Required file is not valid UTF-8 JSON") from None


class Layout:
    def __init__(self, project=None):
        self.project = Path(project or Path(__file__).resolve().parents[2]).resolve()
        require((self.project / "pyproject.toml").is_file(), "PROJECT_NOT_FOUND", "Project marker missing")
        settings = read_json(self.project / "configs/environment/paths.local.json")
        require(isinstance(settings, dict) and isinstance(settings.get("roots"), dict),
                "INVALID_PATH_CONFIG", "paths.local.json requires a roots object")
        require("project" not in settings["roots"], "INVALID_PATH_CONFIG", "Project root cannot be overridden")
        self.roots = {"project": self.project}
        for name, value in {"processed": "data/processed", "outputs": "outputs", **settings["roots"]}.items():
            identifier(name)
            require(isinstance(value, str) and bool(value), "INVALID_PATH_CONFIG", "Root must be a path string")
            self.roots[name] = (self.project / value).resolve()
        require(not self.roots["processed"].is_relative_to(self.roots["outputs"])
                and not self.roots["outputs"].is_relative_to(self.roots["processed"]),
                "INVALID_PATH_CONFIG", "Writable roots must not overlap")
        for name in ("processed", "outputs"):
            writable = self.roots[name]
            require(writable != self.project and not self.project.is_relative_to(writable),
                    "INVALID_PATH_CONFIG", "Writable root must not contain the project")
            for other, root in self.roots.items():
                if other not in ("project", "processed", "outputs"):
                    require(not writable.is_relative_to(root) and not root.is_relative_to(writable),
                            "INVALID_PATH_CONFIG", "Writable roots must not overlap external input roots")

    def path(self, root, path):
        require(root in self.roots, "UNKNOWN_ROOT", "Unknown configured root")
        value = (self.roots[root] / relative(path)).resolve()
        require(value.is_relative_to(self.roots[root]), "PATH_ESCAPE", "Path escaped its declared root")
        return value

    def resolve(self, ref):
        require(isinstance(ref, dict) and set(ref) == {"root", "path"}, "INVALID_REFERENCE", "Expected root and path")
        return self.path(ref["root"], ref["path"])

    def ref(self, root, path):
        value = self.path(root, path)
        return {"root": root, "path": value.relative_to(self.roots[root]).as_posix()}
