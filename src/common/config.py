"""Data selection and model protocol selection deliberately remain separate."""

from dataclasses import dataclass

from .paths import Layout, portable, positive, read_json, relative, require


def outcome_window(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("horizon_hours", "outcome_window_hours"):
                require(type(item) is int and item == 72, "UNSUPPORTED_OUTCOME_WINDOW", "Only the 72h outcome is supported")
            if key == "audit_horizons_hours":
                require(item == [72], "UNSUPPORTED_OUTCOME_WINDOW", "Only one 72h audit window is supported")
            outcome_window(item)
    elif isinstance(value, list):
        for item in value:
            outcome_window(item)


@dataclass(frozen=True)
class DataContext:
    layout: Layout
    version: int
    run: int

    @classmethod
    def load(cls, layout, version, run):
        positive(version)
        positive(run)
        active = portable(read_json(layout.path("project", "configs/version_run.json")))
        require(isinstance(active, dict) and set(active) == {"version", "run"}, "INVALID_VERSION_CONFIG", "Expected version and run only")
        positive(active["version"])
        positive(active["run"])
        require(active == {"version": version, "run": run}, "VERSION_MISMATCH", "Command version/run differs from version_run.json")
        ctx = cls(layout, version, run)
        require(layout.path("project", ctx.directory).is_dir(), "MISSING_VERSION_CONFIG", "Selected data configuration directory is missing")
        return ctx

    @property
    def key(self):
        return f"v{self.version}_r{self.run}"

    @property
    def directory(self):
        return f"configs/data/{self.key}"

    def config_ref(self, path):
        return self.layout.ref("project", f"{self.directory}/{relative(path)}")

    def check_config_ref(self, ref):
        path = self.layout.resolve(ref)
        base = self.layout.path("project", self.directory)
        require(ref["root"] == "project" and path.is_relative_to(base),
                "CROSS_VERSION_CONFIG", "Configuration and review must belong to the selected data version")
        require(path.is_file(), "MISSING_FILE", "Selected configuration or review is missing")
        try:
            portable(path.read_text(encoding="utf-8-sig"))
        except UnicodeError:
            require(False, "INVALID_CONFIG", "Configuration and review must be UTF-8 text")
        if path.suffix == ".json":
            outcome_window(read_json(path))
        return path

    def read(self, path, required=()):
        ref = self.config_ref(path)
        value = portable(read_json(self.check_config_ref(ref)))
        require(isinstance(value, dict), "INVALID_CONFIG", "Configuration must be an object")
        require(all(key in value for key in required), "MISSING_CONFIG_FIELD", "Required configuration field missing")
        outcome_window(value)
        return value


def load_model_config(layout, path, required=()):
    target = layout.path("project", relative(path))
    require(target.is_relative_to(layout.path("project", "configs/models")),
            "INVALID_MODEL_CONFIG", "Model protocol must be in configs/models")
    value = portable(read_json(target))
    require(isinstance(value, dict) and all(key in value for key in required),
            "INVALID_MODEL_CONFIG", "Model configuration fields missing")
    outcome_window(value)
    return value


def add_data_arguments(parser):
    parser.add_argument("--version", type=int, required=True, help="Raw-data version v")
    parser.add_argument("--run", type=int, required=True, help="Data-processing run r")
