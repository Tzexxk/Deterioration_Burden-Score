"""Locate a frozen dataset without using the active data-processing selection."""

import re
from dataclasses import dataclass

from src.common.config import DataContext
from src.common.manifest import ManifestStore
from src.common.paths import read_json, require


@dataclass(frozen=True)
class ReleasedData:
    store: ManifestStore
    acceptance: dict
    export: dict

    @classmethod
    def load(cls, layout, release_id):
        match = re.fullmatch(r"v([1-9][0-9]*)_r([1-9][0-9]*)", release_id) if isinstance(release_id, str) else None
        require(match is not None, "INVALID_DATA_RELEASE", "Use the form v<version>_r<run>")
        context = DataContext(layout, *map(int, match.groups()))
        store = ManifestStore(layout, data=context)
        ref = store.manifest_ref("step13", "validate")
        marker_path = store.control / "release.json"
        require(marker_path.is_file(), "SOURCE_NOT_RELEASED", "The selected dataset must pass and freeze at Step13")
        marker = read_json(marker_path)
        require(marker.get("scope") == store.scope and marker.get("status") == "released"
                and marker.get("acceptance") == ref, "SOURCE_NOT_RELEASED",
                "Release marker does not bind the selected dataset acceptance")
        acceptance = store.validate(ref)
        require(marker.get("generation") == acceptance["generation"]
                and acceptance["details"].get("checks_passed") is True
                and acceptance["details"].get("training_package_accepted") is True,
                "SOURCE_NOT_RELEASED", "Dataset acceptance is stale or incomplete")
        export_ref = store.manifest_ref("step12", "export")
        export = read_json(layout.resolve(export_ref))
        require(acceptance["inputs"] == [{"manifest": export_ref, "generation": export["generation"]}],
                "INVALID_RELEASE", "Dataset acceptance must bind the training export")
        return cls(store, acceptance, export)

    def package_file(self, name):
        ref = self.store.output_ref(self.export, f"package/{name}")
        require(ref["root"] == "processed", "INVALID_RELEASE", "Training package files must be in processed storage")
        return self.store.layout.resolve(ref)
