"""Workspace allocation for clinical tables and report queries."""

import csv
from pathlib import Path

from src.cohort.audit_support import CsvChecks
from src.common.paths import relative, require


def write_csv(path, columns, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


class AuditWorkspace:
    def __init__(self, directory, layout):
        self.directory = directory
        self.checks = CsvChecks(layout)
        self.files = {}

    def path(self, logical):
        logical = Path(logical)
        require(logical.parts[0] in {"data", "reports"}, "INVALID_OUTPUT", "Unknown clinical output group")
        name = relative(Path(*logical.parts[1:]).as_posix())
        target = self.directory / name
        require(all(key == logical or path != target for key, path in self.files.items()),
                "DUPLICATE_OUTPUT", "Clinical output names must be unique")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.files[logical] = target
        return target

    def check_csv(self, path):
        return self.checks.check_csv(path)

    def csv_tail_summary(self):
        return self.checks.summary()
