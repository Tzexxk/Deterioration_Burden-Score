"""Portable serialization and bounded CSV checks for raw-data auditing."""

import csv
import re
from functools import lru_cache
from pathlib import Path

from src.common.csv_io import CSV_TAIL_COLUMNS, _check_csv_tail
from src.common.paths import portable, require


def reference(path, layout):
    path = Path(path).resolve()
    for name, root in sorted(layout.roots.items(), key=lambda item: len(str(item[1])), reverse=True):
        if path.is_relative_to(root):
            return {"root": name, "path": path.relative_to(root).as_posix()}
    require(False, "UNKNOWN_INPUT_PATH", "Input path has no configured root")


def label(path, layout):
    ref = reference(path, layout)
    return ref["root"] + ":" + ref["path"]


@lru_cache(maxsize=32)
def _root_patterns(layout):
    patterns = []
    for name, root in sorted(layout.roots.items(), key=lambda item: len(str(item[1])), reverse=True):
        for prefix in {str(root), root.as_posix()}:
            pattern = re.compile(re.escape(prefix) + r"(?=$|[\\/\s:])(?P<tail>[^\s\"'<>]*)")
            patterns.append((name, prefix, pattern))
    return patterns


def portable_value(value, layout):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "index_path" and isinstance(item, str):
                # Retain the failure type, never echo an unsafe external path.
                parts = item.replace("\\", "/").split("/")
                if item.startswith(("/", "\\")) or ":" in parts[0] or ".." in parts:
                    item = "<invalid_index_path>"
            result[key] = portable_value(item, layout)
        return portable(result)
    if isinstance(value, (tuple, list)):
        return [portable_value(item, layout) for item in value]
    if not isinstance(value, str):
        return value
    if not value:
        return value
    if not value.startswith("/") and not any(marker in value for marker in (":\\", ":/", "\\\\", " /", "\n/")):
        return portable(value)
    text = value
    for name, prefix, pattern in _root_patterns(layout):
        if prefix not in text:
            continue
        def replace(match):
            tail = match.group("tail").replace("\\", "/").lstrip("/")
            return name + ":" + (tail or ".")
        text = pattern.sub(replace, text)
    return portable(text)


class PortableWriter:
    def __init__(self, stream, columns, layout):
        self.writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        self.layout = layout

    def writeheader(self):
        self.writer.writeheader()

    def writerow(self, row):
        self.writer.writerow(portable_value(row, self.layout))


def write_csv(path, columns, rows, *, layout):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = PortableWriter(stream, columns, layout)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CsvChecks:
    def __init__(self, layout):
        self.layout = layout
        self.checks = {}

    def check_csv(self, path):
        path = path.resolve()
        if path not in self.checks:
            self.checks[path] = _check_csv_tail(path, display=label(path, self.layout))
        return self.checks[path]

    def reference(self, path):
        return reference(path, self.layout)

    def summary(self):
        values = list(self.checks.values())
        return {
            "report": "csv_tail_checks.csv",
            "status": "checked" if values else "no_csv_inputs",
            "checked_file_count": len(values),
            "incomplete_file_count": sum(v["status"] == "incomplete" for v in values),
            "unverified_file_count": sum(v["status"] == "unverified" for v in values),
            "skipped_record_count": sum(v["skipped_rows"] for v in values),
        }

    def write_report(self, path):
        write_csv(path, CSV_TAIL_COLUMNS, list(self.checks.values()), layout=self.layout)
