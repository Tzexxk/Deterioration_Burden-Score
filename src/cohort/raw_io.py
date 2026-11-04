"""Raw CXR path resolution and acquisition-time parsing."""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

FULL_DT = re.compile(r"^\d{14}(?:\.\d{1,6})?(?:[+-]\d{4})?$")
FULL_TM = re.compile(r"^\d{6}(?:\.\d{1,6})?$")

def _under_root(root: Path, relative: str) -> Path:
    """Reject absolute/traversal paths from untrusted index or config CSV text."""
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("unsafe_relative_path")
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("unsafe_relative_path")
    return candidate


def _indexed_dicom_path(
    cxr_root: Path,
    dicom_dir: Path,
    index_row: dict[str, str],
    layout: str,
    local_files: set[Path],
    *,
    probe_root: bool = True,
) -> tuple[Path, str, bool]:
    """Resolve a validated index path; skip root probes after a full root scan."""
    value = index_row.get("path", "")
    parts = value.replace("\\", "/").split("/")
    if (
        not value
        or value.startswith(("/", "\\"))
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("unsafe_index_path")

    standard = dicom_dir.joinpath(*parts)
    root_relative = cxr_root.joinpath(*parts)
    candidates = [(standard, "standard_dataset"), (root_relative, "standard_root")]
    if (
        len(parts) == 5
        and parts[0] == "files"
        and re.fullmatch(r"\d+", index_row.get("subject_id", ""))
        and re.fullmatch(r"\d+", index_row.get("study_id", ""))
        and re.fullmatch(r"[0-9a-fA-F-]+", index_row.get("dicom_id", ""))
        and parts[1] == "p" + index_row["subject_id"][:2]
        and parts[2] == "p" + index_row["subject_id"]
        and parts[3] == "s" + index_row["study_id"]
        and parts[4] == index_row["dicom_id"] + ".dcm"
    ):
        flattened = dicom_dir.joinpath(*parts[2:])
    else:
        raise ValueError("index_path_key_mismatch")

    if layout == "flattened":
        candidates = [(flattened, "flattened")]
    elif layout == "standard":
        candidates = [(standard, "standard_dataset")]
    elif layout == "auto":
        candidates.append((flattened, "flattened"))
    else:
        raise ValueError("Unsupported dicom_path_layout")

    for candidate, strategy in candidates:
        if candidate in local_files or (
            probe_root and strategy == "standard_root" and candidate.is_file()
        ):
            if not candidate.resolve().is_relative_to(cxr_root):
                raise ValueError("unsafe_index_path")
            return candidate, strategy, True
    candidate, strategy = candidates[0]
    return candidate, strategy, False


def _timestamp(value: object, *, dicom_date: bool = False) -> datetime | None:
    if isinstance(value, datetime) and not dicom_date:
        return value
    text = str(getattr(value, "original_string", value) or "").strip()
    if dicom_date:
        if not re.fullmatch(r"\d{8}", text):
            return None
        try:
            return datetime.strptime(text, "%Y%m%d")
        except ValueError:
            return None
    if not FULL_DT.fullmatch(text):
        return None
    suffix = re.search(r"[+-]\d{4}$", text)
    if suffix and _dicom_timezone(suffix.group()) is None:
        return None
    zone = "%z" if suffix else ""
    fraction = ".%f" if "." in text else ""
    try:
        return datetime.strptime(text, "%Y%m%d%H%M%S" + fraction + zone)
    except ValueError:
        return None


def _dicom_timezone(value: str) -> timezone | None:
    if not re.fullmatch(r"[+-]\d{4}", value) or value == "-0000":
        return None
    hours, minutes = int(value[1:3]), int(value[3:5])
    offset = (hours * 60 + minutes) * (1 if value[0] == "+" else -1)
    return (
        timezone(timedelta(minutes=offset))
        if minutes < 60 and -720 <= offset <= 840
        else None
    )


def _acquisition_time(dataset: Any) -> tuple[str, str, str]:
    first = dataset.get("AcquisitionDateTime")
    parsed, source = _timestamp(first), "AcquisitionDateTime"
    if parsed is None:
        date_value, time_value = (
            dataset.get("StudyDate", ""),
            dataset.get("StudyTime", ""),
        )
        date = str(getattr(date_value, "original_string", date_value) or "").strip()
        clock = str(getattr(time_value, "original_string", time_value) or "").strip()
        parsed = (
            _timestamp(date + clock)
            if _timestamp(date, dicom_date=True) and FULL_TM.fullmatch(clock)
            else None
        )
        source = "StudyDate+StudyTime"
    if parsed is None:
        failure = (
            "invalid_acquisition_datetime_and_no_valid_study_time"
            if first
            else "missing_or_invalid_study_date_time"
        )
        return "", "missing", failure
    offset = str(dataset.get("TimezoneOffsetFromUTC", "") or "").strip()
    if parsed.tzinfo is None and offset:
        zone = _dicom_timezone(offset)
        if zone is None:
            return "", "missing", "invalid_timezone_offset"
        parsed = parsed.replace(tzinfo=zone)
    return parsed.isoformat(), source, ""


def _file_version(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _scan_physical_dicom(
    directory: Path, suffixes: set[str]
) -> tuple[set[Path], list[str]]:
    if not directory.is_dir():
        return set(), ["missing_dicom_directory"]
    files: set[Path] = set()
    errors: list[str] = []

    def on_error(error: OSError) -> None:
        errors.append(type(error).__name__)

    with _progress("Scan local DICOM", unit="file") as progress:
        for current_dir, _, names in os.walk(
            directory, onerror=on_error, followlinks=False
        ):
            found = 0
            for name in names:
                if Path(name).suffix.lower() in suffixes:
                    files.add(Path(current_dir) / name)
                    found += 1
            progress.update(found)
    return files, errors


def _imaging_settings(imaging: dict[str, Any]) -> tuple[int, int, str]:
    workers, batch_size, layout = (
        imaging["dicom_workers"],
        imaging["dicom_batch_size"],
        imaging["dicom_path_layout"],
    )
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("dicom_workers must be an integer from 1 to 32")
    if type(batch_size) is not int or not workers <= batch_size <= 4096:
        raise ValueError("dicom_batch_size must be between workers and 4096")
    if layout not in {"auto", "standard", "flattened"}:
        raise ValueError("Unsupported dicom_path_layout")
    if imaging["time_priority"] != ["AcquisitionDateTime", "StudyDate+StudyTime"]:
        raise ValueError("Unsupported time_priority")
    return workers, batch_size, layout


def _progress(description: str, *, unit: str = "row", total: int | None = None) -> Any:
    return tqdm(
        desc=description,
        total=total,
        unit=unit,
        unit_scale=True,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    )
