"""Sample contract, population summaries and deterministic portable archives."""

import math
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path

import pyarrow as pa

from src.cohort.burden_io import COMPONENTS

SPLITS = ("train", "validation", "test")


SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("subject_id", pa.int64()),
        ("hadm_id", pa.int64()),
        ("study_id", pa.int64()),
        ("dicom_id", pa.string()),
        ("split", pa.string()),
        ("image_relpath", pa.string()),
        ("burden_score", pa.float32()),
        *[(k, pa.uint8()) for k in COMPONENTS],
        ("phys_death", pa.uint8()),
        ("high_burden", pa.uint8()),
        ("t0", pa.timestamp("us")),
        ("observation_end", pa.timestamp("us")),
        ("observed_hours", pa.float32()),
        ("early_discharge", pa.uint8()),
    ]
)


SUMMARY_COLUMNS = ["split", "metric", "value", "unit", "denominator"]


PACKAGE_TABLES = [
    "samples.parquet",
    "ehr_features.parquet",
    "feature_schema.json",
    "summary.csv",
    "README.md",
    "dataset_metadata.json",
]


def expected_image_path(row: dict) -> str:
    return f"images/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.npy"


def build_samples(cohort: pa.Table, patients: pa.Table) -> pa.Table:
    records = patients.to_pylist()
    assignments = {r["subject_id"]: r["split"] for r in records}
    if (
        len(assignments) != len(records)
        or any(v not in SPLITS for v in assignments.values())
        or set(assignments) != set(cohort["subject_id"].to_pylist())
    ):
        raise ValueError("Patient split keys or assignments disagree with cohort")
    rows = []
    for original in sorted(cohort.to_pylist(), key=lambda r: r["sample_id"]):
        row = {
            k: original[k]
            for k in SAMPLE_SCHEMA.names
            if k not in {"split", "image_relpath"}
        }
        row.update(
            split=assignments[row["subject_id"]], image_relpath=original["npy_relpath"]
        )
        if row["image_relpath"] != expected_image_path(row):
            raise ValueError("Frozen image path does not follow the package layout")
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA)
    validate_samples(table)
    return table


def validate_samples(samples: pa.Table) -> None:
    if (
        samples.schema != SAMPLE_SCHEMA
        or not samples.num_rows
        or any(samples[k].null_count for k in samples.column_names)
    ):
        raise ValueError("Samples schema or required fields are invalid")
    seen, triples, patients, paths = set(), set(), {}, set()
    for row in samples.to_pylist():
        key = row["sample_id"]
        triple = tuple(row[k] for k in ["subject_id", "hadm_id", "study_id"])
        expected_key = f"p{triple[0]}_h{triple[1]}_s{triple[2]}"
        if (
            key != expected_key
            or key in seen
            or triple in triples
            or any(v <= 0 for v in triple)
            or not row["dicom_id"]
        ):
            raise ValueError("Invalid or duplicated sample identity")
        seen.add(key)
        triples.add(triple)
        patient, split = row["subject_id"], row["split"]
        if split not in SPLITS or patient in patients and patients[patient] != split:
            raise ValueError("A patient crosses splits or has an invalid split")
        patients[patient] = split
        path = row["image_relpath"]
        if path != expected_image_path(row) or path in paths:
            raise ValueError("Invalid or duplicated package image path")
        paths.add(path)
        flags = [
            row[k]
            for k in [*COMPONENTS, "phys_death", "high_burden", "early_discharge"]
        ]
        if any(v not in (0, 1) for v in flags):
            raise ValueError("Invalid binary label")
        t = max(row[COMPONENTS[0]], row[COMPONENTS[1]])
        resp = max(row[COMPONENTS[2]], row[COMPONENTS[3]])
        circ, death = row[COMPONENTS[4]], row[COMPONENTS[5]]
        score = t + 2 * resp + 2 * circ + 3 * death
        if (
            row["burden_score"] != score
            or row["phys_death"] != int(bool(resp or circ or death))
            or row["high_burden"] != int(score >= 3)
        ):
            raise ValueError("Exported target formula is inconsistent")
        hours = (row["observation_end"] - row["t0"]).total_seconds() / 3600
        if (
            not 0 <= hours <= 72
            or not math.isfinite(row["observed_hours"])
            or not math.isclose(
                row["observed_hours"], hours, rel_tol=1e-6, abs_tol=1e-4
            )
            or row["observation_end"] > row["t0"] + timedelta(hours=72)
        ):
            raise ValueError("Exported observation times are inconsistent")


def summary_rows(samples: pa.Table) -> list[dict]:
    result = []
    for split in ["all", *SPLITS]:
        rows = [r for r in samples.to_pylist() if split == "all" or r["split"] == split]
        patients = {r["subject_id"] for r in rows}

        def add(metric, value, unit, denominator):
            result.append(
                dict(
                    split=split,
                    metric=metric,
                    value=value,
                    unit=unit,
                    denominator=denominator,
                )
            )

        add("studies", len(rows), "study", len(rows))
        add(
            "admissions",
            len({r["hadm_id"] for r in rows}),
            "admission",
            len({r["hadm_id"] for r in rows}),
        )
        add("patients", len(patients), "patient", len(patients))
        for score in range(9):
            add(
                f"score_{score}_studies",
                sum(r["burden_score"] == score for r in rows),
                "study",
                len(rows),
            )
            add(
                f"maximum_score_{score}_patients",
                sum(
                    max(r["burden_score"] for r in rows if r["subject_id"] == p)
                    == score
                    for p in patients
                ),
                "patient",
                len(patients),
            )
        for field in [*COMPONENTS, "phys_death", "high_burden", "early_discharge"]:
            add(field + "_studies", sum(r[field] for r in rows), "study", len(rows))
            add(
                field + "_patients",
                len({r["subject_id"] for r in rows if r[field]}),
                "patient",
                len(patients),
            )
    return result


def archive_files(target: Path, files: dict[str, Path], release: str) -> None:
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name in sorted(files):
            if (
                name.replace("\\", "/") != name
                or name.startswith("/")
                or any(p in {"", ".", ".."} for p in name.split("/"))
                or ":" in name
            ):
                raise ValueError("Unsafe ZIP entry")
            info = zipfile.ZipInfo(f"{release}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with (
                archive.open(info, "w", force_zip64=True) as destination,
                files[name].open("rb") as source,
            ):
                shutil.copyfileobj(source, destination)
