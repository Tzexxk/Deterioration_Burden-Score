"""Read the sample and patient-fold contract shared by model modalities."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import create_patient_split as split
from src.cohort.burden_io import COMPONENTS, score
from src.common.paths import read_json, require
from src.data.package_contract import CV_SAMPLE_SCHEMA, FOLD_SPLIT_COLUMNS


def validate_folds(sample_table, cv, tables, version):
    ids = sample_table["sample_id"].to_pylist()
    require(ids == sorted(set(ids)) and cv.get("cohort_version") == cv.get("split_version") == version,
            "INVALID_EHR_FOLDS", "Sample order or inherited split identity changed")
    cohort = sample_table.append_column("cohort_version", pa.array([version] * len(ids)))
    split.validate_cv_indices(cohort, cv, tables[0])
    for fold, column, table in zip(cv["folds"], FOLD_SPLIT_COLUMNS, tables, strict=True):
        require(split.select_fold(tables[0], cv, fold["fold"]).equals(table),
                "INVALID_EHR_FOLDS", f"Patient assignments differ from fold {fold['fold']}")
        assignments = {
            "train": fold["train_indices"], "validation": fold["validation_indices"], "test": cv["test_indices"],
        }
        roles = sample_table[column].to_pylist()
        for role, indices in assignments.items():
            require([i for i, value in enumerate(roles) if value == role] == indices,
                    "INVALID_EHR_FOLDS", f"Study assignments differ from fold {fold['fold']}")


def load_samples(release):
    package_file = release.package_file
    sample_table = pq.read_table(package_file("samples.parquet"))
    require(sample_table.schema == CV_SAMPLE_SCHEMA and sample_table.num_rows > 0
            and all(not column.null_count for column in sample_table.columns),
            "INVALID_EHR_SAMPLES", "Expected complete five-fold sample metadata")
    samples = sample_table.to_pandas()
    require(samples.sample_id.is_unique and not samples.duplicated(["subject_id", "hadm_id", "study_id"]).any(),
            "INVALID_EHR_SAMPLES", "Duplicate sample or study identity")
    require(samples[list(COMPONENTS)].isin([0, 1]).all().all(), "INVALID_EHR_SAMPLES", "Invalid event flags")
    labels = pd.DataFrame([score(list(flags)) for flags in samples[list(COMPONENTS)].itertuples(index=False, name=None)])
    for label in ("burden_score", "phys_death", "high_burden"):
        require((samples[label] == labels[label]).all(), "INVALID_EHR_SAMPLES", f"Incorrect label: {label}")
    return sample_table, samples


def load_cv(release, metadata, sample_table, inherited):
    package_file = release.package_file
    fold_files = [f"splits/fold_{i}/{split.TABLE_NAME}" for i in range(1, 6)]
    require(metadata.get("split_columns") == list(FOLD_SPLIT_COLUMNS)
            and metadata.get("cross_validation") == {"indices_file": split.CV_INDICES, "n_folds": 5,
                "split_column_semantics": "per_fold", "patient_split_files": fold_files},
            "INVALID_EHR_FOLDS", "Expected the frozen five-fold package interface")
    cv = read_json(package_file(split.CV_INDICES))
    validate_folds(sample_table, cv, [pq.read_table(package_file(name)) for name in fold_files], inherited)
    return cv
