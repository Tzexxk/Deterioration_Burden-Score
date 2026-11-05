"""EHR feature and patient-fold checks for a frozen training package."""

from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import create_patient_split as split
from src.cohort.burden_io import COMPONENTS, score
from src.common.paths import read_json, require
from src.data.acceptance_checks import validate_ehr_contract
from src.data.package_contract import CV_SAMPLE_SCHEMA, FOLD_SPLIT_COLUMNS, SPLITS
from src.models.released_data import ReleasedData


@dataclass
class EHRPackage:
    release: ReleasedData
    samples: pd.DataFrame
    features: pd.DataFrame
    schema: dict
    metadata: dict
    cv: dict


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


def load_package(release):
    package_file = release.package_file
    root = package_file("samples.parquet").parent
    metadata = read_json(package_file("dataset_metadata.json"))
    schema = read_json(package_file("feature_schema.json"))
    key = release.store.data.key
    inherited = metadata.get("parent_release_id", key)
    require(metadata.get("release_id") == metadata.get("data_run") == metadata.get("cohort_version") == key
            and all(metadata.get(field) == inherited for field in ("dictionary_version", "feature_set_version", "split_version"))
            and metadata.get("feature_set_version") == schema.get("feature_set_version"),
            "INVALID_EHR_PACKAGE", "Package and inherited EHR/split identities disagree")
    require(metadata.get("interface_version") == "1.0" and metadata.get("target_column") == "burden_score"
            and metadata.get("horizon_hours") == 72 and metadata.get("ehr_lookback_hours") == 24
            and metadata.get("split_unit") == "subject_id" and metadata.get("split_values") == list(SPLITS)
            and metadata.get("ehr_preprocessing_applied") is False,
            "INVALID_EHR_PACKAGE", "Unexpected label, time window, split or preprocessing policy")
    registered = {item["name"].removeprefix("package/") for item in release.export["outputs"]
                  if item["name"].startswith("package/")}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(metadata.get("files") == sorted(registered) and actual == registered,
            "INVALID_EHR_PACKAGE", "Package inventory differs from the frozen export")

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

    ehr = pq.read_table(package_file("ehr_features.parquet"))
    feature_names = schema["numeric_features"] + schema["categorical_features"]
    rules = [{"feature_name": n.removesuffix("_last_24h")} for n in schema["numeric_features"] if n.endswith("_last_24h")]
    counts = validate_ehr_contract(ehr, sample_table, schema, rules)
    require(metadata.get("feature_counts") == {"numeric": counts["numeric_features"], "categorical": counts["categorical_features"]}
            and not set(feature_names) & set(samples.columns),
            "INVALID_EHR_FEATURES", "Feature counts or feature whitelist disagree with samples")
    features = ehr.to_pandas().set_index("sample_id").loc[samples.sample_id, feature_names].reset_index(drop=True)

    fold_files = [f"splits/fold_{i}/{split.TABLE_NAME}" for i in range(1, 6)]
    require(metadata.get("split_columns") == list(FOLD_SPLIT_COLUMNS)
            and metadata.get("cross_validation") == {"indices_file": split.CV_INDICES, "n_folds": 5,
                "split_column_semantics": "per_fold", "patient_split_files": fold_files},
            "INVALID_EHR_FOLDS", "Expected the frozen five-fold package interface")
    cv = read_json(package_file(split.CV_INDICES))
    validate_folds(sample_table, cv, [pq.read_table(package_file(name)) for name in fold_files], inherited)
    return EHRPackage(release, samples, features, schema, metadata, cv)


def _input_summary(package):
    samples, cv, schema = package.samples, package.cv, package.schema
    folds = []
    for fold in cv["folds"]:
        train, validation = fold["train_indices"], fold["validation_indices"]
        folds.append({"fold": fold["fold"], "train_studies": len(train), "validation_studies": len(validation),
                      "train_patients": int(samples.iloc[train].subject_id.nunique()),
                      "validation_patients": int(samples.iloc[validation].subject_id.nunique())})
    return {"data_release": package.release.store.data.key, "feature_set_version": schema["feature_set_version"],
            "split_version": cv["split_version"], "studies": len(samples),
            "admissions": int(samples.hadm_id.nunique()), "patients": int(samples.subject_id.nunique()),
            "development_studies": len(cv["development_indices"]), "fixed_test_studies": len(cv["test_indices"]),
            "numeric_features": len(schema["numeric_features"]), "categorical_features": len(schema["categorical_features"]),
            "folds": folds, "accepted_files_verified": len(package.metadata["files"])}


def checked_input_summary(package):
    release = package.release
    summary = _input_summary(package)
    store = release.store
    ref = store.output_ref(release.acceptance, "training_package_acceptance.json")
    evidence = read_json(store.layout.resolve(ref))
    sample_fields = ("studies", "admissions", "patients", "fixed_test_studies")
    require(evidence.get("training_package_accepted") is True
            and evidence.get("source_export") == release.acceptance["inputs"][0]
            and all(summary[k] == evidence["checks"]["samples"]["evidence"][k] for k in sample_fields)
            and len(summary["folds"]) == evidence["checks"]["samples"]["evidence"]["folds_checked"]
            and all(summary[k] == evidence["checks"]["ehr_contract"]["evidence"][k]
                    for k in ("numeric_features", "categorical_features")),
            "INVALID_EHR_PACKAGE", "EHR population, folds or features differ from Step13 evidence")
    return summary
