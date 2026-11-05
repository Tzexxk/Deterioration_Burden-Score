"""Generation-bound training package export without file-hash bindings."""

import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort import create_patient_split as split
from src.cohort.burden_io import open_store
from src.cohort.burden_support import write_csv
from src.cohort.raw_io import _file_version
from src.common.manifest import ManifestStore, write_json
from src.common.paths import portable, read_json, relative, require
from src.data import package_contract as contract
from src.ehr import build_ehr_features as upstream
from src.ehr import build_features as features
from src.ehr import feature_engine as engine
from src.imaging.preprocessing import validate_pixels

IMAGE_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("sample_id", "dicom_id", "dicom_root", "dicom_path", "npy_root", "npy_path",
                                     "npy_relpath", "view_position", "qc_status", "processing_version", "split")]
    + [(name, pa.int64()) for name in ("subject_id", "hadm_id", "study_id")]
    + [("t0", pa.timestamp("us"))]
)


def load_settings(context, path):
    settings = context.read(path, required=("interface_version", "workspace_estimate_bytes"))
    require(set(settings) == {"interface_version", "workspace_estimate_bytes"} and settings["interface_version"] == "1.0",
            "INVALID_EXPORT_CONFIG", "Use interface 1.0; dataset identities come from version/run")
    require(type(settings["workspace_estimate_bytes"]) is int and settings["workspace_estimate_bytes"] > 0,
            "INVALID_EXPORT_CONFIG", "Set a positive disk-space estimate")
    return settings


def load_inputs(context, store, workspace):
    # Earlier stages own clinical replay and patient allocation; verify them before copying.
    prior = store.validate(store.manifest_ref("step11", "audit"))
    audit_settings = read_json(context.layout.resolve(store.output_ref(prior, "ehr_audit.json")))["settings"]
    with tempfile.TemporaryDirectory(dir=workspace, prefix=".export_") as scratch:
        with open_store(Path(scratch), audit_settings["memory_limit"], max_temp_directory_size=audit_settings["max_temp_directory_size"]) as connection:
            cohort = upstream.verify_upstream(context, store, connection)
    manifests = {name: store.validate(store.manifest_ref(step, phase)) for name, step, phase in (
        ("features", "step11", "build"), ("split", "step09", "split"),
        ("images", "step07", "prepare"), ("endpoint", "step05", "freeze"))}
    def output(name, filename):
        return context.layout.resolve(store.output_ref(manifests[name], filename))
    record = read_json(output("features", "ehr_features_freeze.json"))
    require(record.get("data_run") == record.get("feature_set_version") == context.key
            and record.get("ehr_features_ready") is True and record.get("audit_checks_passed") is True
            and record.get("blocking_conflicts") == 0 and record.get("preprocessing_applied") is False,
            "FEATURES_NOT_READY", "Checked, unprocessed current-run features are required")
    schema = read_json(output("features", "feature_schema.json"))
    feature_inputs = features.load_inputs(context, store)
    require(schema == engine.schema_for(feature_inputs["rules"], context.key), "FEATURE_SCHEMA_MISMATCH", "Feature schema differs from reviewed definitions")
    ehr = pq.read_table(output("features", "ehr_features_raw.parquet"))
    engine.validate_wide(ehr, cohort, schema)
    image_settings = read_json(output("images", "effective_configuration.json"))["processing"]
    cv_files = {}
    if feature_inputs["cv_indices"] is not None:
        cv_files[split.CV_INDICES] = output("split", split.CV_INDICES)
        for fold in feature_inputs["cv_indices"]["folds"]:
            name = f"fold_{fold['fold']}/{split.TABLE_NAME}"
            cv_files[f"splits/{name}"] = output("split", name)
    return {"cohort": cohort, "patients": feature_inputs["patients"], "cv_files": cv_files,
            "qc_records": pq.read_table(output("images", "image_qc_status.parquet")).to_pylist(),
            "schema": schema, "ehr": ehr, "ehr_path": output("features", "ehr_features_raw.parquet"),
            "schema_path": output("features", "feature_schema.json"), "image_settings": image_settings,
            "split_settings": read_json(output("split", "split_freeze.json"))["settings"],
            "decision": read_json(output("endpoint", "endpoint_decision.json")), "image_manifest": manifests["images"]}


def image_manifest(context, store, inputs, samples):
    cohort = {row["sample_id"]: row for row in inputs["cohort"].to_pylist()}
    qc = {row["sample_id"]: row for row in inputs["qc_records"]}
    require(len(qc) == len(inputs["qc_records"]), "INVALID_IMAGE_QC", "Duplicate QC keys")
    rows = []
    columns = contract.split_columns(samples)
    schema = IMAGE_SCHEMA
    if columns == contract.FOLD_SPLIT_COLUMNS:
        schema = pa.schema([field for field in IMAGE_SCHEMA if field.name != "split"]
                           + [(name, pa.string()) for name in columns])
    fields = "subject_id hadm_id study_id dicom_id dicom_path t0 npy_relpath processing_version qc_status view_position".split()
    for sample in samples.to_pylist():
        original = cohort[sample["sample_id"]]
        checked = qc.get(sample["sample_id"])
        require(checked is not None and checked["qc_status"] == "pass" and checked["processing_status"] == "complete"
                and all(checked[name] == original[name] for name in fields),
                "INVALID_IMAGE_QC", "Included image must agree with frozen cohort and accepted QC")
        ref = store.output_ref(inputs["image_manifest"], original["npy_relpath"])
        dicom = context.layout.ref("mimic_cxr", original["dicom_path"])
        rows.append({**{name: original[name] for name in schema.names if name not in {"dicom_root", "npy_root", "npy_path", *columns}},
                     "dicom_root": dicom["root"], "dicom_path": dicom["path"], "npy_root": ref["root"], "npy_path": ref["path"],
                     **{name: sample[name] for name in columns}})
    return pa.Table.from_pylist(rows, schema=schema)


def streams_equal(left, right):
    while block := left.read(1024 * 1024):
        if block != right.read(len(block)):
            return False
    return not right.read(1)


def copy_image(source, target, size):
    before = _file_version(source)
    validate_pixels(np.load(source, allow_pickle=False), size)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    with source.open("rb") as left, target.open("rb") as right:
        require(streams_equal(left, right), "IMAGE_COPY_CHANGED", "Copied NPY differs from its source")
    require(before == _file_version(source), "SOURCE_CHANGED", "Source NPY changed during export")
    validate_pixels(np.load(target, allow_pickle=False), size)


def verify_archive(path, files, prefix="package"):
    prefix = relative(prefix)
    expected = {f"{prefix}/{relative(name)}": source for name, source in files.items()}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(expected), "INVALID_PACKAGE_ZIP", "ZIP inventory differs from the directory")
        for name, source in expected.items():
            require(archive.getinfo(name).file_size == source.stat().st_size, "INVALID_PACKAGE_ZIP", "ZIP entry size differs")
            with archive.open(name) as left, source.open("rb") as right:
                require(streams_equal(left, right), "INVALID_PACKAGE_ZIP", "ZIP entry bytes differ from the directory")


def git_state(project):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=project, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_dirty": None}
    return {"git_commit": commit, "git_worktree_dirty": bool(status)}


def package_readme(key, generated, *, five_fold=False):
    text = f'''# Training package {key}

Interface 1.0. Outcome window: 72 hours. EHR lookback: 24 hours.
Generated UTC: {generated}.
EHR, image and fusion models use identical samples, labels and patient assignments.
Read actual population counts from summary.csv and dataset_metadata.json.

```python
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = Path("path/to/package")
samples = pq.read_table(DATA_ROOT / "samples.parquet").to_pandas()
ehr = pq.read_table(DATA_ROOT / "ehr_features.parquet").to_pandas()
schema = json.loads((DATA_ROOT / "feature_schema.json").read_text(encoding="utf-8"))
aligned = samples[["sample_id"]].merge(ehr, on="sample_id", validate="one_to_one")
x = aligned[schema["numeric_features"] + schema["categorical_features"]]
gray = np.load(DATA_ROOT / samples.iloc[0]["image_relpath"], allow_pickle=False)
image = np.repeat(gray[None, :, :], 3, axis=0)
```

NPY files are float32, 224x224, single-channel, finite and in [0,1]. Do not
divide by 255 again. Model normalization and augmentation belong to the loader.
Numeric EHR missing values remain NaN; categorical missing values are __MISSING__.
Fit imputation, scaling and category vocabularies on training patients only.
Never use IDs, labels, split, observation duration or post-imaging fields as inputs.

The target is operational deterioration burden, not a validated severity scale
or an event probability. Missing records do not establish clinical absence.
Hypoxemia uses supplied ICU chartevents only. Live early discharge truncates
follow-up; absent recorded events before the cutoff are operational zeros.
Deidentified anchor ages cannot recover true ages above 89. Service availability
uses transfertime because no independent service storetime is supplied.

ZIP contains this entire package/ directory. After relocation, change only DATA_ROOT.
Step12 export is complete; independent Step13 acceptance remains pending.
Version/run and upstream manifest generations record provenance; no file-hash
verification or file-hash review binding is used.
'''

    if five_fold:
        text += """
## Patient-grouped five-fold cross-validation

For fold n, read split_fold_n in samples.parquet. summary.csv reports each fold's
training and validation counts and the fixed test set once. Patient tables under
splits/ and the zero-based study indices
in cross_validation_indices.json encode the same assignments. Indices refer to samples.parquet in sample_id
order. All five folds use the same fixed test set. Fit learned preprocessing on
each fold's training partition and keep the fixed test set out of model selection.
"""
    return text


def metadata(context, settings, inputs, summaries, files, generated, dependencies):
    split = inputs["split_settings"]
    decision = inputs["decision"]
    cv_files = inputs["cv_files"]
    cross_validation = {
        "cross_validation": {
            "indices_file": "cross_validation_indices.json",
            "n_folds": 5,
            "split_column_semantics": "per_fold",
            "patient_split_files": sorted(name for name in cv_files if name.startswith("splits/")),
        }
    } if cv_files else {}
    return portable({**cross_validation, "interface_version": settings["interface_version"], "release_id": context.key, "data_run": context.key,
        "outcome_version": decision["outcome_version"], "cohort_version": context.key, "feature_set_version": context.key,
        "dictionary_version": context.key, "split_version": context.key, "horizon_hours": 72, "ehr_lookback_hours": 24,
        "target_column": "burden_score", "target_range": [0, 8], "target_formula": decision["target_formula"],
        **({"split_columns": list(contract.FOLD_SPLIT_COLUMNS)} if cv_files else {"split_column": "split"}),
        "split_values": list(contract.SPLITS), "split_unit": "subject_id",
        "split_rules": {name: split[name] for name in ("algorithm_version", "seed", "weights", "primary_stratum", "require_high_burden_in_each_split")},
        "image": {"format": "npy", "dtype": "float32", "shape": [inputs["image_settings"]["size"]]*2,
                  "value_range": [0.0, 1.0], "model_normalization_applied": False},
        "image_config": {**inputs["image_settings"],
                         "intensity_normalization": "After frozen DICOM display transforms, valid pixels are mapped by clip((value-p1)/(p99-p1),0,1); excluded pixels are zero.",
                         "geometry": "Aspect-preserving bilinear resize, nearest-neighbor valid-pixel mask and centered zero padding."},
        "hypoxemia_source": "MIMIC-IV ICU chartevents only",
        "live_discharge_rule": "Retain events in (t0, dischtime] when alive discharge precedes t0+H; absent recorded events are operational zeros.",
        "feature_counts": {"numeric": len(inputs["schema"]["numeric_features"]), "categorical": len(inputs["schema"]["categorical_features"])},
        "split_counts": {group: {row["metric"]: row["value"] for row in summaries if row["split"] == group
                                and row["metric"] in {"studies", "admissions", "patients"}}
                         for group in dict.fromkeys(row["split"] for row in summaries)},
        "generated_utc": generated, **git_state(context.layout.project), "upstream": dependencies,
        "files": sorted(files), "ehr_preprocessing_applied": False, "independent_validation_completed": False,
        "limitations": inputs["schema"].get("limitations", [])})


def run_export(context, config_path="export/training_package.json"):
    settings = load_settings(context, config_path)
    store = ManifestStore(context.layout, data=context)
    refs = [store.manifest_ref(step, phase) for step, phase in (("step11", "build"), ("step09", "split"), ("step08", "freeze"), ("step07", "prepare"))]
    details = {"training_package_exported": True, "independent_validation_completed": False}
    with store.publish("step12", "export", inputs=refs, configs=[context.config_ref(config_path)],
                       data_directories=("package",), data_files=("training_package.zip",), details=details,
                       required_bytes=settings["workspace_estimate_bytes"]) as work:
        inputs = load_inputs(context, store, work)
        cv_files = inputs["cv_files"]
        folds = [pq.read_table(cv_files[f"splits/fold_{i}/{split.TABLE_NAME}"])
                 for i in range(1, 6)] if cv_files else None
        samples = contract.build_samples(inputs["cohort"], inputs["patients"], fold_patients=folds)
        engine.validate_wide(inputs["ehr"], samples, inputs["schema"])
        images = image_manifest(context, store, inputs, samples)
        dependencies = [{"manifest": ref, "generation": store.validate(ref)["generation"]} for ref in refs]
        generated = datetime.now(timezone.utc).isoformat()
        package = work / "package"
        package.mkdir()
        files = {}
        def target(name):
            path = package / relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            files[name] = path
            return path
        pq.write_table(samples, target("samples.parquet"), compression="zstd")
        shutil.copyfile(inputs["ehr_path"], target("ehr_features.parquet"))
        shutil.copyfile(inputs["schema_path"], target("feature_schema.json"))
        summaries = contract.summary_rows(samples)
        write_csv(target("summary.csv"), contract.SUMMARY_COLUMNS, summaries)
        for name, source in inputs["cv_files"].items():
            shutil.copyfile(source, target(name))
        target("README.md").write_text(
            package_readme(context.key, generated, five_fold=bool(inputs["cv_files"])),
            encoding="utf-8", newline="\n",
        )
        for row in images.to_pylist():
            source = context.layout.resolve({"root": row["npy_root"], "path": row["npy_path"]})
            copy_image(source, target(row["npy_relpath"]), inputs["image_settings"]["size"])
        dataset_metadata = target("dataset_metadata.json")
        write_json(dataset_metadata, metadata(context, settings, inputs, summaries, files, generated, dependencies))
        contract.archive_files(work / "training_package.zip", files, "package")
        verify_archive(work / "training_package.zip", files)
        pq.write_table(images, work / "image_manifest.parquet", compression="zstd")
        result = {"data_run": context.key, "release_id": context.key, "training_package_exported": True,
                  "independent_validation_completed": False, "settings": settings, "upstream": dependencies,
                  "studies": samples.num_rows, "images": images.num_rows, "package_files": len(files),
                  "split_counts": read_json(dataset_metadata)["split_counts"], "generated_utc": generated,
                  "package_inventory": [{"name": name, "bytes": path.stat().st_size} for name, path in sorted(files.items())]}
        write_json(work / "training_package_export.json", result)
        write_json(work / "run_metadata.json", result)
        (work / "report.md").write_text("# Training package export\n\nDirectory and ZIP were checked byte-for-byte. "
                                        "Independent Step13 acceptance is pending; this data run is not yet released.\n", encoding="utf-8", newline="\n")
        details.update(studies=samples.num_rows, images=images.num_rows, package_files=len(files))
    return 0
