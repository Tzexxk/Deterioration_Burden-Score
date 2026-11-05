"""Independent reconstruction and acceptance of a generation-bound training package."""

import csv
import json
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pydicom

from src.cohort import create_patient_split as split
from src.cohort.burden_support import write_csv
from src.cohort.raw_io import _file_version
from src.common.manifest import ManifestStore, write_json
from src.common.paths import PipelineError, portable, read_json, relative, require
from src.data import acceptance_checks as checks
from src.imaging.preprocessing import process_dataset
from src.utils import master_reuse

PACKAGE_FILES = {"samples.parquet", "ehr_features.parquet", "feature_schema.json", "summary.csv", "README.md", "dataset_metadata.json"}
PHASES = [("step01", "audit"), ("step02", "build"), ("step03", "select"), ("step04", "match"),
          ("step05", "audit"), ("step05", "freeze"), ("step06", "build"), ("step07", "prepare"),
          ("step08", "freeze"), ("step09", "split"), ("step10", "audit"), ("step10", "review"),
          ("step10", "freeze"), ("step11", "audit"), ("step11", "build"), ("step12", "export")]


def load_settings(context, path):
    settings = context.read(path)
    require(set(settings) == {"interface_version", "expected_counts", "expected_split_studies", "expected_split_patients",
                              "relocation_policy", "failure_detail_limit", "workspace_estimate_bytes"},
            "INVALID_VALIDATION_CONFIG", "Use version/run for dataset identity and configure all acceptance bounds")
    require(settings["interface_version"] == "1.0" and settings["relocation_policy"] == "temporary_directory"
            and type(settings["failure_detail_limit"]) is int and 1 <= settings["failure_detail_limit"] <= 100,
            "INVALID_VALIDATION_CONFIG", "Unsupported interface, relocation policy or diagnostic limit")
    for name, names in [("expected_counts", {"studies", "admissions", "patients"}),
                        ("expected_split_studies", set(checks.SPLITS)), ("expected_split_patients", set(checks.SPLITS))]:
        value = settings[name]
        require(isinstance(value, dict) and set(value) == names and all(type(n) is int and n > 0 for n in value.values()),
                "INVALID_VALIDATION_CONFIG", "Fill all expected population counts with positive integers")
    require(sum(settings["expected_split_studies"].values()) == settings["expected_counts"]["studies"]
            and sum(settings["expected_split_patients"].values()) == settings["expected_counts"]["patients"]
            and settings["expected_counts"]["patients"] <= settings["expected_counts"]["admissions"] <= settings["expected_counts"]["studies"],
            "INVALID_VALIDATION_CONFIG", "Expected population counts are inconsistent")
    require(type(settings["workspace_estimate_bytes"]) is int and settings["workspace_estimate_bytes"] > 0,
            "INVALID_VALIDATION_CONFIG", "Set a positive temporary-space estimate")
    return settings


def verify_provenance(context, store):
    root = store.validate(store.manifest_ref("step12", "export"))
    manifests = {}
    def visit(manifest):
        key = (manifest["stage"], manifest["phase"])
        if key in manifests:
            return
        manifests[key] = manifest
        for dependency in manifest["inputs"]:
            visit(read_json(context.layout.resolve(dependency["manifest"])))
    visit(root)
    if ("step06", "reuse") in manifests:
        return verify_master_provenance(context, store, root, manifests)
    require(set(PHASES) <= set(manifests), "INCOMPLETE_LINEAGE", "Every required data phase must belong to the export lineage")
    def output(stage, phase, name):
        return context.layout.resolve(store.output_ref(manifests[(stage, phase)], name))
    export = read_json(output("step12", "export", "training_package_export.json"))
    require(export.get("data_run") == export.get("release_id") == context.key and export.get("training_package_exported") is True
            and export.get("independent_validation_completed") is False and export.get("upstream") == root["inputs"],
            "INVALID_EXPORT_RECORD", "A current-run export with bound upstream generations is required")
    decision = read_json(output("step05", "freeze", "endpoint_decision.json"))
    require(decision.get("version") == context.version and decision.get("run") == context.run
            and decision.get("checks_passed") is True and decision.get("horizon_hours") == 72
            and decision.get("horizon_frozen") is True and decision.get("label_rules_frozen") is True
            and decision.get("review_policy") in {"lenient", "strict"}, "INVALID_ENDPOINT", "Use a checked, frozen supported 72-hour endpoint")
    for stage, phase, filename, identity, flags in [
        ("step08", "freeze", "cohort_freeze.json", "cohort_version", ["common_cohort_frozen", "cohort_checks_passed"]),
        ("step10", "freeze", "lab_dictionary_freeze.json", "dictionary_version", ["dictionary_frozen", "review_complete"]),
        ("step11", "build", "ehr_features_freeze.json", "feature_set_version", ["ehr_features_ready", "audit_checks_passed"])]:
        record = read_json(output(stage, phase, filename))
        require(record.get("data_run") == record.get(identity) == context.key and all(record.get(flag) is True for flag in flags),
                "INVALID_FROZEN_SOURCE", "A frozen source identity or approval is invalid")
    reviewed = read_json(output("step10", "review", "review_state.json"))
    require(reviewed.get("review_complete") is True and reviewed.get("scope_acknowledgment_complete") is True,
            "INCOMPLETE_LAB_REVIEW", "Complete branch and scope review is required")
    split_record = read_json(output("step09", "split", "split_freeze.json"))
    require(split_record.get("data_run") == context.key and split_record.get("patient_split_frozen") is True
            and split_record.get("split_checks_passed") is True
            and split_record["settings"]["algorithm_version"] in split.SPLIT_METHODS.values(),
            "INVALID_FROZEN_SPLIT", "Use a supported, checked current-run patient split")
    dictionary = pq.read_table(output("step10", "freeze", "lab_dictionary.parquet"))
    require(set(dictionary["dictionary_version"].to_pylist()) == {context.key}, "DICTIONARY_VERSION_MISMATCH", "Dictionary belongs to a different data run")
    rules = checks.retained_rules(dictionary)
    package = output("step12", "export", "package/samples.parquet").parent
    five_fold = split_record["settings"]["algorithm_version"] == split.CV_ALGORITHM
    split_manifest = manifests[("step09", "split")]
    require(not five_fold or split_record.get("split_method") == split_manifest["details"].get("split_method") == "five-fold",
            "INVALID_FROZEN_SPLIT", "Published fold policy is inconsistent")
    cv_files = {}
    if five_fold:
        cv_files[split.CV_INDICES] = output("step09", "split", split.CV_INDICES)
        for fold in range(1, 6):
            name = f"fold_{fold}/{split.TABLE_NAME}"
            cv_files[f"splits/{name}"] = output("step09", "split", name)
    primary_split = cv_files["splits/fold_1/patient_split.parquet"] if five_fold else output("step09", "split", split.TABLE_NAME)
    result = {"manifests": manifests, "export_manifest": root, "export": export, "package": package,
              "archive": output("step12", "export", "training_package.zip"), "decision": decision,
              "cohort": pq.read_table(output("step08", "freeze", "cohort.parquet")),
              "evidence": pq.read_table(output("step08", "freeze", "cohort_audit.parquet")),
              "split": pq.read_table(primary_split), "split_settings": split_record["settings"], "cv_files": cv_files,
              "samples": pq.read_table(package / "samples.parquet"), "ehr": pq.read_table(package / "ehr_features.parquet"),
              "schema": read_json(package / "feature_schema.json"), "rules": rules,
              "images": pq.read_table(output("step12", "export", "image_manifest.parquet")),
              "qc": pq.read_table(output("step07", "prepare", "image_qc_status.parquet")),
              "image_settings": read_json(output("step07", "prepare", "effective_configuration.json"))["processing"],
              "labs": output("step10", "audit", "cache/labs_candidate_subset.parquet"),
              "lab_admissions": pq.read_table(output("step10", "audit", "cache/admissions.parquet")),
              "static": {name: pq.read_table(output("step11", "audit", f"cache/{name}.parquet")) for name in ("patients", "admissions", "services")}}
    require(result["schema"].get("feature_set_version") == context.key and result["schema"].get("preprocessing_applied") is False,
            "FEATURE_SCHEMA_MISMATCH", "Use the unprocessed current-run feature schema")
    for name, upstream_name in [("ehr_features.parquet", "ehr_features_raw.parquet"), ("feature_schema.json", "feature_schema.json")]:
        require((package / name).read_bytes() == output("step11", "build", upstream_name).read_bytes(),
                "EXPORTED_FEATURE_CHANGED", "Packaged EHR or schema differs from the frozen source")
    watched = {context.layout.resolve(store.output_ref(m, entry["name"])) for m in manifests.values() for entry in m["outputs"]}
    watched.update(context.layout.resolve(m["self"]) for m in manifests.values())
    for stage, phase, filename in [("step10", "audit", "lab_audit.json"), ("step11", "audit", "ehr_audit.json")]:
        record = read_json(output(stage, phase, filename))
        require(record.get("data_run") == context.key and record.get("audit_checks_passed") is True,
                "SOURCE_AUDIT_NOT_PASSED", "A source audit is incomplete")
        for name, ref in record["sources"].items():
            path = context.layout.resolve(ref)
            require(_file_version(path) == record["source_versions"][name], "SOURCE_CHANGED", "An audited raw source changed")
            watched.add(path)
    result["watched"] = {path: _file_version(path) for path in watched}
    return result


def verify_master_provenance(context, store, root, manifests):
    required = {("step06", "reuse"), ("step07", "prepare"), ("step08", "freeze"), ("step12", "export")}
    require(set(manifests) == required, "INCOMPLETE_LINEAGE", "Master package lineage is incomplete")
    image, cohort = manifests[("step07", "prepare")], manifests[("step08", "freeze")]
    inputs = master_reuse.package_inputs(context, store, cohort, image)
    reuse = inputs["reuse"]
    require(root["inputs"] == [{"manifest": m["self"], "generation": m["generation"]} for m in (cohort, image)],
            "INCOMPLETE_LINEAGE", "Master export must bind the current cohort and images")
    def output(name):
        return master_reuse.published(store, root, name)
    export = read_json(output("training_package_export.json"))
    require(export.get("data_run") == export.get("release_id") == context.key
            and export.get("training_package_exported") is True and export.get("upstream") == root["inputs"]
            and export.get("independent_validation_completed") is False,
            "INVALID_EXPORT_RECORD", "A current-run export with bound upstream generations is required")
    package = output("package/samples.parquet").parent
    for name, original in (("ehr_features.parquet", inputs["ehr_path"]), ("feature_schema.json", inputs["schema_path"])):
        require((package / name).read_bytes() == original.read_bytes(),
                "EXPORTED_FEATURE_CHANGED", "Packaged EHR differs from the accepted copy")
    watched = {store.layout.resolve(m["self"]) for m in manifests.values()}
    for m in manifests.values():
        watched.update(master_reuse.published(store, m, x["name"]) for x in m["outputs"])
    def cached(name, step, source_name):
        return master_reuse.copied(store, reuse, "cache/" + name, step, "audit", "cache/" + source_name)
    return {
        **inputs, "manifests": manifests, "export_manifest": root, "export": export, "package": package,
        "archive": output("training_package.zip"), "split": inputs["patients"], "evidence": inputs["cohort_audit"],
        "samples": pq.read_table(output("package/samples.parquet")), "rules": checks.retained_rules(inputs["rules_table"]),
        "images": pq.read_table(output("image_manifest.parquet")),
        "qc": pq.read_table(master_reuse.published(store, image, "image_qc_status.parquet")),
        "labs": cached("labs_candidate_subset.parquet", "step10", "labs_candidate_subset.parquet"),
        "lab_admissions": pq.read_table(cached("lab_admissions.parquet", "step10", "admissions.parquet")),
        "static": {name: pq.read_table(cached(f"static_{name}.parquet", "step11", f"{name}.parquet"))
                   for name in ("patients", "admissions", "services")},
        "watched": {path: _file_version(path) for path in watched},
    }


def validate_sample_splits(source, settings):
    cv_files = source.get("cv_files", {})
    samples, cohort, primary = (source[k] for k in ("samples", "cohort", "split"))
    if not cv_files:
        return checks.validate_samples(samples, cohort, primary, settings)

    for name, path in cv_files.items():
        checks.require((source["package"] / name).read_bytes() == path.read_bytes(), "Packaged fold artifact differs from Step09")
    bundle = read_json(cv_files[split.CV_INDICES])
    split.validate_cv_indices(source.get("parent_cohort", cohort), bundle, primary)
    checks.require(bundle["split_version"] == bundle["cohort_version"], "Index data version changed")
    checks.require(samples["sample_id"].to_pylist() == [row["sample_id"] for row in bundle["master"]],
                   "Packaged samples differ from the study index order")
    folds = []
    for fold, column in zip(bundle["folds"], checks.FOLD_SPLIT_COLUMNS, strict=True):
        name = f"splits/fold_{fold['fold']}/{split.TABLE_NAME}"
        folds.append(pq.read_table(cv_files[name]))
        roles = samples[column].to_pylist()
        for role, indices in (("train", fold["train_indices"]), ("validation", fold["validation_indices"]),
                              ("test", bundle["test_indices"])):
            checks.require([i for i, value in enumerate(roles) if value == role] == indices,
                           f"Sample roles differ from Step09 indices for fold {fold['fold']}")
    return {**checks.validate_samples(samples, cohort, primary, settings, folds=folds),
            "folds_checked": len(folds), "fixed_test_studies": len(bundle["test_indices"])}


def validate_images(context, store, source):
    original, images, quality = (checks.indexed(table, "sample_id") for table in (source["cohort"], source["images"], source["qc"]))
    samples = source["samples"].to_pylist()
    checks.require(set(images) == {row["sample_id"] for row in samples}, "Image manifest coverage mismatch")
    prior = source["manifests"][("step07", "prepare")]
    split_columns = checks.FOLD_SPLIT_COLUMNS if source.get("cv_files") else ("split",)
    fields = "subject_id hadm_id study_id dicom_id t0 npy_relpath processing_version qc_status view_position".split()
    for row in samples:
        key = row["sample_id"]
        base, image, qc = original[key], images[key], quality[key]
        checks.require(qc["qc_status"] == "pass" and qc["processing_status"] == "complete", "Image QC not accepted")
        checks.require(all(base[name] == image[name] == qc[name] for name in fields), "Image identity or processing changed")
        checks.require(all(image[column] == row[column] for column in split_columns)
                       and image["npy_relpath"] == row["image_relpath"], "Image assignment changed")
        ref = store.output_ref(prior, row["image_relpath"])
        checks.require({"root": image["npy_root"], "path": image["npy_path"]} == ref, "Image source reference changed")
        checks.require(image["dicom_root"] == "mimic_cxr" and image["dicom_path"] == base["dicom_path"] == qc["dicom_path"], "DICOM reference changed")
        dicom = context.layout.path("mimic_cxr", image["dicom_path"])
        checks.require(_file_version(dicom) == json.loads(qc["source_version"]), "Source DICOM state changed")
        source["watched"][dicom] = _file_version(dicom)
        target = context.layout.resolve(ref)
        path = source["package"] / relative(row["image_relpath"])
        checks.require(target.read_bytes() == path.read_bytes(), "Packaged image differs from its accepted source")
        checks.validate_pixels(path, source["image_settings"]["size"])
        if "inherited_version" in source:
            pixels = process_dataset(pydicom.dcmread(dicom, force=False), source["image_settings"])["pixels"]
            checks.require(pixels is not None and np.array_equal(pixels, np.load(path, allow_pickle=False)),
                           "Master pixels do not reproduce the source DICOM")
    return {"images_checked": len(samples), "source_dicoms_checked": len(samples)}


def validate_metadata(context, source, settings):
    package = source["package"]
    metadata = portable(read_json(package / "dataset_metadata.json"))
    for name in ("release_id", "data_run", "cohort_version", "split_version", "dictionary_version", "feature_set_version"):
        expected_version = source.get("inherited_version", context.key) if name in {"split_version", "dictionary_version", "feature_set_version"} else context.key
        checks.require(metadata[name] == expected_version, "Metadata identity changed")
    if "inherited_version" in source:
        checks.require(metadata.get("parent_release_id") == source["inherited_version"], "Parent identity changed")
    checks.require(metadata["interface_version"] == settings["interface_version"] and metadata["horizon_hours"] == 72
                   and metadata["ehr_lookback_hours"] == 24 and metadata["target_column"] == "burden_score"
                   and metadata["target_range"] == [0, 8] and metadata["target_formula"] == source["decision"]["target_formula"]
                   and metadata["outcome_version"] == source["decision"]["outcome_version"]
                   and metadata["split_values"] == list(checks.SPLITS)
                   and metadata["split_unit"] == "subject_id", "Metadata task contract changed")
    cv_files = source.get("cv_files", {})
    if cv_files:
        checks.require("split_column" not in metadata and metadata.get("split_columns") == list(checks.FOLD_SPLIT_COLUMNS)
                       and metadata.get("cross_validation") == {
                           "indices_file": split.CV_INDICES, "n_folds": 5, "split_column_semantics": "per_fold",
                           "patient_split_files": sorted(name for name in cv_files if name != split.CV_INDICES),
                       }, "Five-fold metadata contract changed")
    else:
        checks.require(metadata.get("split_column") == "split" and "split_columns" not in metadata
                       and "cross_validation" not in metadata, "Holdout metadata contract changed")
    checks.require(metadata["hypoxemia_source"] == "MIMIC-IV ICU chartevents only"
                   and metadata["live_discharge_rule"] == "Retain events in (t0, dischtime] when alive discharge precedes t0+H; absent recorded events are operational zeros.",
                   "Metadata source or discharge rule changed")
    checks.require(metadata["image"] == {"format": "npy", "dtype": "float32", "shape": [source["image_settings"]["size"]] * 2, "value_range": [0.0,1.0], "model_normalization_applied": False}
                   and all(metadata["image_config"][name] == value for name,value in source["image_settings"].items()), "Metadata image policy changed")
    checks.require(metadata["split_rules"] == {name:source["split_settings"][name] for name in ("algorithm_version", "seed", "weights", "primary_stratum", "require_high_burden_in_each_split")}, "Split policy changed")
    checks.require(metadata["feature_counts"] == {"numeric":len(source["schema"]["numeric_features"]), "categorical":len(source["schema"]["categorical_features"])}
                   and metadata["ehr_preprocessing_applied"] is False and metadata["independent_validation_completed"] is False,
                   "Historical export state or feature counts changed")
    expected = PACKAGE_FILES | set(source.get("cv_files", {})) | set(source["samples"]["image_relpath"].to_pylist())
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
    checks.require(actual == expected and metadata["files"] == sorted(expected), "Package inventory changed")
    checks.require(source["export"]["studies"] == source["samples"].num_rows == source["export"]["images"]
                   and source["export"]["package_files"] == len(expected), "Export population or file count changed")
    checks.require(metadata["upstream"] == source["export_manifest"]["inputs"] == source["export"]["upstream"], "Export generation references changed")
    checks.require(source["export"]["package_inventory"] == [{"name":name,"bytes":(package/name).stat().st_size} for name in sorted(expected)], "Export inventory sizes changed")
    summaries = checks.expected_summary(source["samples"])
    with (package / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        checks.require(reader.fieldnames == ["split", "metric", "value", "unit", "denominator"], "Summary columns changed")
        checks.require(list(reader) == [{name:str(value) for name,value in row.items()} for row in summaries], "Summary values changed")
    counts = {group:{row["metric"]:row["value"] for row in summaries if row["split"] == group and row["metric"] in {"studies", "admissions", "patients"}} for group in dict.fromkeys(row["split"] for row in summaries)}
    checks.require(metadata["split_counts"] == source["export"]["split_counts"] == counts, "Population counts changed")
    return {"summary_rows_checked": len(summaries)}


def relocate_archive(source, workspace):
    package = source["package"]
    expected = PACKAGE_FILES | set(source.get("cv_files", {})) | set(source["samples"]["image_relpath"].to_pylist())
    with zipfile.ZipFile(source["archive"]) as archive:
        infos = archive.infolist()
        names = {f"package/{name}": name for name in expected}
        checks.require(len(infos) == len(names) and {info.filename for info in infos} == set(names)
                       and len({info.filename.casefold() for info in infos}) == len(infos), "ZIP duplicate or unexpected inventory")
        for info in infos:
            checks.require("\\" not in info.filename and ":" not in info.filename and not info.is_dir() and not info.flag_bits & 1
                           and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG), "Unsafe ZIP entry type or encryption")
            name = relative(names[info.filename])
            checks.require(info.filename == "package/" + name and info.file_size == (package / name).stat().st_size, "ZIP path or size changed")
        with tempfile.TemporaryDirectory(dir=workspace, prefix=".relocate_") as temporary:
            root = Path(temporary)
            for info in infos:
                path = root / relative(info.filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as left, (package / names[info.filename]).open("rb") as right, path.open("wb") as target:
                    remaining = info.file_size
                    while block := left.read(min(1024*1024, remaining+1)):
                        remaining -= len(block)
                        checks.require(remaining >= 0 and block == right.read(len(block)), "ZIP member bytes changed")
                        target.write(block)
                    checks.require(remaining == 0 and not right.read(1), "ZIP member length changed")
            data = root / "package"
            samples = pq.read_table(data / "samples.parquet")
            checks.require(samples.equals(source["samples"]), "Relocated samples changed")
            ehr, schema = pq.read_table(data / "ehr_features.parquet"), read_json(data / "feature_schema.json")
            checks.validate_ehr_contract(ehr, samples, schema, source["rules"])
            for row in samples.to_pylist():
                checks.validate_pixels(data / relative(row["image_relpath"]), source["image_settings"]["size"])
        checks.require(not root.exists(), "Relocation temporary directory was not removed")
    return {"relocated_members":len(infos), "relocated_studies":samples.num_rows, "relocated_images":samples.num_rows,
            "data_root_only":True, "temporary_directory_removed":True}


def verify_stability(source):
    checks.require(all(_file_version(path) == version for path,version in source["watched"].items()), "An input changed during acceptance")
    return {"watched_files":len(source["watched"]), "source_states_unchanged":True}


def error_detail(error, layout):
    detail = f"{type(error).__name__}: {error}"
    for name, root in sorted(layout.roots.items(), key=lambda item:len(str(item[1])), reverse=True):
        for form in (str(root), root.as_posix(), repr(str(root))[1:-1]):
            detail = detail.replace(form, name + ":")
    try:
        return portable(detail[:2000])
    except PipelineError:
        return type(error).__name__ + ": inspect the named check and local source configuration"


def run_validation(context, config_path="export/validation.json"):
    config_file = context.check_config_ref(context.config_ref(config_path))
    config_version = _file_version(config_file)
    settings = load_settings(context, config_path)
    store = ManifestStore(context.layout, data=context)
    ref = store.manifest_ref("step12", "export")
    initial_error = None
    try:
        store.validate(ref)
    except Exception as error:
        initial_error = error
    results = {name:{"status":"blocked", "detail":"Prerequisite checks not completed"} for name in checks.CHECKS}
    results["train_only_model_preprocessing"] = {"status":"not_applicable", "detail":"Train-fitted preprocessing and validation thresholds belong to modeling"}
    details = {"checks_passed":False, "training_package_accepted":False}
    passed = False
    with store.publish("step13", "validate", inputs=[] if initial_error else [ref], configs=[context.config_ref(config_path)],
                       details=details, required_bytes=settings["workspace_estimate_bytes"]) as work:
        source = {}
        def check(name, operation):
            print(f"Step13: {name}", flush=True)
            try:
                evidence = operation()
                results[name] = {"status":"pass", "evidence":evidence or {}}
                return True
            except Exception as error:
                results[name] = {"status":"blocked" if isinstance(error, FileNotFoundError) else "fail", "detail":error_detail(error, context.layout)}
                return False
        def provenance():
            if initial_error:
                raise initial_error
            source.update(verify_provenance(context, store))
            source["watched"][config_file] = config_version
            if "inherited_version" in source:
                return {"parent_release_id": source["inherited_version"], "copied_inputs_verified": True}
            return {"completed_steps":list(range(1,13)), "step11_audit_and_build":True}
        if check("provenance", provenance):
            check("samples", lambda:validate_sample_splits(source, settings))
            if check("ehr_contract", lambda:checks.validate_ehr_contract(source["ehr"], source["samples"], source["schema"], source["rules"])):
                check("laboratory_reconstruction", lambda:checks.reconstruct_labs(source["labs"], source["lab_admissions"], source["cohort"], source["rules"], source["ehr"]))
                check("static_reconstruction", lambda:checks.reconstruct_static(source["cohort"], source["evidence"], source["static"]["patients"], source["static"]["admissions"], source["static"]["services"], source["ehr"]))
            check("images", lambda:validate_images(context, store, source))
            check("metadata_summary", lambda:validate_metadata(context, source, settings))
            check("zip_relocation", lambda:relocate_archive(source, work))
            check("source_stability", lambda:verify_stability(source))
        for result in results.values():
            result.update(affected_rows=0 if result["status"] == "pass" else None,
                          affected_rows_count_status="complete" if result["status"] == "pass" else "not_computed")
        passed = all(results[name]["status"] == "pass" for name in checks.CHECKS)
        report = {"data_run":context.key, "release_id":context.key, "generated_utc":datetime.now(timezone.utc).isoformat(),
                  "settings":settings, "checks":results, "independent_validation_completed":all(results[name]["status"] != "blocked" for name in checks.CHECKS),
                  "validation_passed":passed, "training_package_accepted":passed,
                  "source_export":{"manifest":ref, "generation":source.get("export_manifest",{}).get("generation")}}
        write_json(work / "training_package_acceptance.json", report)
        write_json(work / "run_metadata.json", report)
        rows = [{"check":name, "status":result["status"], "mandatory":name in checks.CHECKS, "affected_rows":result["affected_rows"],
                 "detail":result.get("detail",""), "evidence":json.dumps(result.get("evidence",{}),sort_keys=True)} for name,result in results.items()]
        write_csv(work / "cross_file_checks.csv", ["check", "status", "mandatory", "affected_rows", "detail", "evidence"], rows)
        failures = [{name:row[name] for name in ("check", "status", "detail")} for row in rows if row["status"] in {"fail", "blocked"}][:settings["failure_detail_limit"]]
        write_csv(work / "failure_details.csv", ["check", "status", "detail"], failures)
        text = "# Independent training acceptance\n\n" + f"Data run: {context.key}. Accepted: {passed}.\n\n"
        text += "\n".join(f"- {row['check']}: {row['status']}. {row['detail']}" for row in rows)
        text += "\n\nReconstruction uses uncleaned laboratory and static source caches. Export metadata stays historical. "
        text += "A failed check stops at its first diagnostic; an empty affected-row count means the total was not computed. "
        text += "Only a successful mandatory acceptance can freeze this data run.\n"
        (work / "validation_report.md").write_text(text, encoding="utf-8", newline="\n")
        details.update(checks_passed=passed, training_package_accepted=passed)
    if passed:
        store.freeze(store.manifest_ref("step13", "validate"))
    return 0 if passed else 1
