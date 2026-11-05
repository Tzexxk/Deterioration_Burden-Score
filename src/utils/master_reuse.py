"""Published parent inputs for a master-image release."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.common.config import DataContext
from src.common.manifest import ManifestStore
from src.common.paths import read_json, require


def reuse_manifest(context, store, parent_run):
    require(type(parent_run) is int and 0 < parent_run < context.run,
            "INVALID_REUSE", "Select an older parent run of the same data version")
    manifest = store.validate(store.manifest_ref("step06", "reuse"))
    parent = ManifestStore(context.layout, data=DataContext(context.layout, context.version, parent_run))
    origin = manifest.get("origin") or {}
    require(manifest.get("action") == "copied" and origin.get("scope") == parent.scope
            and origin.get("manifest") == parent.manifest_ref("step13", "validate"),
            "INVALID_REUSE", "Copy manifest must originate from the selected parent acceptance")
    return manifest


def published(store, manifest, name):
    return store.layout.resolve(store.output_ref(manifest, name))


def copied(store, manifest, name, stage, phase, source_name=None, *, source_root=None):
    """Require the registered source identity without opening the parent's files."""
    parent_scope = manifest["origin"]["scope"]
    parent = ManifestStore(store.layout, data=DataContext(store.layout, parent_scope["version"], parent_scope["run"]))
    sources = [item for item in manifest["details"]["sources"] if item["name"] == name]
    ref = parent.manifest_ref(stage, phase)
    entry = store.output_ref(manifest, name)
    filename = source_name or name
    scope = f"v{parent_scope['version']}_r{parent_scope['run']}"
    root = source_root or entry["root"]
    directory = f"{scope}/steps/{stage}/{phase}" if root == "processed" else f"data/{scope}/steps/{stage}/{phase}"
    expected = store.layout.ref(root, f"{directory}/{filename}")
    require(len(sources) == 1 and sources[0]["manifest"] == ref and sources[0]["source"] == expected,
            "INVALID_REUSE", f"Copied artifact has unexpected provenance: {name}")
    return store.layout.resolve(entry)


def parent_image_settings(store, reuse, settings):
    original = read_json(copied(store, reuse, "parent_image_settings.json", "step07", "prepare",
                                "effective_configuration.json"))["processing"]
    fields = {"size", "processing_version", "parent_run", "reviews_path"}
    require(original.get("size") == 224 and settings.get("size") == 1024
            and {k: v for k, v in original.items() if k not in fields}
            == {k: v for k, v in settings.items() if k not in fields},
            "INVALID_MASTER_POLICY", "Master must inherit parent image rules except for size and identity")


def verify_qc(store, reuse, records):
    original = pq.read_table(copied(store, reuse, "image_qc_status.parquet", "step07", "prepare")).to_pylist()
    old = {row["sample_id"]: row for row in original}
    new = {row["sample_id"]: row for row in records}
    require(len(old) == len(original) and len(new) == len(records) and old.keys() == new.keys(),
            "MASTER_QC_CHANGED", "Master QC does not cover the exact parent samples")
    fields = (
        "subject_id", "hadm_id", "study_id", "dicom_id", "dicom_path", "t0", "view_position", "source_version",
        "processing_status", "error_code", "issues", "qc_status", "review_complete",
        "reviewer", "reviewed_on", "review_rationale",
    )
    def normalized(row, field):
        value = row[field]
        return json.loads(value) if field in {"source_version", "issues"} else value

    for key, row in new.items():
        require(all(normalized(row, f) == normalized(old[key], f) for f in fields),
                "MASTER_QC_CHANGED", f"Master QC or source changed for {key}; inspect before reusing the decision")
    def groups(rows):
        memberships = {}
        for key, row in rows.items():
            for group in json.loads(row["duplicate_groups"]):
                memberships.setdefault(group, set()).add(key)
        return {(group.split(":", 1)[0], frozenset(ids)) for group, ids in memberships.items()}

    require(groups(old) == groups(new), "MASTER_QC_CHANGED", "Image duplicate groups changed; review before reuse")

def rebuild_cohort(context, store, image):
    from src.cohort.freeze_cohort import QC_FIELDS

    parent_run = image["details"]["parent_run"]
    reuse = reuse_manifest(context, store, parent_run)
    require(image["inputs"] == [{"manifest": reuse["self"], "generation": reuse["generation"]}],
            "INVALID_REUSE", "Image generation must bind the copied inputs")
    snapshot = read_json(published(store, image, "effective_configuration.json"))
    settings = snapshot["processing"]
    require(snapshot.get("data_run") == context.key and settings.get("parent_run") == parent_run
            and image["details"].get("ready_for_cohort_freeze") is True,
            "MASTER_QC_CHANGED", "A checked master image publication is required")
    parent_image_settings(store, reuse, settings)
    table = pq.read_table(published(store, image, "image_qc_status.parquet"))
    require(table.equals(pq.read_table(published(store, image, "image_processing.parquet"))),
            "MASTER_QC_CHANGED", "Master QC tables differ")
    records = table.to_pylist()
    verify_qc(store, reuse, records)
    quality = {r["sample_id"]: r for r in records}
    parent = pq.read_table(copied(store, reuse, "cohort.parquet", "step08", "freeze"))
    audit = pq.read_table(copied(store, reuse, "cohort_audit.parquet", "step08", "freeze"))
    included = {r["sample_id"] for r in records if r["qc_status"] == "pass"}
    require(len(included) == parent.num_rows and included == set(parent["sample_id"].to_pylist())
            and set(quality) == {r["sample_id"] for r in audit.to_pylist() if r["qc_performed"] == 1},
            "MASTER_COHORT_CHANGED", "Master QC membership must match the accepted parent")
    def update(table):
        rows = table.to_pylist()
        for row in rows:
            row["cohort_version"] = context.key
            if q := quality.get(row["sample_id"]):
                row.update({k: q[v] for k, v in QC_FIELDS.items()})
        return pa.Table.from_pylist(rows, schema=table.schema)

    return reuse, parent, update(parent), update(audit), settings, records


def package_inputs(context, store, cohort_manifest, image):
    from src.cohort import create_patient_split as split
    from src.data import package_contract
    from src.ehr import feature_engine as engine

    reuse, parent, cohort, audit, image_settings, records = rebuild_cohort(context, store, image)
    require(cohort_manifest["details"].get("parent_run") == image["details"]["parent_run"]
            and cohort_manifest["inputs"] == [{"manifest": m["self"], "generation": m["generation"]} for m in (reuse, image)]
            and cohort.equals(pq.read_table(published(store, cohort_manifest, "cohort.parquet")))
            and audit.equals(pq.read_table(published(store, cohort_manifest, "cohort_audit.parquet"))),
            "MASTER_COHORT_CHANGED", "Frozen master cohort does not reproduce its copied sources")
    def package(name):
        return copied(store, reuse, name, "step12", "export", "package/" + name, source_root="processed")
    cv_files = {split.CV_INDICES: package(split.CV_INDICES)}
    for i in range(1, 6):
        name = f"splits/fold_{i}/{split.TABLE_NAME}"
        cv_files[name] = package(name)
    folds = [pq.read_table(cv_files[f"splits/fold_{i}/{split.TABLE_NAME}"]) for i in range(1, 6)]
    bundle = read_json(cv_files[split.CV_INDICES])
    split.validate_cv_indices(parent, bundle, folds[0])
    samples = pq.read_table(package("samples.parquet"))
    require(package_contract.build_samples(cohort, folds[0], fold_patients=folds).equals(samples),
            "MASTER_COHORT_CHANGED", "Master package would change the accepted samples or assignments")
    for fold in range(1, 6):
        require(split.select_fold(folds[0], bundle, fold).equals(folds[fold - 1]),
                "SPLIT_MISMATCH", "Copied fold table does not reproduce its indices")
    parent_key = f"v{context.version}_r{image['details']['parent_run']}"
    schema_path, ehr_path = package("feature_schema.json"), package("ehr_features.parquet")
    schema = read_json(schema_path)
    ehr = pq.read_table(ehr_path)
    dictionary = pq.read_table(copied(store, reuse, "lab_dictionary.parquet", "step10", "freeze"))
    rules = engine.validate_dictionary(dictionary, parent_key)
    require(schema == engine.schema_for(rules, parent_key), "FEATURE_SCHEMA_MISMATCH", "Copied EHR schema changed")
    engine.validate_wide(ehr, cohort, schema)
    decision = read_json(copied(store, reuse, "endpoint_decision.json", "step05", "freeze"))
    meta = read_json(package("dataset_metadata.json"))
    require(meta.get("data_run") == parent_key and meta.get("target_formula") == decision.get("target_formula")
            and bundle.get("split_version") == parent_key and decision.get("checks_passed") is True,
            "INVALID_REUSE", "Parent feature, split or endpoint identity changed")
    return {"cohort": cohort, "cohort_audit": audit, "parent_cohort": parent,
            "patients": folds[0], "cv_files": cv_files, "qc_records": records, "schema": schema, "ehr": ehr,
            "ehr_path": ehr_path, "schema_path": schema_path, "rules_table": dictionary,
            "image_settings": image_settings, "split_settings": meta["split_rules"], "decision": decision,
            "image_manifest": image, "reuse": reuse, "inherited_version": parent_key}
