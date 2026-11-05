"""Deterministic image preparation and local review with generation-bound inputs."""

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
from PIL import Image
from pydicom import config as dicom_config

from src.cohort.audit_support import portable_value
from src.cohort.burden_io import clinical_time
from src.cohort.burden_support import AuditWorkspace, write_csv
from src.cohort.raw_io import _acquisition_time, _file_version, _progress
from src.common.csv_io import _csv_reader
from src.common.manifest import ManifestStore, write_json
from src.common.paths import read_json, require
from src.imaging.preprocessing import (
    process_dataset,
    validate_pixels,
    validate_settings,
)
from src.utils import master_reuse

REVIEW_COLUMNS = "sample_id dicom_id decision rationale reviewer reviewed_on".split()
KEY_COLUMNS = "sample_id subject_id hadm_id study_id dicom_id dicom_path view_position t0".split()
STRING_COLUMNS = ['sample_id', 'dicom_id', 'dicom_path', 'view_position', 'source_version', 'transfer_syntax_uid', 'photometric_interpretation', 'modality_path', 'voi_path', 'presentation_path', 'display_metadata', 'processing_status', 'error_code', 'issues', 'processing_version', 'npy_relpath', 'before_relpath', 'after_relpath', 'before_method', 'duplicate_groups', 'auto_qc_status', 'qc_status', 'qc_reason', 'reviewer', 'reviewed_on', 'review_rationale']
INT_COLUMNS = (
    "original_height original_width resized_height resized_width pad_top pad_bottom pad_left pad_right"
).split()
FLOAT_COLUMNS = (
    "p1 p99 valid_pixel_fraction clipped_low_fraction clipped_high_fraction".split()
)
SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRING_COLUMNS]
    + [(name, pa.int64()) for name in ("subject_id", "hadm_id", "study_id")]
    + [("t0", pa.timestamp("us"))]
    + [(name, pa.int32()) for name in INT_COLUMNS]
    + [(name, pa.float64()) for name in FLOAT_COLUMNS]
    + [
        (name, pa.bool_())
        for name in (
            "inversion_applied",
            "review_required",
            "review_complete",
            "sampled_for_review",
        )
    ]
)


def load_inputs(context, store, *, parent_run=None):
    refs = [store.manifest_ref(step, phase) for step, phase in [("step02", "build"), ("step03", "select"), ("step06", "build")]]
    if parent_run is None:
        manifests = [store.validate(ref) for ref in refs]
        index_path, selection_path, label_path = [context.layout.resolve(store.output_ref(m, name)) for m, name in zip(
            manifests, ("cxr_index.parquet", "study_image_selection.parquet", "study_burden_labels.parquet"), strict=True)]
        metadata = read_json(context.layout.resolve(store.output_ref(manifests[-1], "run_metadata.json")))
        label_run = context.key
    else:
        reuse = master_reuse.reuse_manifest(context, store, parent_run)
        refs = [reuse["self"]]
        index_path = master_reuse.copied(store, reuse, "cxr_index.parquet", "step02", "build")
        selection_path = master_reuse.copied(store, reuse, "study_image_selection.parquet", "step03", "select")
        label_path = master_reuse.copied(store, reuse, "study_burden_labels.parquet", "step06", "build")
        metadata = read_json(master_reuse.copied(store, reuse, "labels_metadata.json", "step06", "build", "run_metadata.json"))
        label_run = f"v{context.version}_r{parent_run}"
    require(all(metadata.get(key) is True for key in ("formal_labels_published", "label_checks_passed", "horizon_frozen", "label_rules_frozen"))
            and metadata.get("horizon_hours") == 72 and metadata.get("data_run") == label_run,
            "LABELS_NOT_READY", "Formal 72-hour labels are required")
    labels = pq.read_table(
        label_path, columns=KEY_COLUMNS + ["label_status", "candidate_status"]
    ).to_pylist()
    if (
        len(labels) != metadata["counts"]["eligible_studies"]
        or len({r["sample_id"] for r in labels}) != len(labels)
        or len({(r["subject_id"], r["hadm_id"], r["study_id"]) for r in labels})
        != len(labels)
    ):
        raise ValueError("Formal label keys or counts are invalid")
    selected = {}
    for row in pq.read_table(selection_path).to_pylist():
        key = (row["subject_id"], row["study_id"])
        if key in selected:
            raise ValueError("Duplicate selected study")
        selected[key] = row
    index = {}
    columns = "subject_id study_id dicom_id dicom_path view_position acquisition_datetime rows columns".split()
    for row in pq.read_table(index_path, columns=columns).to_pylist():
        key = (row["subject_id"], row["study_id"], row["dicom_id"])
        if key in index:
            raise ValueError("Duplicate indexed image")
        index[key] = row
    imaging = context.read("imaging/raw_audit.json")
    dicom_dir = context.layout.path("mimic_cxr", imaging["raw_audit"]["dicom_directory"])
    matched = {}
    for row in labels:
        if (
            row["label_status"] != "complete"
            or row["candidate_status"] != "eligible"
            or any(
                type(row[k]) is not int or row[k] <= 0
                for k in ("subject_id", "hadm_id", "study_id")
            )
        ):
            raise ValueError("Invalid formal label eligibility")
        sample = f"p{row['subject_id']}_h{row['hadm_id']}_s{row['study_id']}"
        chosen = selected.get((row["subject_id"], row["study_id"]))
        indexed = index.get((row["subject_id"], row["study_id"], row["dicom_id"]))
        if (
            row["sample_id"] != sample
            or not chosen
            or not indexed
            or row["dicom_id"] != chosen["selected_dicom_id"]
            or row["view_position"] not in ("AP", "PA")
        ):
            raise ValueError("Formal labels do not match selected/indexed images")
        for other, time_field in (
            (chosen, "acquisition_datetime"),
            (indexed, "acquisition_datetime"),
        ):
            if any(row[k] != other[k] for k in ("dicom_path", "view_position")) or row[
                "t0"
            ] != clinical_time(other[time_field]):
                raise ValueError("Selected image path, view or t0 changed")
        path = context.layout.path("mimic_cxr", row["dicom_path"])
        tail = (
            f"p{row['subject_id']}",
            f"s{row['study_id']}",
            row["dicom_id"] + ".dcm",
        )
        standard = ("files", "p" + str(row["subject_id"])[:2], *tail)
        if (
            not re.fullmatch(r"[0-9a-fA-F-]+", row["dicom_id"])
            or not path.is_relative_to(dicom_dir)
            or path.relative_to(dicom_dir).parts not in (tail, standard)
        ):
            raise ValueError("Selected DICOM path/key mismatch")
        matched[sample] = {**indexed, "path": path}
    return sorted(labels, key=lambda r: r["sample_id"]), matched, refs


def load_reviews(context, path, run):
    result = {}
    if path is None:
        return result
    ref = context.config_ref(path)
    path = context.check_config_ref(ref)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = _csv_reader(stream, run.check_csv(path))
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames)
                and set(REVIEW_COLUMNS) <= set(reader.fieldnames), "INVALID_IMAGE_REVIEW", "Missing or duplicate review columns")
        for row in reader:
            require(None not in row, "INVALID_IMAGE_REVIEW", "Unexpected extra CSV fields")
            row = {key: row[key].strip() for key in REVIEW_COLUMNS}
            require(all(row.values()) and row["decision"] in {"pass", "exclude"} and row["sample_id"] not in result,
                    "INVALID_IMAGE_REVIEW", "Incomplete, invalid or duplicate review decision")
            date.fromisoformat(row["reviewed_on"])
            result[row["sample_id"]] = row
    return result


def _duplicate_fingerprint(path):
    # Content digests classify exact duplicates only; they never validate files or bind reviews.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_qc(records: list[dict], settings: dict, reviews: dict) -> list[dict]:
    """Sampling and duplicate review use imaging metadata only, never endpoint labels."""
    for row in records:
        row["_duplicate_groups"] = []
        row["sampled_for_review"] = False
        for field in (
            "reviewer",
            "reviewed_on",
            "review_rationale",
        ):
            row[field] = ""
    duplicates = []
    for field in ("_source_duplicate", "_decoded_duplicate", "_prepared_duplicate"):
        groups = defaultdict(list)
        for row in records:
            if row.get(field):
                groups[row[field]].append(row)
        for number, (digest, group) in enumerate(sorted(groups.items()), 1):
            if len(group) > 1:
                group_id = field.removeprefix("_") + ":group_" + str(number)
                for row in group:
                    row["_duplicate_groups"].append(group_id)
                    duplicates.append(
                        dict(
                            group_id=group_id,
                            duplicate_type=field.removeprefix("_"),
                            sample_id=row["sample_id"],
                        )
                    )
    strata = defaultdict(list)
    for row in records:
        if (
            row["processing_status"] == "complete"
            and not row["issues"]
            and not row["_duplicate_groups"]
        ):
            strata[
                tuple(
                    row[k]
                    for k in (
                        "view_position",
                        "photometric_interpretation",
                        "modality_path",
                        "voi_path",
                        "presentation_path",
                    )
                )
            ].append(row)
    for group in strata.values():
        ranked = sorted(
            group,
            key=lambda r: hashlib.sha256(
                f"{settings['review_seed']}:{r['sample_id']}".encode()
            ).hexdigest(),
        )
        for row in ranked[: settings["review_per_group"]]:
            row["sampled_for_review"] = True
    ids = {r["sample_id"] for r in records}
    if set(reviews) - ids:
        raise ValueError("Image review references an unpublished study")
    for row in records:
        reasons = list(row["issues"])
        if row["error_code"]:
            reasons.append(row["error_code"])
        if row["_duplicate_groups"]:
            reasons.append("duplicate_image")
        if row["sampled_for_review"]:
            reasons.append("stratified_sample")
        row["review_required"] = bool(reasons) or row["processing_status"] != "complete"
        row["auto_qc_status"] = (
            "fail"
            if row["processing_status"] == "fail"
            else "pending_review"
            if row["review_required"]
            else "pass"
        )
        row["qc_status"] = row["auto_qc_status"]
        row["qc_reason"] = ";".join(reasons) or "automatic_checks_passed"
        row["review_complete"] = not row["review_required"]
        decision = reviews.get(row["sample_id"])
        if decision:
            if decision["dicom_id"] != row["dicom_id"]:
                raise ValueError("Review DICOM identity differs from current sample")
            if decision["decision"] == "pass" and (
                row["processing_status"] != "complete" or not row.get("_prepared_duplicate")
            ):
                raise ValueError(
                    "Manual pass cannot bypass a hard failure or incomplete transform"
                )
            row.update(
                qc_status="pass" if decision["decision"] == "pass" else "fail",
                review_complete=True,
                reviewer=decision["reviewer"],
                reviewed_on=decision["reviewed_on"],
                review_rationale=decision["rationale"],
            )
    return duplicates


def _preview(run: AuditWorkspace, path: Path, pixels: np.ndarray, maximum: int) -> None:
    image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((maximum, maximum), Image.Resampling.BILINEAR)
    image.save(run.path(path), format="PNG")


def _gallery(records: list[dict]) -> str:
    cards = []
    for row in sorted(
        records, key=lambda r: (not r["review_required"], r["sample_id"])
    ):
        pictures = []
        for field, caption in (
            ("before_relpath", row["before_method"]),
            ("after_relpath", "Prepared float32 NPY, displayed as PNG"),
        ):
            if row.get(field):
                pictures.append(
                    f'<figure><img loading="lazy" src="{html.escape(row[field], quote=True)}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
                )
        cards.append(
            f'<article data-required="{str(row["review_required"]).lower()}"><h2>{html.escape(row["sample_id"])}</h2><p>{html.escape(row["view_position"])} | {html.escape(row["qc_status"])} | {html.escape(row["qc_reason"])}</p><p>Polarity: {html.escape(row.get("photometric_interpretation", ""))}; modality: {html.escape(row.get("modality_path", ""))}; VOI: {html.escape(row.get("voi_path", ""))}; presentation: {html.escape(row.get("presentation_path", ""))}.</p><div class="pair">{"".join(pictures) or "No reliable preview available; inspect the failure reason."}</div></article>'
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Image QC review</title><style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#17202a}header{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #aaa}article{background:white;padding:16px;margin:16px 0;border:1px solid #ccc}.pair{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;max-width:48%}img{max-width:100%;max-height:640px;background:#000}figcaption{max-width:640px}h2{font-size:18px}button{padding:8px;cursor:pointer}</style><header><h1>Image QC review</h1><p>Compare display direction, lung coverage, proportions, padding, clipping, artifacts, text and AP/PA. Outcome labels are hidden.</p><p>Record decisions in an current data-run review CSV using the generated template. A pass requires a validated NPY; failures need acknowledgment or correction.</p><button id="toggle">Show all images</button></header>"""
        + "".join(cards)
        + """<script>let requiredOnly=true;function update(){document.querySelectorAll('article').forEach(e=>e.hidden=requiredOnly&&e.dataset.required!=='true');document.querySelector('#toggle').textContent=requiredOnly?'Show all images':'Show required reviews only'}document.querySelector('#toggle').onclick=()=>{requiredOnly=!requiredOnly;update()};update()</script></html>\n"""
    )


def prepare_images(context, processing_path="imaging/processing.json", reviews_path=None):
    started = datetime.now(timezone.utc)
    settings = context.read(processing_path, required=("processing_version", "size", "reviews_path"))
    validate_settings(settings)
    reviews_path = settings["reviews_path"] if reviews_path is None else reviews_path
    require(reviews_path is None or isinstance(reviews_path, str), "INVALID_IMAGE_REVIEW", "Review path must be relative or null")
    store = ManifestStore(context.layout, data=context)
    parent_run = settings.get("parent_run")
    reuse = None
    if parent_run is not None:
        reuse = master_reuse.reuse_manifest(context, store, parent_run)
        master_reuse.parent_image_settings(store, reuse, settings)
    labels, index, inputs = load_inputs(context, store, parent_run=parent_run)
    configs = [context.config_ref(processing_path), context.config_ref("imaging/raw_audit.json")]
    if reviews_path is not None:
        configs.append(context.config_ref(reviews_path))
    details = {"horizon_hours": 72}
    if reuse is not None:
        details["parent_run"] = parent_run
    required_bytes = len(labels) * (settings["size"] ** 2 * 8 + settings["preview_size"] ** 2 * 8) + 8 * 1024 * 1024
    image_dir, qc_dir = Path("data"), Path("reports")
    with store.publish("step07", "prepare", inputs=inputs, configs=configs, details=details,
                       data_files=("image_processing.parquet", "image_qc_status.parquet"), data_directories=("images",),
                       required_bytes=required_bytes) as work:
        run = AuditWorkspace(work, context.layout)
        (work / "images").mkdir()
        reviews = load_reviews(context, reviews_path, run)
        records = []
        progress = _progress(
            "Prepare selected label images", unit="image", total=len(labels)
        )
        try:
            for label in labels:
                row = {k: label[k] for k in KEY_COLUMNS}
                row.update(
                    processing_status="fail",
                    error_code="",
                    issues=[],
                    _duplicate_groups=[],
                    sampled_for_review=False,
                    processing_version=settings["processing_version"],
                    before_method="unavailable",
                    modality_path="not_applied",
                    voi_path="not_applied",
                    presentation_path="not_applied",
                    _source_duplicate="",
                    _prepared_duplicate="",
                )
                info = index[row["sample_id"]]
                path = info["path"]
                version = _file_version(path)
                row["_source_version"] = version
                if not version["exists"]:
                    row["error_code"] = "missing_source_dicom"
                else:
                    try:
                        row["_source_duplicate"] = _duplicate_fingerprint(path)
                        dataset = pydicom.dcmread(path, force=False)
                    except Exception as error:
                        row["error_code"] = (
                            f"metadata_read_error:{type(error).__name__}"
                        )
                    else:
                        time, _, _ = _acquisition_time(dataset)
                        if (
                            clinical_time(time) != row["t0"]
                            or str(dataset.get("ViewPosition", "")).strip().upper()
                            != row["view_position"]
                            or dataset.get("Rows") != info["rows"]
                            or dataset.get("Columns") != info["columns"]
                        ):
                            raise ValueError(
                                f"DICOM source metadata changed: {row['sample_id']}"
                            )
                        row["transfer_syntax_uid"] = (
                            str(dataset.file_meta.get("TransferSyntaxUID", ""))
                            if hasattr(dataset, "file_meta")
                            else ""
                        )
                        result = process_dataset(dataset, settings)
                        pixels, before = result.pop("pixels"), result.pop("before")
                        row.update(result)
                        if pixels is not None:
                            relative = f"images/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}.npy"
                            target = image_dir / relative
                            staged = run.path(target)
                            with staged.open("wb") as stream:
                                np.save(stream, pixels, allow_pickle=False)
                            validate_pixels(np.load(staged, allow_pickle=False), settings["size"])
                            row.update(
                                npy_relpath=relative, _prepared_duplicate=_duplicate_fingerprint(staged)
                            )
                            relative = f"previews/after/{row['sample_id']}.png"
                            _preview(
                                run,
                                qc_dir / relative,
                                pixels,
                                settings["preview_size"],
                            )
                            row["after_relpath"] = relative
                        if before is not None:
                            relative = f"previews/before/{row['sample_id']}.png"
                            _preview(
                                run,
                                qc_dir / relative,
                                before,
                                settings["preview_size"],
                            )
                            row["before_relpath"] = relative
                if _file_version(path) != version:
                    raise ValueError("A source DICOM changed during processing")
                records.append(row)
                progress.update(1)
        finally:
            progress.close()
        duplicates = assign_qc(records, settings, reviews)
        required = [row for row in records if row["review_required"]]
        template = []
        for row in required:
            decision = reviews.get(row["sample_id"], {})
            template.append({"sample_id": row["sample_id"], "dicom_id": row["dicom_id"],
                             **{key: decision.get(key, "") for key in REVIEW_COLUMNS[2:]},
                             "qc_reason": row["qc_reason"], "auto_qc_status": row["auto_qc_status"]})
        write_csv(work / "image_qc_review.csv", REVIEW_COLUMNS + ["qc_reason", "auto_qc_status"], template)
        write_csv(work / "duplicate_images.csv", ["group_id", "duplicate_type", "sample_id"], duplicates)
        review_fields = "sample_id view_position photometric_interpretation modality_path voi_path presentation_path qc_reason".split()
        write_csv(work / "review_sample.csv", review_fields,
                  [{key: row.get(key) for key in review_fields} for row in records if row["sampled_for_review"]])
        failure_fields = "sample_id dicom_id processing_status error_code qc_status review_complete".split()
        write_csv(work / "dicom_failures.csv", failure_fields,
                  [{key: row.get(key) for key in failure_fields} for row in records if row["processing_status"] != "complete"])
        (work / "image_qc_review.html").write_text(_gallery(records), encoding="utf-8", newline="\n")
        serialized = []
        for row in records:
            serialized.append(portable_value({**{key: row.get(key) for key in SCHEMA.names},
                "source_version": json.dumps(row["_source_version"], sort_keys=True),
                "display_metadata": json.dumps(row.get("display_metadata", {}), sort_keys=True),
                "issues": json.dumps(row["issues"]), "duplicate_groups": json.dumps(row["_duplicate_groups"])}, context.layout))
        if reuse is not None:
            master_reuse.verify_qc(store, reuse, serialized)
        table = pa.Table.from_pylist(serialized, schema=SCHEMA)
        for name in ("image_processing.parquet", "image_qc_status.parquet"):
            pq.write_table(table, work / name, compression="zstd")
        require(all(_file_version(index[row["sample_id"]]["path"]) == row["_source_version"] for row in records),
                "SOURCE_CHANGED", "A source DICOM changed during image preparation")
        counts = {status: {"studies": sum(r["qc_status"] == status for r in records),
                          "admissions": len({r["hadm_id"] for r in records if r["qc_status"] == status}),
                          "patients": len({r["subject_id"] for r in records if r["qc_status"] == status})}
                  for status in ("pass", "fail", "pending_review")}
        metadata = {"data_run": context.key, "processing_version": settings["processing_version"],
                    "processing_settings": settings, "processing_config": context.config_ref(processing_path),
                    "reviews": context.config_ref(reviews_path) if reviews_path is not None else None,
                    "csv_tail_checks": run.csv_tail_summary(), "image_checks_passed": True,
                    "qc_review_complete": all(r["review_complete"] for r in records),
                    "common_cohort_frozen": False,
                    "ready_for_cohort_freeze": all(r["review_complete"] for r in records) and counts["pass"]["studies"] > 0 and counts["pending_review"]["studies"] == 0,
                    "total_studies": len(records), "total_admissions": len({r["hadm_id"] for r in records}),
                    "total_patients": len({r["subject_id"] for r in records}), "qc_counts": counts,
                    "processing_counts": dict(Counter(r["processing_status"] for r in records)),
                    "view_counts": dict(Counter(r["view_position"] for r in records)),
                    "review_required_studies": len(required), "unreviewed_studies": sum(not r["review_complete"] for r in records),
                    "sampled_studies": sum(r["sampled_for_review"] for r in records),
                    "duplicate_studies": len({r["sample_id"] for r in duplicates}),
                    "npy_count": sum(bool(r.get("npy_relpath")) for r in records),
                    "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "runtime_versions": {"numpy": np.__version__, "pillow": PIL.__version__, "pydicom": pydicom.__version__},
                    "decoder_handlers": {m.HANDLER_NAME: m.is_available() for m in dicom_config.pixel_data_handlers}}
        run.checks.write_report(work / "csv_tail_checks.csv")
        write_json(work / "run_metadata.json", metadata)
        write_json(work / "effective_configuration.json", {"data_run": context.key, "processing": settings, "reviews": list(reviews.values())})
        report = ("# Image preparation and QC\n\n"
                  f"Data run: {context.key}. Studies: {len(records)}. Prepared NPY: {metadata['npy_count']}.\n"
                  f"QC counts: {counts}. Required: {len(required)}. Unreviewed: {metadata['unreviewed_studies']}.\n\n"
                  "Open image_qc_review.html for local before/after previews. Copy image_qc_review.csv to the "
                  "current data-run configuration review path, fill decisions and rerun. A manual pass cannot "
                  "bypass incomplete processing. Content fingerprints classify duplicates only; they are not "
                  "file-integrity or review bindings and are not published. Sampling uses the historical "
                  "deterministic ranking. This step does not freeze the common cohort.\n")
        (work / "report.md").write_text(report, encoding="utf-8", newline="\n")
        details.update(image_checks_passed=True, qc_counts=counts, total_studies=len(records),
                       unreviewed_studies=metadata["unreviewed_studies"], ready_for_cohort_freeze=metadata["ready_for_cohort_freeze"])
    return 0
