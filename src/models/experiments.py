"""Batch configuration and discoverable output indexes for model experiments."""

import json

from src.common.config import load_model_config
from src.common.manifest import ManifestStore, file_lock
from src.common.paths import PipelineError, identifier, read_json, require


def experiment_store(layout, stage, modality, model, experiment):
    return ManifestStore(layout, family=modality, experiment=experiment, model_stage=stage, model=model)


def read_batch(layout, path, selected=None):
    document = load_model_config(layout, path)
    required = {"data_release", "defaults", "experiments"}
    require(set(document) == required and isinstance(document["defaults"], dict),
            "INVALID_MODEL_BATCH", "Expected data_release, defaults and experiments")
    entries = document["experiments"]
    require(isinstance(entries, list) and bool(entries), "INVALID_MODEL_BATCH", "At least one experiment is required")
    by_id = {}
    for entry in entries:
        require(isinstance(entry, dict) and {"id", "parameters"} <= set(entry) <= {"id", "description", "parameters"},
                "INVALID_MODEL_BATCH", "Each experiment needs id and parameters, with optional description")
        key = identifier(entry["id"])
        require(key == key.lower() and key not in by_id, "INVALID_MODEL_BATCH", "Experiment IDs must be unique and lowercase")
        description = entry.get("description", "")
        require(isinstance(description, str) and "\n" not in description and "\r" not in description,
                "INVALID_MODEL_BATCH", "Description must be one line")
        require(isinstance(entry["parameters"], dict), "INVALID_MODEL_BATCH", "Experiment parameters must be an object")
        by_id[key] = {"id": key, "description": description, "data_release": document["data_release"],
                      "parameters": {**document["defaults"], **entry["parameters"]}}
    if selected is not None:
        require(selected in by_id, "UNKNOWN_EXPERIMENT_ID", "Requested ID is absent from the configuration")
        return [by_id[selected]]
    return list(by_id.values())


def changed_parameters(previous, current, prefix=""):
    changes = []
    for key in sorted(previous.keys() | current.keys()):
        name = f"{prefix}{key}"
        a, b = previous.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.extend(changed_parameters(a, b, name + "."))
        elif a != b or (key in previous) != (key in current):
            value = json.dumps(b, ensure_ascii=False) if key in current else "(removed)"
            changes.append(f"{name}={value}")
    return changes


def update_index(root, stage):
    completed, pending = [], []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        stage_dir = folder / "steps" / stage / "run"
        state = folder / ".control" / f"{stage}.run.state.json"
        if (stage_dir / "manifest.json").is_file():
            manifest = read_json(stage_dir / "manifest.json")
            snapshot = read_json(stage_dir / "effective_configuration.json")
            completed.append((manifest, snapshot, stage_dir.relative_to(root).as_posix()))
        elif state.is_file():
            pending.append((folder.name, read_json(state)["status"]))
    completed.sort(key=lambda item: (item[0]["created_utc"], item[1]["id"]))
    lines = [f"# {stage.title()} experiment IDs", ""]
    if completed:
        baseline = completed[0][1]
        lines.extend([f"Parameter differences are relative to `{baseline['id']}`.", ""])
        for manifest, config, path in completed:
            details = changed_parameters(baseline["parameters"], config["parameters"])
            description = f" {config['description']}." if config["description"] else ""
            changes = "; ".join(details) or "same parameters as baseline"
            train_id = manifest["details"].get("train_id")
            upstream = f" Training ID: `{train_id}`." if train_id else ""
            lines.append(f"- **{config['id']}** · {manifest['status']} · {config['data_release']}.{description}{upstream} {changes}. [Artifacts]({path}/)")
    lines.extend(f"- **{key}** · {status}" for key, status in pending)
    temp = root / ".INDEX.md.tmp"
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(root / "INDEX.md")


def completed_result(store, ref, experiment, inputs):
    if not store.layout.resolve(ref).is_file():
        return None
    manifest = store.validate(ref)
    snapshot = read_json(store.layout.resolve(store.output_ref(manifest, "effective_configuration.json")))
    require(manifest["scope"] == store.scope and snapshot == experiment and manifest["inputs"] == inputs,
            "EXPERIMENT_ID_CONFLICT", "Completed ID has a different configuration or upstream binding; choose a new ID")
    return {"id": experiment["id"], "status": "unchanged", "manifest": ref}


def run_model_batch(layout, stage, modality, model, experiments, publish):
    root = layout.path("outputs", f"models/{stage}/{modality}/{model}")
    results = []
    with file_lock(root / ".batch.lock"):
        for experiment in experiments:
            try:
                result = publish(experiment)
            except PipelineError as error:
                result = {"id": experiment["id"], "status": "failed", "error": error.code}
            except Exception as error:
                result = {"id": experiment["id"], "status": "failed", "error": type(error).__name__}
            results.append(result)
        update_index(root, stage)
    return {"status": "failed" if any(item["status"] == "failed" for item in results) else "complete",
            "data_release": experiments[0]["data_release"], "experiments": results}
