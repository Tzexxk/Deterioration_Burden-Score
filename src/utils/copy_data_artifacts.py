"""Copy selected published files between data runs with explicit provenance."""

import shutil

from src.common.config import DataContext, load_utility_config
from src.common.manifest import ManifestStore, file_lock, write_json
from src.common.paths import identifier, positive, read_json, relative, require


def validate_plan(plan, current_run):
    require(isinstance(plan, dict) and set(plan) == {"source_run", "target", "items"},
            "INVALID_COPY_PLAN", "Copy plan requires source_run, target and items")
    positive(plan["source_run"])
    require(plan["source_run"] < current_run, "INVALID_REUSE", "Copy source must be an older run of the same data version")
    target = plan["target"]
    require(isinstance(target, dict) and set(target) == {"stage", "phase"},
            "INVALID_COPY_PLAN", "Target requires stage and phase")
    require(isinstance(target["stage"], str) and target["stage"] in {f"step{i:02d}" for i in range(1, 14)}, "INVALID_STAGE", "Data stages are step01 through step13")
    identifier(target["phase"])
    items = plan["items"]
    require(isinstance(items, list) and bool(items), "INVALID_COPY_PLAN", "Copy items must be a nonempty list")
    names = []
    for item in items:
        require(isinstance(item, dict) and set(item) == {"source", "target"},
                "INVALID_COPY_PLAN", "Each item requires source and target")
        source, dest = item["source"], item["target"]
        require(isinstance(source, dict) and set(source) == {"stage", "phase", "name"}
                and isinstance(dest, dict) and set(dest) == {"name", "root"},
                "INVALID_COPY_PLAN", "Invalid source or target fields")
        require(isinstance(source["stage"], str) and source["stage"] in {f"step{i:02d}" for i in range(1, 14)}, "INVALID_STAGE", "Source stage is invalid")
        identifier(source["phase"])
        require(relative(source["name"]) != ".", "INVALID_COPY_PLAN", "Source must name a file")
        require(dest["root"] in ("processed", "outputs"), "INVALID_COPY_PLAN", "Target root must be processed or outputs")
        name = relative(dest["name"]).casefold()
        require(name not in (".", "manifest.json", "copy_plan.json"), "INVALID_COPY_PLAN", "Target name is reserved")
        require(not any(name == old or name.startswith(old + "/") or old.startswith(name + "/") for old in names),
                "COPY_TARGET_COLLISION", "Copy targets must be unique and cannot overlap")
        names.append(name)
    return plan


def copy_sources(source_store, items):
    """Read-only preflight; callers hold the source writer lock while copying."""
    release_path = source_store.control / "release.json"
    require(release_path.is_file(), "SOURCE_NOT_RELEASED", "Copy source must be a frozen release")
    release = read_json(release_path)
    acceptance_ref = source_store.manifest_ref("step13", "validate")
    require(release.get("scope") == source_store.scope and release.get("status") == "released"
            and release.get("acceptance") == acceptance_ref,
            "SOURCE_NOT_RELEASED", "Copy release identity or acceptance reference is invalid")
    acceptance = source_store.validate(acceptance_ref)
    require(release.get("generation") == acceptance["generation"],
            "UPSTREAM_REBUILT", "Copy release acceptance generation differs")
    lineage = {}

    def visit(manifest):
        key = manifest["self"]["path"]
        if key in lineage:
            return
        lineage[key] = manifest
        for item in manifest["inputs"]:
            visit(source_store.validate(item["manifest"]))

    visit(acceptance)
    selected = []
    for item in items:
        stage, phase, name = (item["source"][key] for key in ("stage", "phase", "name"))
        root = item["target"]["root"]
        ref = source_store.manifest_ref(stage, phase)
        require(ref["path"] in lineage, "SOURCE_NOT_ACCEPTED", "Required copy stage is outside the accepted graph")
        manifest = lineage[ref["path"]]
        source = source_store.output_ref(manifest, name)
        path = source_store.layout.resolve(source)
        require(not path.is_symlink(), "INVALID_REUSE", "Copy source cannot be a symbolic link")
        selected.append({"name": item["target"]["name"], "root": root, "source": source,
                         "manifest": ref, "generation": manifest["generation"], "bytes": path.stat().st_size})
    return acceptance, selected



def copy_data_artifacts(context, config):
    plan = validate_plan(load_utility_config(context.layout, config), context.run)
    source_run = plan["source_run"]
    source_store = ManifestStore(context.layout, data=DataContext(context.layout, context.version, source_run))
    target_store = ManifestStore(context.layout, data=context)
    stage, phase = plan["target"]["stage"], plan["target"]["phase"]
    with file_lock(source_store.control / "writer.lock"):
        source_store._recover_unlocked()
        acceptance, selected = copy_sources(source_store, plan["items"])
        origin = {"manifest": acceptance["self"], "scope": acceptance["scope"], "generation": acceptance["generation"]}
        with target_store.publish(
            stage, phase, action="copied", origin=origin, utility_configs=[context.layout.ref("project", config)],
            details={"sources": selected},
            data_files=[item["name"] for item in selected if item["root"] == "processed"],
            required_bytes=sum(item["bytes"] for item in selected) + len(str(plan).encode("utf-8")) * 4 + 16384,
        ) as work:
            write_json(work / "copy_plan.json", plan)
            for item in selected:
                destination = work / relative(item["name"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(context.layout.resolve(item["source"]), destination)
    ref = target_store.manifest_ref(stage, phase)
    manifest = target_store.validate(ref)
    return {"status": "complete", "data_run": context.key, "generation": manifest["generation"],
            "manifest": ref, "copied_files": len(selected), "source_run": source_run}
