"""Build and verify frozen image-only variants of a released 1024-pixel master."""

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import numpy as np
import PIL
import pyarrow.parquet as pq
from PIL import Image

from src.common.config import load_utility_config
from src.common.manifest import ManifestStore, file_lock, write_json
from src.common.paths import identifier, read_json, relative, require
from src.imaging.preprocessing import validate_pixels

MASTER_SIZE = 1024
RECIPE = "pillow_float32_bilinear_full_square_reducing_gap_none"


def load_plan(layout, path):
    plan = load_utility_config(layout, path)
    require(set(plan) == {"batch", "sizes", "workers"}, "INVALID_VARIANT_PLAN",
            "Specify batch, sizes and workers")
    batch = identifier(plan["batch"])
    require(batch == batch.lower() and len(batch) <= 32, "INVALID_VARIANT_PLAN",
            "Batch must be lowercase and at most 32 characters")
    sizes, workers = plan["sizes"], plan["workers"]
    require(isinstance(sizes, list) and sizes
            and all(type(size) is int and 2 <= size < MASTER_SIZE for size in sizes)
            and len(sizes) == len(set(sizes)), "INVALID_VARIANT_PLAN",
            "Supply unique integer sizes from 2 to 1023")
    require(type(workers) is int and 1 <= workers <= 8, "INVALID_VARIANT_PLAN",
            "Workers must be an integer from 1 to 8")
    return batch, sorted(sizes), workers


def released_master(context):
    layout = context.layout
    store = ManifestStore(layout, data=context)
    release_path = store.control / "release.json"
    acceptance_ref = store.manifest_ref("step13", "validate")
    require(release_path.is_file(), "MASTER_NOT_RELEASED", "Run Step13 before deriving images")
    release = read_json(release_path)
    require(release.get("scope") == store.scope and release.get("status") == "released"
            and release.get("acceptance") == acceptance_ref,
            "MASTER_NOT_RELEASED", "Release must bind the selected run's Step13")
    acceptance = store.validate(acceptance_ref)
    require(release.get("generation") == acceptance["generation"]
            and acceptance["details"].get("checks_passed") is True
            and acceptance["details"].get("training_package_accepted") is True,
            "MASTER_NOT_RELEASED", "Step13 acceptance is incomplete or stale")
    export_ref = store.manifest_ref("step12", "export")
    export = read_json(layout.resolve(export_ref))
    export_binding = {"manifest": export_ref, "generation": export["generation"]}
    require(acceptance["inputs"] == [export_binding], "INVALID_MASTER", "Step13 must bind the master export")

    metadata = read_json(layout.resolve(store.output_ref(export, "package/dataset_metadata.json")))
    require(metadata.get("data_run") == metadata.get("release_id") == context.key
            and metadata.get("image") == {"format": "npy", "dtype": "float32",
                                          "shape": [MASTER_SIZE, MASTER_SIZE], "value_range": [0.0, 1.0],
                                          "model_normalization_applied": False},
            "INVALID_MASTER", "Expected a released 1024 x 1024 float32 master")
    samples_ref = store.output_ref(export, "package/samples.parquet")
    samples = pq.read_table(layout.resolve(samples_ref), columns=["sample_id", "image_relpath"])
    rows = samples.to_pylist()
    sample_ids, images = set(), {}
    for row in rows:
        sample, name = row["sample_id"], row["image_relpath"]
        require(isinstance(sample, str) and sample and sample not in sample_ids, "INVALID_MASTER",
                "Each master sample needs a unique sample ID")
        require(relative(name) == name and name.startswith("images/") and PurePosixPath(name).suffix == ".npy"
                and name.casefold() not in images, "INVALID_MASTER", "Each master sample needs a unique image path")
        sample_ids.add(sample)
        images[name.casefold()] = name
    names = set(images.values())
    registered = {entry["name"] for entry in export["outputs"]
                  if entry["name"].startswith("package/images/")}
    require(names and registered == {f"package/{name}" for name in names},
            "INVALID_MASTER", "Master images must exactly cover the frozen sample table")
    sources = {name: layout.resolve(store.output_ref(export, f"package/{name}")) for name in sorted(names)}
    package = layout.resolve(samples_ref).parent
    require(all(path == package / name for name, path in sources.items()),
            "INVALID_MASTER", "Master images must be physical files in the frozen package")
    parent = {"data_run": context.key, "acceptance": {"manifest": acceptance_ref, "generation": acceptance["generation"]},
              "export": export_binding, "samples": samples_ref}
    return parent, sources


def derive_image(path, size):
    master = np.load(path, allow_pickle=False)
    validate_pixels(master, MASTER_SIZE)
    image = Image.fromarray(master).resize((size, size), Image.Resampling.BILINEAR, reducing_gap=None)
    pixels = np.asarray(image, dtype=np.float32)
    validate_pixels(pixels, size)
    require(np.ptp(pixels) > 0, "CONSTANT_VARIANT_IMAGE", "A derived image has no pixel variation")
    return pixels


def size_folder(size):
    return f"{size}x{size}"


def generate_size(folder, sources, size):
    target = folder / size_folder(size)
    for name, master in sources.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, derive_image(master, size), allow_pickle=False)


def validate_size(folder, sources, size):
    target = folder / size_folder(size)
    files = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    require(files == sources.keys(), "VARIANT_INVENTORY_MISMATCH",
            f"Image inventory differs from master samples for {size_folder(size)}")
    for name, master in sources.items():
        path = target / name
        require(path.resolve() == path, "INVALID_VARIANT_PATH", "Variant images must be physical files")
        pixels = np.load(path, allow_pickle=False)
        validate_pixels(pixels, size)
        require(np.array_equal(pixels, derive_image(master, size)), "VARIANT_PIXELS_MISMATCH",
                f"Stored pixels do not reproduce the master for {size_folder(size)}")


def validate_batch(folder, sources, sizes, workers, expected, *, frozen):
    entries = {size_folder(size) for size in sizes}
    if frozen:
        entries.add("manifest.json")
        path = folder / "manifest.json"
        require(path.resolve() == path, "INVALID_VARIANT_PATH", "Variant manifest must be a physical file")
        actual = read_json(path)
        require(isinstance(actual, dict) and isinstance(actual.get("pillow_version"), str)
                and {**actual, "pillow_version": expected["pillow_version"]} == expected,
                "INVALID_VARIANT_MANIFEST", "Frozen batch does not match the plan and master release")
    require({path.name for path in folder.iterdir()} == entries, "VARIANT_INVENTORY_MISMATCH",
            "Variant batch has unexpected entries")
    with ThreadPoolExecutor(max_workers=min(workers, len(sizes))) as pool:
        list(pool.map(lambda size: validate_size(folder, sources, size), sizes))


def run_variants(context, config, *, validate_only=False):
    layout = context.layout
    batch, sizes, workers = load_plan(layout, config)
    parent, sources = released_master(context)
    root = layout.roots["processed"] / context.key / "image_variants"
    destination, staging = root / f"batch-{batch}", root / f".batch-{batch}.staging"
    lock = root / ".writer.lock"
    require(all(path.resolve() == path for path in (root, destination, staging, lock)),
            "INVALID_VARIANT_PATH", "Variant directories and lock cannot be redirected")
    manifest = {"schema": 1, "status": "released", "batch": batch, "source": parent,
                "sizes": sizes, "recipe": RECIPE, "pillow_version": PIL.__version__,
                "image_dtype": "float32", "value_range": [0.0, 1.0], "images_per_size": len(sources)}
    with file_lock(lock):
        if validate_only:
            require(destination.is_dir(), "VARIANT_NOT_RELEASED", "Generate this batch before validating it")
            validate_batch(destination, sources, sizes, workers, manifest, frozen=True)
        else:
            require(not destination.exists(), "VARIANT_IMMUTABLE",
                    "This batch is frozen; validate it or choose a new batch name")
            if staging.exists():
                shutil.rmtree(staging)
            estimate = len(sources) * sum(4 * size * size + 4096 for size in sizes) + 1024 ** 2
            require(shutil.disk_usage(root).free >= estimate, "INSUFFICIENT_DISK_SPACE",
                    "Insufficient space to stage all configured image sizes")
            staging.mkdir()
            try:
                with ThreadPoolExecutor(max_workers=min(workers, len(sizes))) as pool:
                    list(pool.map(lambda size: generate_size(staging, sources, size), sizes))
                validate_batch(staging, sources, sizes, workers, manifest, frozen=False)
                write_json(staging / "manifest.json", manifest)
                staging.rename(destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
    ref = layout.ref("processed", f"{context.key}/image_variants/batch-{batch}/manifest.json")
    return {"status": "validated" if validate_only else "released", "data_run": context.key,
            "batch": batch, "sizes": sizes, "images_per_size": len(sources), "manifest": ref}
