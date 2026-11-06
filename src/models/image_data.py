"""Read pixels and cohort metadata from a frozen image release."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PipelineError, identifier, read_json, relative, require
from src.data.package_contract import SPLITS
from src.models.released_cohort import load_cv, load_samples
from src.models.released_data import ReleasedData
from src.utils.build_image_variants import size_folder


@dataclass
class ImagePackage:
    release: ReleasedData
    samples: pd.DataFrame
    cv: dict
    shape: tuple[int, int]
    image_root: Path
    source: dict

    def read_image(self, index):
        path = self.image_root / self.samples.iloc[index].image_relpath
        require(path.is_file() and path.resolve() == path, "INVALID_IMAGE_PATH", "Expected a physical NPY image")
        try:
            image = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise PipelineError("INVALID_IMAGE_PIXELS", "Cannot read the released NPY image") from error
        require(isinstance(image, np.ndarray) and image.shape == self.shape and image.dtype == np.float32
                and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1,
                "INVALID_IMAGE_PIXELS", "Expected declared shape and finite float32 pixels in [0, 1]")
        return image


def image_source(release, selection, names, master_shape):
    key = release.store.data.key
    layout = release.store.layout
    require(isinstance(selection, dict), "INVALID_IMAGE_SOURCE", "Select a package or variant image source")
    kind = selection.get("kind")
    if kind == "package":
        require(set(selection) == {"kind"}, "INVALID_IMAGE_SOURCE", "Package images use their declared shape")
        return release.package_file("samples.parquet").parent, master_shape, {"kind": kind}
    require(kind == "variant" and set(selection) == {"kind", "batch", "size"},
            "INVALID_IMAGE_SOURCE", "Variant images require batch and size")
    batch = identifier(selection["batch"])
    size = selection["size"]
    require(type(size) is int and size > 0, "INVALID_IMAGE_SOURCE", "Select a positive declared variant size")
    root = layout.roots["processed"] / key / "image_variants" / f"batch-{batch}"
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file() and manifest_path.resolve() == manifest_path,
            "VARIANT_NOT_RELEASED", "A physical frozen variant manifest is required")
    manifest = read_json(manifest_path)
    expected = {"data_run": key, "acceptance": {"manifest": release.acceptance["self"],
                "generation": release.acceptance["generation"]}, "export": release.acceptance["inputs"][0],
                "samples": release.store.output_ref(release.export, "package/samples.parquet")}
    require(manifest.get("schema") == 1 and manifest.get("status") == "released" and manifest.get("batch") == batch
            and manifest.get("source") == expected and manifest.get("image_dtype") == "float32"
            and manifest.get("value_range") == [0.0, 1.0] and manifest.get("images_per_size") == len(names),
            "INVALID_VARIANT_MANIFEST", "Variant batch must bind the exact frozen package and image contract")
    sizes = manifest.get("sizes")
    require(isinstance(sizes, list) and all(type(value) is int and value > 0 for value in sizes)
            and len(sizes) == len(set(sizes)) and size in sizes,
            "VARIANT_SIZE_NOT_RELEASED", "The requested size is absent from the released variant manifest")
    directory = root / size_folder(size)
    require(directory.is_dir() and directory.resolve() == directory, "INVALID_IMAGE_PATH", "Variant size directory is missing or redirected")
    require({path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()} == names,
            "VARIANT_INVENTORY_MISMATCH", "The variant must cover exactly the frozen sample image paths")
    return directory, (size, size), {**selection, "manifest": layout.ref(
        "processed", manifest_path.relative_to(layout.roots["processed"]).as_posix()), "release": manifest}


def load_image_cohort(release):
    metadata = read_json(release.package_file("dataset_metadata.json"))
    key = release.store.data.key
    inherited = metadata.get("parent_release_id", key)
    require(metadata.get("release_id") == metadata.get("data_run") == metadata.get("cohort_version") == key
            and metadata.get("split_version") == inherited and metadata.get("interface_version") == "1.0"
            and metadata.get("target_column") == "burden_score" and metadata.get("horizon_hours") == 72
            and metadata.get("split_unit") == "subject_id" and metadata.get("split_values") == list(SPLITS),
            "INVALID_IMAGE_PACKAGE", "Unexpected frozen release, label or split contract")
    sample_table, samples = load_samples(release)
    cv = load_cv(release, metadata, sample_table, inherited)
    return metadata, samples, cv


def load_image_package(release, selection):
    metadata, samples, cv = load_image_cohort(release)
    image = metadata.get("image", {})
    shape = image.get("shape")
    require(isinstance(shape, list) and len(shape) == 2 and all(type(size) is int and size > 0 for size in shape)
            and image.get("format") == "npy" and image.get("dtype") == "float32"
            and image.get("value_range") == [0.0, 1.0] and image.get("model_normalization_applied") is False,
            "INVALID_IMAGE_PACKAGE", "Expected NPY image metadata with a declared float32 shape and range")
    paths = samples.image_relpath.tolist()
    require(len({name.casefold() for name in paths}) == len(paths)
            and all(relative(name) == name and name.startswith("images/") and name.endswith(".npy") for name in paths),
            "INVALID_IMAGE_PATH", "Each sample must refer to a unique relative NPY image path")
    names = set(paths)
    registered = {item["name"].removeprefix("package/") for item in release.export["outputs"]
                  if item["name"].startswith("package/images/") and item["root"] == "processed"}
    require(registered == names, "INVALID_IMAGE_PACKAGE", "Package image inventory must cover the frozen samples")
    directory, resolved_shape, source = image_source(release, selection, names, tuple(shape))
    return ImagePackage(release, samples, cv, resolved_shape, directory, source)
