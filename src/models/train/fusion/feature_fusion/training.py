"""Fold-specific image representations appended to the unchanged EHR protocol."""

from dataclasses import replace
from importlib.metadata import version

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.common.manifest import write_json
from src.common.paths import identifier, read_json, require
from src.models.assessment.comparison import load_predictions
from src.models.assessment.evaluation import training_source
from src.models.ehr_data import checked_input_summary, load_package
from src.models.experiments import (
    completed_result,
    experiment_store,
    read_batch,
    run_model_batch,
)
from src.models.image_data import load_image_package
from src.models.released_data import ReleasedData
from src.models.train.ehr.catboost.protocol import validate_training_config
from src.models.train.ehr.catboost.training import fit_fold
from src.models.train.image.resnet50.training import (
    IMAGENET,
    build_model,
    input_transform,
)
from src.models.train.image.training import (
    MEAN,
    STD,
    FoldImages,
    configure_runtime,
    image_runtime_versions,
    seed_worker,
)

MODEL = "feature_fusion"
VECTOR_DIM = 2048
LAYER = "backbone.avgpool.flatten_before_event_head"
INPUT_TRANSFORM = {"id": "resnet50_imagenet_input_v1", "input_range": [0.0, 1.0], "repeat_channels": 3,
                   "mean": list(MEAN), "std": list(STD), "resize": False}


def validate_parameters(params):
    require(set(params) == {"ehr_source", "image_source", "pca_components", "extraction"},
            "INVALID_FUSION_CONFIG", "Specify both sources, PCA components and extraction settings")
    for modality, model in (("ehr", "catboost"), ("image", "resnet50")):
        choice = params[f"{modality}_source"]
        require(isinstance(choice, dict) and set(choice) == {"model", "assessment_id"}
                and choice["model"] == model, "INVALID_FUSION_CONFIG", "F2 requires CatBoost and ResNet50 sources")
        identifier(choice["assessment_id"])
    require(type(params["pca_components"]) is int and 0 < params["pca_components"] <= VECTOR_DIM,
            "INVALID_FUSION_CONFIG", "Use a positive PCA dimension within the representation size")
    settings = params["extraction"]
    require(isinstance(settings, dict) and set(settings) == {"device", "batch_size", "num_workers"}
            and settings["device"] in ("auto", "cpu", "cuda")
            and type(settings["batch_size"]) is int and settings["batch_size"] > 0
            and type(settings["num_workers"]) is int and settings["num_workers"] >= 0,
            "INVALID_FUSION_CONFIG", "Invalid extraction runtime settings")


def output(store, manifest, name):
    return store.layout.resolve(store.output_ref(manifest, name))


def resolve_sources(layout, package, params):
    sources, records = {}, {}
    for modality in ("ehr", "image"):
        choice = params[f"{modality}_source"]
        model, key = choice["model"], choice["assessment_id"]
        assessment_store = experiment_store(layout, "assessment", modality, model, key)
        assessment = assessment_store.validate(assessment_store.manifest_ref("assessment", "run"))
        require(assessment["scope"] == assessment_store.scope, "INVALID_FUSION_SOURCE", "Source scope differs")
        load_predictions(package.release, package.samples, package.cv,
                         [(model, key, assessment_store, assessment)], modality=modality)
        store, manifest = training_source(layout, package.release, assessment["details"]["train_id"], model, modality=modality)
        snapshot = read_json(output(store, manifest, "effective_configuration.json"))
        require(snapshot["id"] == assessment["details"]["train_id"]
                and snapshot["data_release"] == package.release.store.data.key,
                "INVALID_FUSION_SOURCE", "Source training identity differs")
        records[modality] = {"assessment": {"manifest": assessment["self"], "generation": assessment["generation"]},
                             "training": {"manifest": manifest["self"], "generation": manifest["generation"]},
                             "effective_training_configuration": snapshot}
        sources[modality] = (store, manifest, snapshot)
    validate_training_config(records["ehr"]["effective_training_configuration"])
    config = records["image"]["effective_training_configuration"]
    require(config["parameters"].get("pretraining") == IMAGENET
            and config["parameters"].get("event_components") is True and "layer4_lr" in config["parameters"]
            and config["runtime_versions"] == image_runtime_versions(config["parameters"]),
            "INVALID_FUSION_SOURCE", "Use the selected fine-tuned event-component image model and its runtime")
    image_package = load_image_package(package.release, config["parameters"]["image_source"])
    require(config["resolved_image_source"] == image_package.source
            and config["image_shape"] == list(image_package.shape) == sources["image"][1]["details"]["image_shape"]
            and image_package.samples.equals(package.samples) and image_package.cv == package.cv,
            "INVALID_FUSION_SOURCE", "Image source, dimensions or patient folds differ")
    return sources, records, image_package


def fold_interface(source, package, fold):
    store, manifest, _ = source
    interface = read_json(output(store, manifest, f"fold_{fold['fold']}/interface.json"))
    require(interface["fold"] == fold["fold"]
            and interface["train_sample_ids"] == package.samples.iloc[fold["train_indices"]].sample_id.tolist()
            and interface["validation_sample_ids"] == package.samples.iloc[fold["validation_indices"]].sample_id.tolist(),
            "INVALID_FUSION_CHECKPOINT", "Source checkpoint patient-fold identity differs")
    return interface


def extract_vectors(package, fold, source, settings, label):
    store, manifest, snapshot = source
    config = snapshot["parameters"]
    interface = fold_interface(source, package, fold)
    prefix = f"fold_{fold['fold']}"
    history = read_json(output(store, manifest, f"{prefix}/history.json"))["epochs"]
    valid = np.array([row["validation"]["rmse"] for row in history])
    require(interface["seed"] == config["seed"] and valid.size > 0 and np.isfinite(valid).all()
            and [row["epoch"] for row in history] == list(range(1, len(history) + 1))
            and interface["best_epoch"] == int(valid.argmin()) + 1,
            "INVALID_FUSION_CHECKPOINT", "Image checkpoint epoch or seed differs")
    indices = sorted(fold["train_indices"] + fold["validation_indices"])
    require(indices == sorted(package.cv["development_indices"]) and len(indices) == len(set(indices))
            and not set(indices) & set(package.cv["test_indices"]),
            "INVALID_FUSION_FOLDS", "Extract only the complete, disjoint development fold")
    device = configure_runtime(settings["device"], config["seed"])
    model = build_model(config)
    checkpoint_ref = store.output_ref(manifest, f"{prefix}/best.pt")
    model.load_state_dict(torch.load(store.layout.resolve(checkpoint_ref), map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval().requires_grad_(False)
    loader = DataLoader(FoldImages(package, indices, augment=False, transform=input_transform(config)),
                        batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"]))
    vectors, ids = [], []
    print(f"{label} fold={fold['fold']}: extracting {len(indices)} development vectors on {device}", flush=True)
    with torch.inference_mode():
        for pixels, _labels, batch_ids in loader:
            features = model.backbone(pixels.to(device)).cpu().numpy()
            require(features.shape == (len(batch_ids), VECTOR_DIM) and np.isfinite(features).all(),
                    "INVALID_IMAGE_FEATURES", "Expected finite 2048-dimensional pooled vectors")
            vectors.append(features)
            ids.extend(batch_ids)
    require(ids == package.samples.iloc[indices].sample_id.tolist(), "INVALID_IMAGE_FEATURES", "Extraction order differs")
    return np.concatenate(vectors), {"sample_ids": ids, "fold": fold["fold"], "layer": LAYER,
        "dimension": VECTOR_DIM, "checkpoint": checkpoint_ref,
        "training_manifest": {"manifest": manifest["self"], "generation": manifest["generation"]},
        "best_epoch": interface["best_epoch"], "image_source": snapshot["resolved_image_source"],
        "image_shape": snapshot["image_shape"], "pretraining": config["pretraining"],
        "transform": INPUT_TRANSFORM, "augmentation": False, "mode": "eval",
        "seed": config["seed"], "runtime_versions": image_runtime_versions(config), "device": str(device)}


def fit_projection(vectors, train_rows, components, train_ids):
    require(vectors.ndim == 2 and vectors.shape[1] == VECTOR_DIM and np.isfinite(vectors).all()
            and len(train_rows) > components, "INVALID_IMAGE_FEATURES", "Insufficient finite training vectors for PCA")
    train = vectors[train_rows].astype(np.float64)
    scaler = StandardScaler().fit(train)
    require((scaler.var_ > 0).any(), "INSUFFICIENT_IMAGE_RANK", "Training vectors are constant")
    pca = PCA(n_components=components, whiten=False, svd_solver="full").fit(scaler.transform(train))
    tolerance = max(train.shape) * np.finfo(np.float64).eps * pca.singular_values_[0]
    require(np.isfinite(pca.components_).all() and pca.singular_values_[-1] > tolerance,
            "INSUFFICIENT_IMAGE_RANK", "Training vectors cannot support the requested PCA dimension; review the plan")
    arrays = dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_,
                  pca_mean=pca.mean_, components=pca.components_, singular_values=pca.singular_values_,
                  explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_)
    projection = {"training_sample_ids": train_ids, "input_dimension": VECTOR_DIM,
        "n_components": components, "whiten": False, "svd_solver": "full", "standardize": True,
        "rank_tolerance": float(tolerance), "sklearn_version": version("scikit-learn"),
        "feature_names": [f"img_pc_{i:02d}" for i in range(1, components + 1)]}
    return projection, arrays


def augmented_package(package, fold, vectors, metadata, projection, arrays):
    indices = sorted(fold["train_indices"] + fold["validation_indices"])
    train_ids = package.samples.iloc[fold["train_indices"]].sample_id.tolist()
    count = projection["n_components"]
    names = [f"img_pc_{i:02d}" for i in range(1, count + 1)]
    require(metadata["sample_ids"] == package.samples.iloc[indices].sample_id.tolist()
            and metadata["fold"] == fold["fold"] and metadata["layer"] == LAYER and metadata["dimension"] == VECTOR_DIM
            and metadata["augmentation"] is False and metadata["mode"] == "eval"
            and projection["training_sample_ids"] == train_ids and projection["feature_names"] == names
            and projection["input_dimension"] == VECTOR_DIM and projection["whiten"] is False
            and projection["standardize"] is True and projection["svd_solver"] == "full"
            and projection["sklearn_version"] == version("scikit-learn")
            and vectors.shape == (len(indices), VECTOR_DIM) and np.isfinite(vectors).all(),
            "INVALID_FUSION_FEATURES", "Cached vectors, projection or training identity differs")
    shapes = {"scaler_mean": (VECTOR_DIM,), "scaler_scale": (VECTOR_DIM,), "scaler_var": (VECTOR_DIM,),
              "pca_mean": (VECTOR_DIM,), "components": (count, VECTOR_DIM), "singular_values": (count,),
              "explained_variance": (count,), "explained_variance_ratio": (count,)}
    require(set(arrays) == set(shapes) and all(arrays[k].shape == shape and np.isfinite(arrays[k]).all()
            for k, shape in shapes.items()) and (arrays["scaler_scale"] > 0).all()
            and (arrays["singular_values"] > projection["rank_tolerance"]).all(),
            "INVALID_FUSION_PROJECTION", "Projection arrays are non-finite or inconsistent")
    # Verify the scaler against training rows only; validation vectors never determine its statistics.
    positions = {index: pos for pos, index in enumerate(indices)}
    fitted = StandardScaler().fit(vectors[[positions[i] for i in fold["train_indices"]]].astype(np.float64))
    require(np.array_equal(fitted.mean_, arrays["scaler_mean"]) and np.array_equal(fitted.scale_, arrays["scaler_scale"])
            and np.array_equal(fitted.var_, arrays["scaler_var"]),
            "INVALID_FUSION_PROJECTION", "Scaler statistics differ from the declared training subset")
    projected = ((vectors.astype(np.float64) - arrays["scaler_mean"]) / arrays["scaler_scale"]
                 - arrays["pca_mean"]) @ arrays["components"].T
    require(np.isfinite(projected).all() and not set(names) & set(package.features.columns),
            "INVALID_FUSION_FEATURES", "Projection must be finite and new feature names must be unique")
    features = package.features.copy()
    for i, name in enumerate(names):
        features[name] = np.nan
        features.loc[indices, name] = projected[:, i]
    return replace(package, features=features)


def load_fold_package(store, manifest, package, fold):
    prefix = f"fold_{fold['fold']}"
    projection = read_json(output(store, manifest, f"{prefix}/projection.json"))
    metadata = read_json(output(store, manifest, f"{prefix}/image_features.json"))
    sources = read_json(output(store, manifest, "sources.json"))
    config = read_json(output(store, manifest, "effective_configuration.json"))
    image = sources["image"]
    train = read_json(store.layout.resolve(image["training"]["manifest"]))
    snapshot = image["effective_training_configuration"]
    interface = fold_interface((store, train, snapshot), package, fold)
    require(metadata["training_manifest"] == image["training"]
            and metadata["checkpoint"] == store.output_ref(train, f"{prefix}/best.pt")
            and metadata["image_source"] == snapshot["resolved_image_source"]
            and metadata["image_shape"] == snapshot["image_shape"]
            and metadata["seed"] == snapshot["parameters"]["seed"]
            and metadata["best_epoch"] == interface["best_epoch"]
            and metadata["runtime_versions"] == snapshot["runtime_versions"]
            and metadata["pretraining"] == snapshot["parameters"]["pretraining"]
            and metadata["transform"] == INPUT_TRANSFORM
            and projection["n_components"] == config["parameters"]["pca_components"],
            "INVALID_FUSION_FEATURES", "Cached features must bind the selected fold checkpoint and PCA configuration")
    with np.load(output(store, manifest, f"{prefix}/image_features.npz"), allow_pickle=False) as saved:
        require(set(saved.files) == {"vectors"}, "INVALID_FUSION_FEATURES", "Unexpected feature-cache arrays")
        vectors = saved["vectors"]
    with np.load(output(store, manifest, f"{prefix}/projection.npz"), allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    return augmented_package(package, fold, vectors, metadata, projection, arrays)


def publish_training(layout, config_path, experiment, package, input_summary):
    params = experiment["parameters"]
    sources, records, images = resolve_sources(layout, package, params)
    store = experiment_store(layout, "train", "fusion", MODEL, experiment["id"])
    acceptance = package.release.acceptance
    inputs = [{"manifest": acceptance["self"], "generation": acceptance["generation"]},
              *[records[m]["assessment"] for m in ("ehr", "image")]]
    ref = store.manifest_ref("train", "run")
    existing = completed_result(store, ref, experiment, inputs)
    if existing:
        return existing
    effective_ehr = records["ehr"]["effective_training_configuration"]
    with store.publish("train", "run", inputs=[item["manifest"] for item in inputs],
                       configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "folds": 5}) as work:
        require(all(store.validate(item["manifest"])["generation"] == item["generation"] for item in inputs),
                "UPSTREAM_REBUILT", "Source changed after it was loaded")
        results = []
        for fold in package.cv["folds"]:
            interface = fold_interface(sources["ehr"], package, fold)
            require(interface["features"] == list(package.features.columns)
                    and interface["categorical_features"] == package.schema["categorical_features"]
                    and interface["catboost_version"] == version("catboost"),
                    "INVALID_FUSION_CHECKPOINT", "The EHR input protocol differs from the source checkpoint")
            vectors, metadata = extract_vectors(images, fold, sources["image"], params["extraction"],
                                                f"{MODEL}/{experiment['id']}")
            indices = sorted(fold["train_indices"] + fold["validation_indices"])
            positions = {index: pos for pos, index in enumerate(indices)}
            projection, arrays = fit_projection(vectors, [positions[i] for i in fold["train_indices"]],
                params["pca_components"], package.samples.iloc[fold["train_indices"]].sample_id.tolist())
            augmented = augmented_package(package, fold, vectors, metadata, projection, arrays)
            print(f"{MODEL}/{experiment['id']} fold={fold['fold']}: fitting matched CatBoost with {params['pca_components']} image PCs", flush=True)
            folder = work / f"fold_{fold['fold']}"
            results.append(fit_fold(augmented, effective_ehr, fold, folder))
            np.savez(folder / "projection.npz", **arrays)
            write_json(folder / "projection.json", projection)
            np.savez(folder / "image_features.npz", vectors=vectors)
            write_json(folder / "image_features.json", metadata)
        write_json(work / "effective_configuration.json", experiment)
        write_json(work / "effective_ehr_configuration.json", effective_ehr)
        write_json(work / "sources.json", records)
        write_json(work / "input_summary.json", {**input_summary, "image_vector_dimension": VECTOR_DIM,
                                                 "added_features": params["pca_components"]})
        write_json(work / "training_summary.json", {"id": experiment["id"], "folds": results,
            "test_consumed_for_fitting": False, "image_test_consumed_for_extraction": False,
            "validation": "exploratory_existing_image_checkpoint_early_stopping_protocol",
            "prior_fixed_test_exposure": "Both source test results were viewed and used for source selection."})
    return {"id": experiment["id"], "status": "complete", "manifest": ref}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment["parameters"])
    require(version("catboost") == "1.2.10", "INVALID_CATBOOST_VERSION", "Use the source CatBoost 1.2.10 runtime")
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    summary = checked_input_summary(package)
    return run_model_batch(layout, "train", "fusion", MODEL, experiments,
                           lambda e: publish_training(layout, config_path, e, package, summary))
