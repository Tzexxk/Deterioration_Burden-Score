"""Shared image fold fitting and publication."""

import math
import os
import random
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.common.manifest import write_json
from src.common.paths import require
from src.models.experiments import (
    completed_result,
    experiment_store,
    read_batch,
    run_model_batch,
)
from src.models.image_data import load_image_package
from src.models.released_data import ReleasedData

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def imagenet_input(gray):
    return TF.normalize(gray.repeat(3, 1, 1), MEAN, STD)


class FoldImages(Dataset):
    def __init__(self, package, indices, *, augment, transform=imagenet_input):
        self.package = package
        self.indices = indices
        self.augment = augment
        self.transform = transform
        rows = package.samples.iloc[indices]
        self.ids = rows.sample_id.tolist()
        self.targets = rows.burden_score.to_numpy(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        gray = torch.from_numpy(self.package.read_image(self.indices[index])).unsqueeze(0)
        if self.augment:
            angle = torch.empty(()).uniform_(-7, 7).item()
            translate = [torch.empty(()).uniform_(-.03 * size, .03 * size).item()
                         for size in (gray.shape[2], gray.shape[1])]
            gray = TF.affine(gray, angle, translate, scale=1.0, shear=[0.0, 0.0],
                             interpolation=InterpolationMode.BILINEAR, fill=0.0)
            gray = TF.adjust_contrast(gray, torch.empty(()).uniform_(0.9, 1.1).item())
        return self.transform(gray), torch.tensor([self.targets[index]]), self.ids[index]


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def configure_runtime(requested, seed):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if requested == "cpu":
        torch.random.default_generator.manual_seed(seed)
        return torch.device("cpu")

    require(torch.cuda.is_available(), "CUDA_NOT_AVAILABLE", "Requested CUDA device is unavailable")
    workspace = ":4096:8"
    require(not torch.cuda.is_initialized() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace,
            "INVALID_CUDA_RUNTIME", "CUDA must be initialized with the deterministic workspace")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda:0")


def run_epoch(model, loader, device, optimizer=None):
    squared = absolute = 0.0
    ids, predictions = [], []
    training = optimizer is not None
    if not training:
        model.eval()
    with torch.set_grad_enabled(training):
        for image, labels, batch_ids in loader:
            image, labels = image.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(image)
            require(scores.shape == labels.shape and bool(torch.isfinite(scores).all()),
                    "INVALID_IMAGE_PREDICTIONS", "Expected finite scalar scores for every image")
            error = scores - labels
            if training:
                error.square().mean().backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.parameters() if p.requires_grad),
                        "INVALID_IMAGE_GRADIENT", "Missing or nonfinite image model gradients")
                optimizer.step()
            error = error.detach().double()
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            ids.extend(batch_ids)
            predictions.extend(scores.detach().cpu().flatten().tolist())
    require(len(ids) == len(loader.dataset) and set(ids) == set(loader.dataset.ids),
            "INVALID_IMAGE_COVERAGE", "Epoch must cover each selected sample exactly once")
    return {"mse": squared / len(ids), "rmse": math.sqrt(squared / len(ids)),
            "mae": absolute / len(ids), "count": len(ids)}, ids, predictions


def fit_epochs(package, fold, config, model, optimizer, folder, *, label, device,
               transform=imagenet_input, augment=False, schedule=None, head_only_epochs=None):
    seed = config["seed"]
    train, validation = (
        DataLoader(FoldImages(package, fold[f"{role}_indices"], augment=augment and role == "train",
                              transform=transform),
                   batch_size=config["batch_size"], shuffle=role == "train",
                   num_workers=config["num_workers"], worker_init_fn=seed_worker,
                   generator=torch.Generator().manual_seed(seed))
        for role in ("train", "validation"))
    folder.mkdir()
    checkpoint = folder / "best.pt"
    history = []
    best_epoch, best_rmse, stale = 0, math.inf, 0
    for epoch in range(1, config["max_epochs"] + 1):
        stage2 = head_only_epochs is not None and epoch > head_only_epochs
        if stage2 and epoch == head_only_epochs + 1:
            stale = 0
        if schedule is not None:
            schedule(model, epoch, head_only_epochs)
        train_metrics, _, _ = run_epoch(model, train, device, optimizer)
        valid_metrics, _, _ = run_epoch(model, validation, device)
        if valid_metrics["rmse"] < best_rmse:
            best_epoch, best_rmse, stale = epoch, valid_metrics["rmse"], 0
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
        elif stage2:
            stale += 1
        history.append({"epoch": epoch, "stage": 2 if stage2 else 1, "train": train_metrics, "validation": valid_metrics})
        print(f"{label} epoch={epoch}/{config['max_epochs']} train RMSE={train_metrics['rmse']:.6f} "
              f"validation RMSE={valid_metrics['rmse']:.6f} best_epoch={best_epoch}", flush=True)
        if stage2 and stale >= config["patience"]:
            break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    metrics, ids, scores = run_epoch(model, validation, device)
    require(metrics["rmse"] == best_rmse and ids == validation.dataset.ids,
            "INVALID_IMAGE_CHECKPOINT", "Reloaded best checkpoint must reproduce validation results")
    pd.DataFrame({"sample_id": ids, "predicted_score": scores}).to_parquet(folder / "validation_predictions.parquet", index=False)
    write_json(folder / "history.json", {"epochs": history})
    write_json(folder / "interface.json", {"fold": fold["fold"], "seed": seed, "best_epoch": best_epoch,
               "device": str(device), "train_sample_ids": train.dataset.ids, "validation_sample_ids": ids})
    return {"fold": fold["fold"], "best_epoch": best_epoch, "best_rmse": best_rmse, "epochs": len(history)}


def validate_frozen_settings(config):
    integers = {"seed", "num_workers", "batch_size", "max_epochs"}
    rates = {"head_lr", "weight_decay"}
    require(set(config) == integers | rates | {"image_source", "weights", "device"}
            and config["device"] in ("auto", "cpu", "cuda")
            and all(type(config[key]) is int and config[key] >= 0 for key in integers)
            and config["batch_size"] > 0 and config["max_epochs"] > 0 and config["seed"] < 2**32
            and all(type(config[key]) in (int, float) and math.isfinite(config[key]) and config[key] >= 0
                    for key in rates)
            and config["head_lr"] > 0,
            "INVALID_IMAGE_CONFIG", "Invalid frozen image training settings")


def validate_pretrained_state(state):
    require(isinstance(state, dict) and bool(state) and all(
        isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ), "INVALID_PRETRAINED_WEIGHTS", "Expected finite pretrained model tensors")
    return state


def fit_frozen_head(package, fold, config, model, head, folder, *, label, device):
    model.to(device).eval().requires_grad_(False)
    head.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_lr"],
                                  weight_decay=config["weight_decay"], foreach=False, fused=False)
    return fit_epochs(package, fold, config, model, optimizer, folder, label=label, device=device)


def image_runtime_versions(_config):
    return {name: version(name) for name in ("torch", "torchvision")}


def publish_training(layout, config_path, experiment, release, *, model, fit_fold, load_weights, runtime_versions):
    key, config = experiment["id"], experiment["parameters"]
    print(f"{key}: checking released image source and patient folds", flush=True)
    package = load_image_package(release, config["image_source"])
    source = release.acceptance
    store = experiment_store(layout, "train", "image", model, key)
    ref = store.manifest_ref("train", "run")
    inputs = [{"manifest": source["self"], "generation": source["generation"]}]
    snapshot = {**experiment, "resolved_image_source": package.source, "image_shape": list(package.shape),
                "runtime_versions": runtime_versions(config)}
    existing = completed_result(store, ref, snapshot, inputs)
    if existing:
        return existing
    path = layout.resolve(config["weights"])
    require(path.is_file(), "PRETRAINED_WEIGHTS_MISSING", "Set the local pretrained weights root in paths.local.json")
    weights = load_weights(path, config)
    with store.publish("train", "run", inputs=[source["self"]], configs=[layout.ref("project", config_path)],
                       details={"data_release": experiment["data_release"], "folds": len(package.cv["folds"]),
                                "image_shape": list(package.shape)}) as work:
        results = []
        for fold in package.cv["folds"]:
            label = f"{key} fold={fold['fold']}"
            print(f"{label}: fitting development split {package.shape}", flush=True)
            results.append(fit_fold(package, fold, config, weights, work / f"fold_{fold['fold']}", label=label))
        write_json(work / "effective_configuration.json", snapshot)
        write_json(work / "input_summary.json", {"image_shape": list(package.shape), "image_dtype": "float32",
                   "studies": len(package.samples), "development_studies": len(package.cv["development_indices"]),
                   "fixed_test_studies": len(package.cv["test_indices"]), "folds": len(package.cv["folds"])})
        write_json(work / "training_summary.json", {"id": key, "folds": results, "test_consumed_for_fitting": False})
    return {"id": key, "status": "complete", "manifest": ref}


def train_model(layout, config_path, selected, model, validate_settings, fit_fold, load_weights,
                runtime_versions=image_runtime_versions):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_settings(experiment["parameters"])
    release = ReleasedData.load(layout, experiments[0]["data_release"])
    return run_model_batch(layout, "train", "image", model, experiments,
                           lambda experiment: publish_training(
                               layout, config_path, experiment, release, model=model, fit_fold=fit_fold,
                               load_weights=load_weights, runtime_versions=runtime_versions))
