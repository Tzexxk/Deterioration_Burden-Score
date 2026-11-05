"""Train the historical CPU PyTorch MLP and reload its best fold weights."""

from importlib.metadata import version

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch import nn

from src.common.manifest import write_json
from src.common.paths import read_json, require
from src.models.ehr_fold import fold_interface, read_fold_interface
from src.models.ehr_preprocessing import fit_preprocessing, transform_features
from src.models.experiments import read_batch
from src.models.train.ehr.training import train_model

TORCH_VERSION = "2.14.0+cu132"
FIXED_SETTINGS = {"device": "cpu", "threads": 1, "dtype": "float32", "activation": "relu",
                  "output": "linear", "batch": "full", "gradient_clipping": False, "scheduler": None}


class MLP(nn.Module):
    def __init__(self, width, hidden, mean=0.0):
        super().__init__()
        self.hidden = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, 1)
        nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity="relu")
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, mean)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x))).squeeze(-1)


def tensor(values):
    return torch.from_numpy(np.asarray(values, dtype=np.float32).copy())


def finite_number(value, minimum=0):
    return type(value) in (float, int) and np.isfinite(value) and value >= minimum


def validate_experiment(experiment):
    params = experiment["parameters"]
    require(set(params) == {"seed", "candidate", "torch_training"}, "INVALID_TORCH_MLP_CONFIG",
            "Specify seed, candidate and torch_training")
    candidate, settings = params["candidate"], params["torch_training"]
    require(type(params["seed"]) is int and 0 <= params["seed"] < 2**32
            and isinstance(candidate, dict) and set(candidate) == {"name", "hidden", "lr"}
            and type(candidate["hidden"]) is int and candidate["hidden"] > 0
            and finite_number(candidate["lr"]) and candidate["lr"] > 0,
            "INVALID_TORCH_MLP_CONFIG", "Invalid seed, hidden width or learning rate")
    require(isinstance(settings, dict)
            and set(settings) == FIXED_SETTINGS.keys() | {"max_epochs", "patience", "minimum_epochs", "min_delta",
                                                         "weight_decay", "betas", "eps"}
            and all(settings[key] == value for key, value in FIXED_SETTINGS.items())
            and all(type(settings[key]) is int and settings[key] > 0
                    for key in ("max_epochs", "patience", "minimum_epochs"))
            and settings["minimum_epochs"] <= settings["max_epochs"]
            and all(finite_number(settings[key]) for key in ("min_delta", "weight_decay", "eps"))
            and settings["eps"] > 0 and isinstance(settings["betas"], list) and len(settings["betas"]) == 2
            and all(finite_number(value) and value < 1 for value in settings["betas"]),
            "INVALID_TORCH_MLP_CONFIG", "Invalid CPU MLP optimizer or stopping settings")


def load_model(path, width, hidden):
    model = MLP(width, hidden).float()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    require(all(bool(torch.isfinite(value).all()) for value in model.parameters()),
            "INVALID_CHECKPOINT", "Nonfinite PyTorch MLP weights")
    return model.eval()


@torch.no_grad()
def predict(model, values):
    scores = model(values).numpy().astype(float)
    require(scores.shape == (len(values),) and np.isfinite(scores).all(),
            "INVALID_CHECKPOINT", "PyTorch MLP predictions must be finite and match the fold")
    return scores


def fit_fold(package, experiment, fold, folder):
    train, validation = fold["train_indices"], fold["validation_indices"]
    x_train, x_validation = (package.features.iloc[indices] for indices in (train, validation))
    y, vy = (tensor(package.samples.iloc[indices].burden_score.to_numpy(float)) for indices in (train, validation))
    params = experiment["parameters"]
    candidate, settings = params["candidate"], params["torch_training"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(params["seed"])
    with threadpool_limits(limits=1):
        state = fit_preprocessing(x_train, package.schema)
        x, vx = (tensor(transform_features(features, state)) for features in (x_train, x_validation))
    model = MLP(x.shape[1], candidate["hidden"], float(y.mean())).float()
    weights = [p for name, p in model.named_parameters() if not name.endswith("bias")]
    biases = [p for name, p in model.named_parameters() if name.endswith("bias")]
    optimizer = torch.optim.AdamW([{"params": weights, "weight_decay": settings["weight_decay"]},
                                   {"params": biases, "weight_decay": 0.0}],
                                  lr=candidate["lr"], betas=tuple(settings["betas"]), eps=settings["eps"],
                                  foreach=False, fused=False)
    best_score = anchor = float("inf")
    best_epoch = stale = 0
    learn, valid = [], []
    reason = "budget_limited"
    for epoch in range(1, settings["max_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(x) - y) ** 2).mean()
        require(bool(torch.isfinite(loss)), "NUMERICAL_FAILURE", "Nonfinite PyTorch training loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "NUMERICAL_FAILURE", "Missing or nonfinite PyTorch gradient")
        optimizer.step()
        with torch.no_grad():
            train_score = float(torch.sqrt(((model(x) - y) ** 2).mean()))
            score = float(torch.sqrt(((model(vx) - vy) ** 2).mean()))
        require(np.isfinite(train_score) and np.isfinite(score),
                "NUMERICAL_FAILURE", "Nonfinite PyTorch RMSE")
        learn.append(train_score)
        valid.append(score)
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if score < anchor - settings["min_delta"]:
            anchor, stale = score, 0
        else:
            stale += 1
        if epoch >= settings["minimum_epochs"] and stale >= settings["patience"]:
            reason = "validation_early_stop"
            break
    folder.mkdir()
    path = folder / "best.pt"
    torch.save(best_weights, path)
    model.load_state_dict(best_weights)
    expected = predict(model.eval(), vx)
    reloaded = load_model(path, x.shape[1], candidate["hidden"])
    np.testing.assert_array_equal(predict(reloaded, vx), expected)
    write_json(folder / "preprocessing.json", state)
    write_json(folder / "history.json", {"learn": learn, "validation": valid})
    write_json(folder / "interface.json", {**fold_interface(package, fold, params), "torch_version": version("torch"),
               "input_width": x.shape[1], "best_epoch": best_epoch, "stop_reason": reason})
    return {"fold": fold["fold"], "fit_method": reason, "best_epoch": best_epoch, "epochs": len(valid)}


def checkpoint_prediction(store, manifest, package, fold):
    interface = read_fold_interface(store, manifest, package, fold)
    require(interface["torch_version"] == version("torch"), "INVALID_CHECKPOINT", "PyTorch version differs")

    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    torch.set_num_threads(1)
    features = package.features.iloc[fold["validation_indices"]]
    values = tensor(transform_features(features, read_json(output("preprocessing.json"))))
    require(interface["input_width"] == values.shape[1], "INVALID_CHECKPOINT", "PyTorch MLP input width differs")
    model = load_model(output("best.pt"), values.shape[1], interface["parameters"]["candidate"]["hidden"])
    history = read_json(output("history.json"))
    learn, valid = (np.asarray(history[part], dtype=float) for part in ("learn", "validation"))
    best = interface["best_epoch"]
    require(learn.shape == valid.shape and learn.ndim == 1 and learn.size > 0
            and np.isfinite(learn).all() and np.isfinite(valid).all() and best == int(np.argmin(valid)) + 1,
            "INVALID_TRAINING_HISTORY", "RMSE history differs from the PyTorch checkpoint")
    return predict(model, values), {"fold": fold["fold"], "best_round": best, "learn": learn, "validation": valid}


def run_training(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_experiment(experiment)
    require(version("torch") == TORCH_VERSION and version("scikit-learn") == "1.9.1",
            "INVALID_MODEL_RUNTIME", "Use the pinned PyTorch and scikit-learn versions")
    return train_model(layout, config_path, experiments, "mlp_torch", fit_fold)
