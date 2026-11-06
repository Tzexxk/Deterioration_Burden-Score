"""Fit a linear regression head on the frozen Microsoft RAD-DINO encoder."""

from importlib.metadata import PackageNotFoundError, version

from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from src.common.paths import PipelineError
from src.models.train.image.training import (
    configure_runtime,
    fit_frozen_head,
    image_runtime_versions,
    train_model,
    validate_frozen_settings,
    validate_pretrained_state,
)

MODEL = "rad_dino"
PATCH_SIZE = 14
REFERENCE_SIZE = 518
MEAN = (0.5307,) * 3
STD = (0.2583,) * 3


def rad_dino_input(gray):
    height, width = gray.shape[-2:]
    bottom, right = -height % PATCH_SIZE, -width % PATCH_SIZE
    gray = F.pad(gray, (right // 2, right - right // 2, bottom // 2, bottom - bottom // 2))
    return TF.normalize(gray.repeat(3, 1, 1), MEAN, STD)


def backbone_config():
    from transformers import Dinov2Config

    return Dinov2Config(image_size=REFERENCE_SIZE, patch_size=PATCH_SIZE, attn_implementation="sdpa")


class RadDinoRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Dinov2Model

        config = backbone_config()
        self.backbone = Dinov2Model(config)
        self.head = nn.Linear(config.hidden_size, 1)

    def encode(self, images):
        """Return final-layer CLS features and a channel-first spatial patch grid."""
        tokens = self.backbone(pixel_values=images).last_hidden_state
        batch, _, height, width = images.shape
        patches = tokens[:, 1:].transpose(1, 2).reshape(batch, tokens.shape[-1],
                                                         height // PATCH_SIZE, width // PATCH_SIZE)
        return tokens[:, 0], patches

    def forward(self, images):
        global_features, _ = self.encode(images)
        return self.head(global_features)


def build_model(_config):
    return RadDinoRegressor().float()


def input_transform(_config):
    return rad_dino_input


def runtime_versions(config):
    versions = image_runtime_versions(config)
    try:
        versions.update({name: version(name) for name in ("transformers", "safetensors")})
    except PackageNotFoundError as error:
        raise PipelineError("RAD_DINO_DEPENDENCY_MISSING", "Install transformers and safetensors") from error
    return versions


def load_weights(path, _config):
    from safetensors.torch import load_file

    return validate_pretrained_state(load_file(str(path), device="cpu"))


def fit_fold(package, fold, config, weights, folder, *, label):
    device = configure_runtime(config["device"], config["seed"])
    model = build_model(config)
    model.backbone.load_state_dict(weights, strict=True)
    return fit_frozen_head(package, fold, config, model, model.head, folder, label=label, device=device,
                           transform=rad_dino_input)


def run_training(layout, config_path, selected=None):
    return train_model(layout, config_path, selected, MODEL, validate_frozen_settings, fit_fold, load_weights,
                       runtime_versions)
