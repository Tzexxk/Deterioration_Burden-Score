"""Train a frozen ImageNet-1K CAFormer-S18 on the development folds."""

from functools import partial

from src.models.train.image.timm_frozen import build_timm_model, train_timm_model

MODEL = "caformer_s18"
build_model = partial(build_timm_model, "caformer_s18.sail_in1k")


def run_training(layout, config_path, selected=None):
    return train_timm_model(layout, config_path, selected, MODEL, build_model)
