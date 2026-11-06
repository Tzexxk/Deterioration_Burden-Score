"""Train a frozen ImageNet-1K DeiT-B/16 on the development folds."""

from functools import partial

from src.models.train.image.timm_frozen import build_timm_model, train_timm_model

MODEL = "deit_base_patch16_224"
build_model = partial(build_timm_model, "deit_base_patch16_224.fb_in1k", dynamic_img_size=True, dynamic_img_pad=True)


def run_training(layout, config_path, selected=None):
    return train_timm_model(layout, config_path, selected, MODEL, build_model)
