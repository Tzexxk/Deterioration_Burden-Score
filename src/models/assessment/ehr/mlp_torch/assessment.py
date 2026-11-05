"""Assess EHR PyTorch MLP checkpoints and plot fold training histories."""

from functools import partial

from src.models.assessment.ehr.evaluation import assess_model
from src.models.assessment.plots import plot_history
from src.models.train.ehr.mlp_torch.training import checkpoint_prediction


def run_assessment(layout, config_path, selected=None):
    plot = partial(plot_history, round_label="Epoch")
    return assess_model(layout, config_path, selected, "mlp_torch", checkpoint_prediction, plot)
