"""Bind fixed-test comparison to image evaluations."""

from functools import partial
from types import SimpleNamespace

from src.models.image_data import load_image_cohort
from src.models.test.comparison import run_comparison as compare


def load_cohort(release):
    _, samples, cv = load_image_cohort(release)
    return SimpleNamespace(samples=samples, cv=cv)


run_comparison = partial(compare, modality="image", load_cohort=load_cohort)
