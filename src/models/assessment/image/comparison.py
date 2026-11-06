"""Bind the shared development comparison to chest X-ray assessments."""

from functools import partial

from src.models.assessment.comparison import run_comparison as compare
from src.models.image_data import load_image_cohort


def load_development(release):
    _, samples, cv = load_image_cohort(release)
    return samples, cv


run_comparison = partial(compare, modality="image", load_development=load_development)
