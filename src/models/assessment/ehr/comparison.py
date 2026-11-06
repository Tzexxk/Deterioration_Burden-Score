"""Bind the shared development comparison to EHR."""

from functools import partial

from src.models.assessment.comparison import run_comparison as compare
from src.models.ehr_data import load_package


def load_development(release):
    package = load_package(release)
    return package.samples, package.cv


run_comparison = partial(compare, modality="ehr", load_development=load_development)
