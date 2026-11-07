"""Bind fixed-test comparison to the EHR modality."""

from functools import partial

from src.models.ehr_data import checked_input_summary, load_package
from src.models.test.comparison import run_comparison as compare


def load_cohort(release):
    package = load_package(release)
    checked_input_summary(package)
    return package


run_comparison = partial(compare, modality="ehr", load_cohort=load_cohort)
