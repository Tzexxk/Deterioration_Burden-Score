"""Bind the shared development assessment to the EHR modality."""

from src.models.assessment.evaluation import (
    metric_pair as metric_pair,
)
from src.models.assessment.evaluation import (
    publish_assessment,
    validate_parameters,
)
from src.models.assessment.evaluation import (
    training_source as resolve_training_source,
)
from src.models.ehr_data import checked_input_summary, load_package
from src.models.experiments import read_batch, run_model_batch
from src.models.released_data import ReleasedData


def training_source(layout, package, train_id, model):
    return resolve_training_source(layout, package.release, train_id, model, modality="ehr")


def assess_model(layout, config_path, selected, model, predict, plot_history=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment)
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    checked_input_summary(package)

    def assess(experiment):
        source = training_source(layout, package, experiment["parameters"]["train_id"], model)
        return publish_assessment(layout, config_path, experiment, package, model, predict, plot_history,
                                  modality="ehr", source=source)

    return run_model_batch(layout, "assessment", "ehr", model, experiments, assess)
