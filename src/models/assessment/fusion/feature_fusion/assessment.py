"""Assess published feature-fusion models without extracting images again."""

from src.common.paths import read_json, require
from src.models.assessment.ehr.catboost.assessment import checkpoint_prediction
from src.models.assessment.evaluation import publish_assessment, validate_parameters
from src.models.assessment.plots import plot_history
from src.models.ehr_data import checked_input_summary, load_package
from src.models.experiments import experiment_store, read_batch, run_model_batch
from src.models.released_data import ReleasedData
from src.models.train.fusion.feature_fusion.training import MODEL, load_fold_package, output, resolve_sources


def predict_fold(store, manifest, package, fold):
    augmented = load_fold_package(store, manifest, package, fold)
    print(f"{MODEL}/{manifest['scope']['experiment']} fold={fold['fold']}: reloading validation checkpoint", flush=True)
    return checkpoint_prediction(store, manifest, augmented, fold)


def run_assessment(layout, config_path, selected=None):
    experiments = read_batch(layout, config_path, selected)
    for experiment in experiments:
        validate_parameters(experiment)
    package = load_package(ReleasedData.load(layout, experiments[0]["data_release"]))
    checked_input_summary(package)

    def assess(experiment):
        store = experiment_store(layout, "train", "fusion", MODEL, experiment["parameters"]["train_id"])
        manifest = store.validate(store.manifest_ref("train", "run"))
        config = read_json(output(store, manifest, "effective_configuration.json"))
        require(manifest["scope"] == store.scope and config["id"] == store.scope["experiment"]
                and config["data_release"] == experiment["data_release"],
                "ASSESSMENT_DATA_MISMATCH", "F2 training and assessment identities differ")
        _, records, _ = resolve_sources(layout, package, config["parameters"])
        acceptance = package.release.acceptance
        inputs = [{"manifest": acceptance["self"], "generation": acceptance["generation"]},
                  *[records[m]["assessment"] for m in ("ehr", "image")]]
        require(manifest["inputs"] == inputs and read_json(output(store, manifest, "sources.json")) == records
                and read_json(output(store, manifest, "effective_ehr_configuration.json"))
                == records["ehr"]["effective_training_configuration"],
                "ASSESSMENT_DATA_MISMATCH", "F2 must bind the exact frozen data and matched source protocols")
        return publish_assessment(layout, config_path, experiment, package, MODEL, predict_fold, plot_history,
                                  modality="fusion", source=(store, manifest))

    return run_model_batch(layout, "assessment", "fusion", MODEL, experiments, assess)
