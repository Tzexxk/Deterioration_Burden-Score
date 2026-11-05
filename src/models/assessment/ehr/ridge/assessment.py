"""Assess saved Ridge coefficients on each development validation fold."""

from importlib.metadata import version

from src.common.paths import read_json, require
from src.models.assessment.ehr.evaluation import assess_model
from src.models.train.ehr.ridge.training import predict_checkpoint


def checkpoint_prediction(store, manifest, package, fold):
    def output(name):
        ref = store.output_ref(manifest, f"fold_{fold['fold']}/{name}")
        return store.layout.resolve(ref)

    interface = read_json(output("interface.json"))
    train, validation = fold["train_indices"], fold["validation_indices"]
    require(interface["fold"] == fold["fold"] and interface["scikit_learn_version"] == version("scikit-learn")
            and interface["features"] == list(package.features.columns)
            and interface["train_sample_ids"] == package.samples.iloc[train].sample_id.tolist()
            and interface["validation_sample_ids"] == package.samples.iloc[validation].sample_id.tolist(),
            "INVALID_CHECKPOINT", "Ridge feature, version or fold identity differs")
    checkpoint = read_json(output("model.json"))
    return predict_checkpoint(checkpoint, package.features.iloc[validation]), None


def run_assessment(layout, config_path, selected=None):
    return assess_model(layout, config_path, selected, "ridge", checkpoint_prediction)
