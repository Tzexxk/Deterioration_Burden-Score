"""Record and validate the feature and sample ordering of EHR checkpoints."""

from importlib.metadata import version

from src.common.paths import read_json, require


def fold_interface(package, fold, parameters):
    train, validation = fold["train_indices"], fold["validation_indices"]
    return {"fold": fold["fold"], "parameters": parameters, "features": list(package.features.columns),
            "scikit_learn_version": version("scikit-learn"),
            "train_sample_ids": package.samples.iloc[train].sample_id.tolist(),
            "validation_sample_ids": package.samples.iloc[validation].sample_id.tolist()}


def read_fold_interface(store, manifest, package, fold):
    ref = store.output_ref(manifest, f"fold_{fold['fold']}/interface.json")
    interface = read_json(store.layout.resolve(ref))
    expected = fold_interface(package, fold, interface["parameters"])
    require(all(interface[key] == value for key, value in expected.items()),
            "INVALID_CHECKPOINT", "Checkpoint feature, version or fold identity differs")
    return interface
