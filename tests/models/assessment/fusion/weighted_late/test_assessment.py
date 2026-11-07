"""Synthetic dual-source publications; never read the real project outputs."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.assessment.evaluation import metric_pair
from src.models.assessment.fusion.weighted_late.assessment import paired_differences, run_assessment
from src.models.assessment.operating_points import operating_point
from src.models.ehr_metrics import ENDPOINTS
from src.models.experiments import experiment_store
from src.models.image_data import load_image_cohort
from src.models.released_data import ReleasedData
from tests.models.train.ehr.catboost import test_training

CONFIG = "configs/models/assessment/fusion/weighted_late.json"


class WeightedLateTests(unittest.TestCase):
    def setUp(self):
        fixture = test_training.CatBoostTrainingTests()
        temp_root = Path(__file__).resolve().parents[5] / "outputs/fusion_tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with patch("tempfile.tempdir", str(temp_root)):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish(image_shape=(4, 4))
        fixture.store.freeze(fixture.store.manifest_ref("step13", "validate"))
        self.layout = fixture.layout
        self.release = ReleasedData.load(self.layout, "v1_r3")
        _, self.samples, self.cv = load_image_cohort(self.release)
        template = test_training.TEMPLATE.parents[2] / "assessment/fusion/weighted_late.json"
        self.config = read_json(template)
        self.config["data_release"] = "v1_r3"
        self.config["defaults"]["bootstrap_draws"] = 40
        self.config["defaults"]["image_source"]["assessment_id"] = "selected_image"
        write_json(self.layout.path("project", CONFIG), self.config)
        self.sources = {}
        for modality, model, key in (("ehr", "catboost", "d2_l210"), ("image", "resnet50", "selected_image")):
            train = experiment_store(self.layout, "train", modality, model, key)
            snapshot = {"id": key, "data_release": "v1_r3", "parameters": {}}
            details = {"data_release": "v1_r3"}
            if modality == "image":
                snapshot.update(parameters={"image_source": {"kind": "package"}},
                                resolved_image_source={"kind": "package"}, image_shape=[4, 4])
                details["image_shape"] = [4, 4]
            with train.publish("train", "run", inputs=[self.release.acceptance["self"]], details=details) as work:
                write_json(work / "effective_configuration.json", snapshot)
            frames = []
            for fold in self.cv["folds"]:
                rows = self.samples.iloc[fold["validation_indices"]][
                    ["sample_id", "subject_id", "hadm_id", "study_id", "burden_score", *ENDPOINTS]].copy()
                mean = float(self.samples.iloc[fold["train_indices"]].burden_score.to_numpy(float).mean())
                # Include out-of-range raw predictions to exercise secondary clipping.
                predicted = np.array([-0.5, 2.5]) if modality == "ehr" else np.array([0.3, 9.0])
                frames.append(rows.assign(fold=fold["fold"], predicted_score=predicted,
                                          postprocessed_score=np.clip(predicted, 0, 8), baseline_mean_score=mean))
            predictions = pd.concat(frames).sort_values("sample_id").reset_index(drop=True)
            store = experiment_store(self.layout, "assessment", modality, model, key)
            with store.publish("assessment", "run", inputs=[train.manifest_ref("train", "run")],
                               details={"data_release": "v1_r3", "train_id": key}) as work:
                predictions.iloc[::-1].to_parquet(work / "oof_predictions.parquet", index=False)
                points = {rule: {name: {ep: operating_point(predictions[ep].to_numpy(bool),
                            predictions[column].to_numpy(float), rule, 0.8) for ep in ENDPOINTS}
                    for name, column in (("model", "predicted_score"), ("baseline_mean", "baseline_mean_score"))}
                    for rule in ("sensitivity", "specificity")}
                write_json(work / "operating_points.json", points)
            self.sources[modality] = (store, predictions)

    def rewrite_predictions(self, modality, rows):
        store, _ = self.sources[modality]
        ref = store.manifest_ref("assessment", "run")
        manifest = read_json(self.layout.resolve(ref))
        path = self.layout.resolve(store.output_ref(manifest, "oof_predictions.parquet"))
        rows.to_parquet(path, index=False)
        next(x for x in manifest["outputs"] if x["name"] == "oof_predictions.parquet")["bytes"] = path.stat().st_size
        write_json(self.layout.resolve(ref), manifest)

    def test_exact_scores_metrics_sources_and_id_reuse(self):
        result = run_assessment(self.layout, CONFIG)
        self.assertEqual(result["status"], "complete", result)
        entry = result["experiments"][0]
        store = experiment_store(self.layout, "assessment", "fusion", "weighted_late", "f1_a020")
        manifest = store.validate(entry["manifest"])
        self.assertEqual(manifest["inputs"], [{"manifest": self.sources[m][0].manifest_ref("assessment", "run"),
                                               "generation": 1} for m in ("ehr", "image")])
        folder = self.layout.resolve(entry["manifest"]).parent
        rows = pd.read_parquet(folder / "oof_predictions.parquet")
        self.assertEqual(rows.sample_id.tolist(), [f"sample-{i:02d}" for i in range(2, 12)])
        expected = 0.8 * self.sources["ehr"][1].predicted_score + 0.2 * self.sources["image"][1].predicted_score
        np.testing.assert_array_equal(rows.predicted_score, expected)
        np.testing.assert_array_equal(rows.ehr_shrink_score, 0.8 * rows.ehr_predicted_score + 0.2 * rows.baseline_mean_score)
        metrics = read_json(folder / "metrics.json")
        self.assertEqual(metrics["pooled_oof"]["weighted_late"], metric_pair(rows, expected.to_numpy()))
        self.assertNotEqual(metrics["pooled_oof"]["weighted_late"]["raw"],
                            metrics["pooled_oof"]["weighted_late"]["postprocessed"])
        self.assertEqual(len(metrics["folds"]), 5)
        differences = read_json(folder / "paired_differences.json")
        for mode in ("raw", "postprocessed"):
            for reference in ("ehr", "image", "baseline_mean", "ehr_shrink"):
                for metric in ("rmse", "mae"):
                    delta = differences["results"][mode][reference][metric]
                    self.assertAlmostEqual(delta["point"], metrics["pooled_oof"]["weighted_late"][mode][metric]
                                           - metrics["pooled_oof"][reference][mode][metric])
                    self.assertEqual(delta["valid_draws"], 40)
        points = read_json(folder / "operating_points.json")
        for rule in ("sensitivity", "specificity"):
            for name in ("model", "ehr", "image", "baseline_mean", "ehr_shrink"):
                for endpoint in ENDPOINTS:
                    self.assertGreaterEqual(points[rule][name][endpoint]["metrics"][rule], 0.8)
            self.assertTrue((folder / "threshold_based" / f"{rule}_80/confusion_matrices.svg").is_file())
        self.assertTrue((folder / "sources.json").is_file())
        self.assertIn("fixed-test", str(read_json(folder / "decision_summary.json")))
        self.assertFalse(self.layout.path("outputs", "models/train/fusion").exists())
        self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["status"], "unchanged")
        self.config["defaults"]["image_weight"] = 0.3
        write_json(self.layout.path("project", CONFIG), self.config)
        self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "EXPERIMENT_ID_CONFLICT")

    def test_invalid_pairing_is_rejected_for_either_source(self):
        for modality in ("ehr", "image"):
            original = self.sources[modality][1]
            variants = [original.iloc[:-1], pd.concat([original, original.iloc[:1]])]
            for column in ("subject_id", "hadm_id", "study_id", "fold", "burden_score", "phys_death", "high_burden", "baseline_mean_score"):
                changed = original.copy()
                changed.loc[0, column] += 1
                variants.append(changed)
            extra = original.copy()
            extra.loc[0, "sample_id"] = "sample-00"
            variants.append(extra)
            for changed in variants:
                with self.subTest(modality=modality, columns=changed.columns.tolist(), rows=len(changed)):
                    self.rewrite_predictions(modality, changed)
                    self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "INVALID_OOF_COVERAGE")
            for value in (np.nan, np.inf):
                changed = original.copy()
                changed.loc[0, "predicted_score"] = value
                self.rewrite_predictions(modality, changed)
                self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "INVALID_OOF_PREDICTIONS")
            self.rewrite_predictions(modality, original)

    def test_recursive_source_state_and_generation_are_checked(self):
        for modality, model, key in (("ehr", "catboost", "d2_l210"), ("image", "resnet50", "selected_image")):
            train = experiment_store(self.layout, "train", modality, model, key)
            path = self.layout.resolve(train.manifest_ref("train", "run"))
            original = read_json(path)
            state_path = self.layout.resolve(original["state"])
            state = read_json(state_path)
            write_json(state_path, {**state, "status": "failed"})
            self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "UPSTREAM_INCOMPLETE")
            write_json(state_path, {**state, "generation": 2})
            write_json(path, {**original, "generation": 2})
            self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "UPSTREAM_REBUILT")
            write_json(state_path, state)
            write_json(path, original)

    def test_patient_bootstrap_keeps_multiple_studies_together(self):
        rows = pd.DataFrame({"subject_id": [1, 1, 2], "burden_score": [0., 0., 0.]})
        scores = {key: np.zeros(3) for key in ("ehr", "image", "baseline_mean", "ehr_shrink")}
        scores["model"] = np.array([1., 1., 4.])
        result = paired_differences(rows, scores, 1000, 42)["results"]["raw"]["ehr"]
        # Resampling two patient groups gives only errors 1, 2 and 4 for MAE.
        self.assertEqual(result["mae"]["point"], 2.)
        self.assertEqual(result["mae"]["lower"], 1.)
        self.assertEqual(result["mae"]["upper"], 4.)
        self.assertEqual(result["rmse"]["point"], np.sqrt(6.))
        self.assertEqual(result["rmse"]["lower"], 1.)
        self.assertEqual(result["rmse"]["upper"], 4.)

    def test_image_snapshot_and_source_targets_are_validated(self):
        self.config["defaults"]["sensitivity_target"] = 0.9
        write_json(self.layout.path("project", CONFIG), self.config)
        self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "OPERATING_POINT_MISMATCH")
        self.config["defaults"]["sensitivity_target"] = 0.8
        write_json(self.layout.path("project", CONFIG), self.config)
        store = experiment_store(self.layout, "train", "image", "resnet50", "selected_image")
        ref = store.manifest_ref("train", "run")
        manifest = read_json(self.layout.resolve(ref))
        config_path = self.layout.resolve(store.output_ref(manifest, "effective_configuration.json"))
        snapshot = read_json(config_path)
        snapshot["image_shape"] = [8, 8]
        write_json(config_path, snapshot)
        next(x for x in manifest["outputs"] if x["name"] == "effective_configuration.json")["bytes"] = config_path.stat().st_size
        write_json(self.layout.resolve(ref), manifest)
        self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["error"], "INVALID_FUSION_SOURCE")

    def test_publication_failure_rolls_back_and_retry_succeeds(self):
        with patch("src.models.assessment.fusion.weighted_late.assessment.confusion_plot", side_effect=RuntimeError("synthetic")):
            self.assertEqual(run_assessment(self.layout, CONFIG)["experiments"][0]["status"], "failed")
        self.assertFalse(self.layout.path("outputs", "models/assessment/fusion/weighted_late/f1_a020").exists())
        self.assertEqual(run_assessment(self.layout, CONFIG)["status"], "complete")

    def test_invalid_parameters_fail_before_creating_fusion_outputs(self):
        defaults = copy.deepcopy(self.config["defaults"])
        for key, value in (("image_weight", float("nan")), ("image_weight", 1.1), ("bootstrap_draws", 0),
                           ("bootstrap_seed", -1), ("sensitivity_target", True)):
            self.config["defaults"] = {**defaults, key: value}
            # Non-finite JSON is rejected at the writer boundary; test the validator directly.
            if key == "image_weight" and np.isnan(value):
                from src.models.assessment.fusion.weighted_late.assessment import validate_parameters
                with self.assertRaises(PipelineError):
                    validate_parameters(self.config["defaults"])
                continue
            write_json(self.layout.path("project", CONFIG), self.config)
            with self.assertRaises(PipelineError):
                run_assessment(self.layout, CONFIG)
        self.assertFalse(self.layout.path("outputs", "models/assessment/fusion").exists())
