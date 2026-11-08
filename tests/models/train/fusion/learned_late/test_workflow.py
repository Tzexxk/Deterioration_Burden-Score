"""Exercise nested splits, fixed-round fits and coefficient isolation on synthetic data."""

import json
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.models.assessment.fusion import compare as compare_cli
from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.assessment.fusion.comparison import run_comparison
from src.models.assessment.fusion.feature_fusion.assessment import (
    run_assessment as assess_feature,
)
from src.models.assessment.fusion.learned_late import assessment
from src.models.assessment.fusion.weighted_late.assessment import (
    run_assessment as assess_fixed,
)
from src.models.experiments import experiment_store
from src.models.train.fusion.learned_late import training
from src.models.train.image.resnet50.training import BurdenHeadResNet, EventHead
from tests.models.train.fusion.feature_fusion.test_training import (
    ASSESSMENT_CONFIG,
    TEMPLATE_ROOT,
    FusionFixture,
)


def tiny_event_model(_config, weights=None):
    layer = nn.Sequential(nn.Flatten(), nn.Linear(48, 4), nn.ReLU())
    return BurdenHeadResNet(nn.Sequential(OrderedDict(layer4=layer)), EventHead(4))


class NestedFusionTests(unittest.TestCase):
    def test_coefficient_formula_and_constant_difference_fallback(self):
        ehr = np.array([1., 2., 3.])
        d = np.array([-1., 0., 1.])
        y = ehr + .5 * d + 2
        self.assertEqual(assessment.fit_coefficients(ehr, ehr + d, y), {
            "f3a": {"alpha": .5, "bias": 0.}, "f3b": {"alpha": .5, "bias": 2.}, "ehr_bias": {"alpha": 0., "bias": 2.}})
        self.assertEqual(assessment.fit_coefficients(ehr, ehr, y)["f3b"], {"alpha": 0., "bias": 2.})
        self.assertEqual(assessment.fit_coefficients(ehr, ehr + 1, y)["f3b"], {"alpha": 0., "bias": 2.})
        self.assertEqual(assessment.fit_coefficients(ehr, ehr + d, ehr + 2*d)["f3b"]["alpha"], 1.)
        self.assertEqual(assessment.fit_coefficients(ehr, ehr + d, ehr - d)["f3a"]["alpha"], 0.)

    def test_nested_workflow_uses_fifteen_disjoint_fits_and_matches_outer_scores(self):
        self.check_workflow(use_median=False)

    def test_training_scope_medians_use_only_fold_local_histories(self):
        self.assertEqual(training.median_lengths([
            {"ehr_best_round": n, "image_best_epoch": e} for n, e in zip([1, 2, 3, 4], [4, 4, 5, 5])]),
            {"ehr_rounds": 3, "image_epochs": 5})
        self.check_workflow(use_median=True)

    def check_workflow(self, *, use_median):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        fixture = FusionFixture(self)
        layout, package = fixture.layout, fixture.package
        configs = {}
        key = "f3_nested_median" if use_median else "f3_nested_fixed"
        for stage in ("train", "assessment"):
            path = f"configs/models/{stage}/fusion/learned_late.json"
            config = read_json(TEMPLATE_ROOT / f"{stage}/fusion/learned_late.json")
            config["data_release"] = "v1_r3"
            config["experiments"][0]["id"] = key
            if stage == "train" and not use_median:
                config["defaults"].pop("round_policy")
                config["defaults"].update(ehr_rounds=2, image_epochs=2)
            if stage == "assessment":
                config["defaults"]["bootstrap_draws"] = 30
                config["experiments"][0]["parameters"]["train_id"] = key
            write_json(layout.path("project", path), config)
            configs[stage] = path
        resolve = training.resolve_sources

        def synthetic_sources(*args):
            sources, records, images = resolve(*args)
            image = records["image"]["effective_training_configuration"]["parameters"]
            image.update(device="cpu", batch_size=2, num_workers=0, head_only_epochs=1, max_epochs=2,
                         head_lr=.001, weight_decay=.0001, weights={"root": "project", "path": "pyproject.toml"})
            return sources, records, images

        with patch.object(training, "resolve_sources", side_effect=synthetic_sources), \
                patch.object(training.image_model.training, "build_model", side_effect=tiny_event_model), \
                patch.object(training.image_model.training, "load_weights", return_value={}), \
                patch.object(training, "calibrate_ehr", wraps=training.calibrate_ehr) as calibrate, \
                patch.object(training.image_model, "refit", wraps=training.image_model.refit) as refit:
            result = training.run_training(layout, configs["train"])
            self.assertEqual(result["status"], "complete", result)
            self.assertEqual(refit.call_count, 15)
            self.assertEqual(calibrate.call_count, 50 if use_median else 0)
            self.assertEqual(training.run_training(layout, configs["train"])["experiments"][0]["status"], "unchanged")
            self.assertEqual(refit.call_count, 15)
        train = experiment_store(layout, "train", "fusion", training.MODEL, key)
        tm = train.validate(result["experiments"][0]["manifest"])
        summary = read_json(training.output(train, tm, "training_summary.json"))
        self.assertEqual((summary["inner_fits_per_modality"], summary["outer_fits_per_modality"]), (10, 5))
        self.assertEqual(len(set(summary["base_fit_ids"])), 15)
        self.assertEqual(summary["calibration_fits_per_modality"], 50 if use_median else 0)
        if use_median:
            for base_id in summary["base_fit_ids"]:
                prefix = f"base_fits/{base_id}"
                meta = read_json(training.output(train, tm, f"{prefix}/fit.json"))
                self.assertEqual([row["validation_fold"] for row in meta["calibration"]], meta["train_folds"])
                for row in meta["calibration"]:
                    folder = f"{prefix}/calibration/fold_{row['validation_fold']}"
                    for family in ("ehr", "image"):
                        interface = read_json(training.output(train, tm, f"{folder}/{family}/interface.json"))
                        tr, val = set(interface["train_sample_ids"]), set(interface["validation_sample_ids"])
                        self.assertTrue(tr.isdisjoint(val))
                        self.assertEqual(tr | val, set(meta["train_sample_ids"]))
                        self.assertTrue((tr | val).isdisjoint(meta["score_sample_ids"]))
        with patch.object(assessment, "resolve_sources", side_effect=synthetic_sources), \
                patch.object(training.image_model, "refit", side_effect=AssertionError("Assessment cannot fit bases")):
            result = assessment.run_assessment(layout, configs["assessment"])
        self.assertEqual(result["status"], "complete", result)
        store = experiment_store(layout, "assessment", "fusion", training.MODEL, key)
        manifest = store.validate(result["experiments"][0]["manifest"])
        self.assertEqual(manifest["inputs"], [{"manifest": tm["self"], "generation": tm["generation"]}])
        rows = pd.read_parquet(training.output(store, manifest, "oof_predictions.parquet"))
        self.assertEqual(rows.sample_id.tolist(), sorted(package.samples.iloc[package.cv["development_indices"]].sample_id))
        for outer in range(1, 6):
            inner = pd.read_parquet(training.output(store, manifest, f"fold_{outer}/inner_oof.parquet"))
            outer_rows = rows[rows.fold == outer]
            self.assertTrue(set(inner.subject_id).isdisjoint(outer_rows.subject_id))
            for (key, scored), group in inner.groupby(["base_fit_id", "inner_fold"]):
                fit = read_json(training.output(train, tm, f"base_fits/{key}/fit.json"))
                self.assertEqual(fit["score_folds"], sorted([outer, scored]))
                self.assertTrue(set(group.sample_id).isdisjoint(fit["train_sample_ids"]))
            coef = read_json(training.output(store, manifest, f"fold_{outer}/coefficients.json"))["coefficients"]
            for candidate, settings in coef.items():
                alpha, bias = settings["alpha"], settings["bias"]
                np.testing.assert_array_equal(outer_rows[candidate], (1-alpha)*outer_rows.ehr + alpha*outer_rows.image + bias)
        np.testing.assert_array_equal(rows.predicted_score, rows.f3b)
        np.testing.assert_array_equal(rows.fixed_alpha_0p6, .4*rows.ehr + .6*rows.image)
        return fixture

    def test_comparison_publishes_protocol_matched_groups(self):
        fixture = self.check_workflow(use_median=False)
        layout = fixture.layout
        self.assertEqual(fixture.train()["status"], "complete")
        self.assertEqual(assess_feature(layout, ASSESSMENT_CONFIG)["status"], "complete")
        fixed_path = "configs/models/assessment/fusion/weighted_late.json"
        fixed = read_json(TEMPLATE_ROOT / "assessment/fusion/weighted_late.json")
        fixed["data_release"] = "v1_r3"
        fixed["defaults"]["bootstrap_draws"] = 30
        write_json(layout.path("project", fixed_path), fixed)
        self.assertEqual(assess_fixed(layout, fixed_path)["status"], "complete")
        selected = ["ehr/catboost/d2_l210", "image/resnet50/selected_image", "fusion/weighted_late/f1_a020",
                    "fusion/feature_fusion/f2_pca08", "fusion/learned_late/f3_nested_fixed"]
        args = ["--id", "synthetic_comparison"] + [part for name in selected for part in ("--assessment", name)]
        with patch.object(compare_cli, "Layout", return_value=layout), \
                patch.object(compare_cli, "run_comparison", partial(run_comparison, draws=30)), \
                redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()):
            self.assertEqual(compare_cli.main(args), 0)
        result = json.loads(stdout.getvalue())
        store = experiment_store(layout, "assessment", "fusion", "comparison", "synthetic_comparison")
        manifest = store.validate(result["manifest"])
        self.assertEqual(len(manifest["inputs"]), 5)
        self.assertEqual(manifest["configs"], [])
        snapshot = read_json(training.output(store, manifest, "effective_configuration.json"))
        self.assertEqual(snapshot["parameters"]["assessment_ids"], sorted(selected))
        summary = read_json(training.output(store, manifest, "summary.json"))
        self.assertIn("EHR_shrink/f1_a020", summary["protocols"]["historical"])
        self.assertIn("f3b", summary["protocols"]["nested"])
        self.assertNotIn("f3b", summary["protocols"]["historical"])
        nested = read_json(training.output(store, manifest, "nested/paired_differences.json"))
        self.assertEqual(nested["raw"]["f3b"]["ehr"]["rmse"]["requested_draws"], 30)
        self.assertTrue(training.output(store, manifest, "historical/threshold_free/rmse_relative_pct.svg").is_file())
        again = run_comparison(layout, "synthetic_comparison", assessment_ids=selected[::-1], draws=30)
        self.assertEqual(again["status"], "unchanged")
        with self.assertRaisesRegex(PipelineError, "DUPLICATE_ASSESSMENT"):
            run_comparison(layout, "duplicate", assessment_ids=[*selected, selected[0]], draws=30)
