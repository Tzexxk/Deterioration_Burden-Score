"""Freeze from development, then reuse the exact two published base evaluations."""

import unittest
from statistics import median

import numpy as np
import pandas as pd

from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from src.models.assessment.fusion.learned_late.assessment import fit_coefficients
from src.models.assessment.fusion.weighted_late.assessment import run_assessment
from src.models.experiments import experiment_store
from src.models.test.evaluation import (
    SAMPLE_COLUMNS,
    frozen_thresholds,
    output,
    write_fixed_evaluation,
)
from src.models.test.fusion.comparison import run_comparison
from src.models.test.fusion.workflow import CANDIDATES, binding, run_test
from tests.models.train.fusion.feature_fusion.test_training import TEMPLATE_ROOT
from tests.models.train.fusion.learned_late import test_workflow as nested

CONFIG = "configs/models/test/fusion/{}.json"


class FusionFixedTestTests(unittest.TestCase):
    def test_development_freeze_precedes_shared_fixed_evaluation(self):
        fixture = nested.NestedFusionTests.check_workflow(self, use_median=True)
        layout, package = fixture.layout, fixture.package
        path = "configs/models/assessment/fusion/weighted_late.json"
        f1 = read_json(TEMPLATE_ROOT / "assessment/fusion/weighted_late.json")
        f1["data_release"] = "v1_r3"
        f1["defaults"].update(image_weight=0.6, bootstrap_draws=30)
        f1["experiments"][0]["id"] = "f1_a060"
        write_json(layout.path("project", path), f1)
        self.assertEqual(run_assessment(layout, path)["status"], "complete")
        configs = {}
        for model in CANDIDATES:
            config = read_json(TEMPLATE_ROOT / f"test/fusion/{model}.json")
            config["data_release"] = "v1_r3"
            configs[model] = config
            write_json(layout.path("project", CONFIG.format(model)), config)
        development = package.samples.iloc[package.cv["development_indices"]]
        mean = float(development.burden_score.to_numpy(float).mean())
        bases = {}
        for family, model, key, field in (("ehr", "catboost", "d2_l210", "best_round"),
                                         ("image", "resnet50", "selected_image", "best_epoch")):
            assess = experiment_store(layout, "assessment", family, model, key)
            assessment_ref = assess.manifest_ref("assessment", "run")
            assessment = assess.validate(assessment_ref)
            train = read_json(layout.resolve(assessment["inputs"][0]["manifest"]))
            snap = read_json(output(assess, train, "effective_configuration.json"))
            lengths = read_json(output(assess, train, "training_summary.json"))["folds"]
            points = read_json(output(assess, assessment, "operating_points.json"))
            thresholds = frozen_thresholds(development, mean, points)
            store = experiment_store(layout, "test", family, model, key)
            with store.publish("test", "refit", inputs=[assessment_ref],
                               details={"data_release": "v1_r3", "assessment_id": key}) as work:
                write_json(work / "thresholds.json", thresholds)
                summary = {"train_sample_ids": development.sample_id.tolist(), "baseline_mean": mean,
                           "refit_rounds": int(median(f[field] for f in lengths)), "training_parameters": snap["parameters"],
                           "test_consumed_for_fitting": False}
                if family == "image":
                    summary.update({k: snap[k] for k in ("resolved_image_source", "image_shape", "runtime_versions")})
                write_json(work / "fit_summary.json", summary)
                (work / ("model.cbm" if family == "ehr" else "model.pt")).write_bytes(b"synthetic fixture")
            bases[family] = (store, thresholds)

        stores, refits, summaries = {}, {}, {}
        for model, candidate in CANDIDATES.items():
            config = configs[model]
            store = experiment_store(layout, "test", "fusion", model, config["id"])
            stores[model] = store
            result = run_test(layout, CONFIG.format(model), "refit", model)
            frozen = store.validate(result["manifest"])
            refits[model] = frozen
            self.assertEqual(len(frozen["inputs"]), 3)
            self.assertEqual(frozen["inputs"][2]["manifest"]["path"],
                             f"models/assessment/fusion/{model}/{config['assessment_id']}/manifest.json")
            summaries[model] = read_json(output(store, frozen, "fit_summary.json"))
            self.assertEqual(summaries[model]["candidate"], candidate)
        self.assertEqual(summaries["weighted_late"]["coefficients"]["f1"], {"alpha": .6, "bias": 0.})
        f3_store = experiment_store(layout, "assessment", "fusion", "learned_late", "f3_nested_median")
        f3 = f3_store.validate(f3_store.manifest_ref("assessment", "run"))
        oof = pd.read_parquet(output(f3_store, f3, "oof_predictions.parquet"))
        expected = fit_coefficients(oof.ehr.to_numpy(float), oof.image.to_numpy(float), oof.burden_score.to_numpy(float))
        self.assertEqual(summaries["learned_late"]["coefficients"]["f3b"], expected["f3b"])

        rows = package.samples.iloc[package.cv["test_indices"]][SAMPLE_COLUMNS].reset_index(drop=True)
        for family, (base, thresholds) in bases.items():
            ref = base.manifest_ref("test", "refit")
            raw = np.arange(len(rows), dtype=float) + (0.2 if family == "ehr" else 0.7)
            with base.publish("test", "evaluate", inputs=[ref], details={"data_release": "v1_r3"}) as work:
                write_fixed_evaluation(work, {"data_release": "v1_r3"}, rows, raw, mean, thresholds)
        ehr_store = bases["ehr"][0]
        ehr_eval_path = layout.resolve(ehr_store.manifest_ref("test", "evaluate"))
        ehr_eval = read_json(ehr_eval_path)
        ehr_refit = read_json(layout.resolve(ehr_store.manifest_ref("test", "refit")))
        write_json(ehr_eval_path, {**ehr_eval, "inputs": ehr_refit["inputs"]})
        with self.assertRaisesRegex(PipelineError, "INVALID_TEST_EVALUATION"):
            run_test(layout, CONFIG.format("weighted_late"), "evaluate", "weighted_late")
        write_json(ehr_eval_path, ehr_eval)
        for model, candidate in CANDIDATES.items():
            path = CONFIG.format(model)
            result = run_test(layout, path, "evaluate", model)
            store = stores[model]
            evaluation = store.validate(result["manifest"])
            self.assertEqual(evaluation["inputs"][0], binding(refits[model]))
            self.assertEqual(len(evaluation["inputs"]), 3)
            self.assertEqual(run_test(layout, path, "evaluate", model)["status"], "unchanged")
            scores = pd.read_parquet(output(store, evaluation, "test_predictions.parquet"))
            np.testing.assert_array_equal(scores.sample_id, rows.sample_id)
            coef = summaries[model]["coefficients"][candidate]
            alpha, bias = coef["alpha"], coef["bias"]
            ehr = np.arange(len(rows), dtype=float) + .2
            image = np.arange(len(rows), dtype=float) + .7
            np.testing.assert_array_equal(scores.predicted_score, (1-alpha)*ehr + alpha*image + bias)

        selected = ["ehr/catboost/d2_l210", "image/resnet50/selected_image",
                    *(f"fusion/{model}/{config['id']}" for model, config in configs.items())]
        with self.assertRaisesRegex(PipelineError, "INVALID_TEST_EVALUATION"):
            run_comparison(layout, "selected", evaluation_ids=selected[:-1], draws=30)
        result = run_comparison(layout, "selected", evaluation_ids=selected, draws=30)
        compare = experiment_store(layout, "test", "fusion", "comparison", "selected")
        manifest = compare.validate(result["manifest"])
        self.assertEqual(len(manifest["inputs"]), 4)
        paired = read_json(output(compare, manifest, "paired_differences.json"))
        self.assertEqual(paired["raw"]["f3b"]["f1"]["rmse"]["requested_draws"], 30)
        self.assertIn("f3a", paired["raw"])
        self.assertIn("ehr_shrink", paired["raw"])
        self.assertEqual(run_comparison(layout, "selected", evaluation_ids=selected, draws=30)["status"], "unchanged")

        path = CONFIG.format("weighted_late")
        write_json(layout.path("project", path), {**configs["weighted_late"], "image_weight": .8})
        with self.assertRaisesRegex(PipelineError, "REFIT_CONFIG_MISMATCH"):
            run_test(layout, path, "evaluate", "weighted_late")
