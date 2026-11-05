"""Check paired bootstrap rank metrics and threshold reporting."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.assessment.bootstrap import patient_weights, rank_metrics
from src.models.assessment.operating_points import choose_threshold, operating_point


class BootstrapMetricTests(unittest.TestCase):
    def test_weighted_ranking_with_ties_matches_sklearn(self):
        labels = np.array([True, False, True, False, False])
        scores = np.array([0.8, 0.8, 0.7, 0.7, 0.5])
        weights = patient_weights([1, 1, 2, 3, 3], 31, 42)
        np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
        np.testing.assert_array_equal(weights[:, 3], weights[:, 4])
        roc, ap = rank_metrics(labels, scores, weights)
        for i, w in enumerate(weights):
            if w[labels].sum() == 0 or w[~labels].sum() == 0:
                self.assertTrue(np.isnan(roc[i]) and np.isnan(ap[i]))
            else:
                self.assertAlmostEqual(roc[i], roc_auc_score(labels, scores, sample_weight=w))
                self.assertAlmostEqual(ap[i], average_precision_score(labels, scores, sample_weight=w))

    def test_ties_and_empty_alerts_have_explicit_operating_points(self):
        labels = np.array([True, True, False, False, False])
        scores = np.array([0.9, 0.7, 0.8, 0.7, 0.4])
        self.assertEqual(choose_threshold(labels, scores, "sensitivity", 0.8), 0.7)
        self.assertEqual(choose_threshold(labels, scores, "specificity", 0.8), 0.9)
        point = operating_point(labels, np.zeros(len(labels)), "specificity", 0.8)
        self.assertIsNone(point["threshold"])
        self.assertEqual(point["comparison"], "no_alerts")
        self.assertEqual(point["confusion_matrix"], {"tp": 0, "fp": 0, "fn": 2, "tn": 3})
        self.assertEqual(point["metrics"]["ppv"], None)
        self.assertEqual(point["metrics"]["npv"], 0.6)
