"""Small, deterministic tests for Experiment C metrics (no retraining required)."""
import unittest

import pandas as pd

from experiment_c import (
    equal_frequency_calibration, non_overlapping_subset,
    selective_curve, summary_metrics, assert_reproduces_original,
)


class ExperimentCTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            {"subject": "S2", "window_start_s": 0, "window_end_s": 120, "y_true": 0, "y_pred": 0, "p_stress": .1, "confidence": .9},
            {"subject": "S2", "window_start_s": 60, "window_end_s": 180, "y_true": 1, "y_pred": 0, "p_stress": .4, "confidence": .6},
            {"subject": "S2", "window_start_s": 120, "window_end_s": 240, "y_true": 1, "y_pred": 1, "p_stress": .9, "confidence": .9},
            {"subject": "S3", "window_start_s": 0, "window_end_s": 120, "y_true": 1, "y_pred": 1, "p_stress": .8, "confidence": .8},
            {"subject": "S3", "window_start_s": 60, "window_end_s": 180, "y_true": 0, "y_pred": 1, "p_stress": .7, "confidence": .7},
            {"subject": "S3", "window_start_s": 180, "window_end_s": 300, "y_true": 0, "y_pred": 0, "p_stress": .2, "confidence": .8},
        ])

    def test_calibration_count_and_ece(self):
        bins, ece = equal_frequency_calibration(self.data, bins=5)
        self.assertEqual(sum(b["n"] for b in bins), len(self.data))
        self.assertTrue(0 <= ece <= 1)

    def test_threshold_curve_no_optimization(self):
        result = selective_curve(self.data, thresholds=[.5, .8, .95])
        self.assertEqual(result[0]["coverage"], 1)
        self.assertEqual(result[-1]["n_accepted"], 0)
        self.assertIsNone(result[-1]["accuracy"])
        self.assertLess(result[1]["coverage"], result[0]["coverage"])

    def test_non_overlap_uses_timestamps_not_row_parity(self):
        subset = non_overlapping_subset(self.data)
        self.assertEqual(len(subset), 4)
        self.assertEqual(subset.loc[subset["subject"] == "S2", "window_start_s"].tolist(), [0, 120])
        self.assertEqual(subset.loc[subset["subject"] == "S3", "window_start_s"].tolist(), [0, 180])

    def test_metrics_have_observed_values(self):
        metrics = summary_metrics(self.data)
        self.assertAlmostEqual(metrics["pooled_accuracy"], 4/6)
        self.assertEqual(metrics["n_subjects"], 2)
        self.assertEqual(metrics["n_windows"], 6)

    def test_mismatch_stops_export(self):
        with self.assertRaisesRegex(ValueError, "EXPERIMENT C STOPPED"):
            assert_reproduces_original(self.data, {
                "best_accuracy": .99, "best_f1": .99, "best_auc": .99,
            })


if __name__ == "__main__":
    unittest.main()
