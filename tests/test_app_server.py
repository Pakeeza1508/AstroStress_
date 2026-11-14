"""Small regression tests for the local AstroStress HTTP data bridge.

Run with: python -m unittest discover -s tests
No WESAD download or trained joblib file is required for these tests.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app_server import clean_table, research_payload, subjects_payload, window_payload, reliability_payload, global_shap_payload, prerecorded_shap_payload, physiology_trend_payload


class LocalApiPayloadTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "features": pd.DataFrame({
                "subject": ["S2", "S2", "S10"],
                "label": [0, 1, 0],
                "hr_mean": [70.1, 90.2, 65.3],
                "window_start_s": [0.0, 60.0, 0.0],
                "window_end_s": [120.0, 180.0, 120.0],
            }),
            "summary": {
                "run_id": "test-run", "best_model": "Random Forest",
                "best_accuracy": 0.8874, "best_f1": 0.7835,
                "best_auc": 0.9436, "n_subjects": 15, "n_windows": 608,
                "n_candidate_features": 19,
            },
        }
        self.model = {"feature_cols": ["hr_mean"], "model_name": "Random Forest"}

    def test_subjects_sorted_and_counted(self):
        self.assertEqual(subjects_payload(self.data), [
            {"id": "S2", "windows": 2}, {"id": "S10", "windows": 1}])

    @patch("app_server.predict_recorded_window", return_value=(0.1232, 0))
    def test_recorded_window_has_real_provenance_and_label(self, predict):
        payload = window_payload(self.data, self.model, "S10", 0)
        self.assertEqual(payload["source"], "WESAD laboratory recording")
        self.assertEqual(payload["evaluation_type"], "in_sample_final_model_replay")
        self.assertFalse(payload["is_live"])
        self.assertEqual(payload["prediction"]["stress_probability"], 0.1232)
        self.assertEqual(payload["prediction"]["actual_class"], 0)
        self.assertTrue(payload["prediction"]["matches_protocol_label"])
        self.assertEqual(payload["features"]["hr_mean"], 65.3)
        predict.assert_called_once()

    def test_physiology_trend_uses_measured_features_and_correct_order(self):
        self.data["features"]["HR_mean"] = [72.0, 84.0, 97.88]
        self.data["features"]["EDA_mean"] = [0.11, np.nan, 0.381]
        trend = physiology_trend_payload(self.data, "S2")
        self.assertEqual(trend["run_id"], "test-run")
        self.assertEqual(trend["source"], "verified WESAD processed physiological features")
        self.assertFalse(trend["is_raw_waveform"])
        self.assertEqual(trend["window_count"], 2)
        self.assertEqual([row["index"] for row in trend["windows"]], [0, 1])
        self.assertEqual(trend["windows"][0]["HR_mean"], 72.0)
        self.assertEqual(trend["windows"][0]["EDA_mean"], 0.11)
        self.assertIsNone(trend["windows"][1]["EDA_mean"])
        self.assertEqual(trend["units"]["HR_mean"], "BPM")
        with self.assertRaisesRegex(ValueError, "Unknown WESAD participant"):
            physiology_trend_payload(self.data, "S999")

    def test_invalid_window_and_subject_rejected(self):
        for subject, index in [("S999", 0), ("S2", -1), ("S2", 2)]:
            with self.subTest(subject=subject, index=index):
                with self.assertRaises(ValueError):
                    window_payload(self.data, self.model, subject, index)

    def test_json_clean_table_replaces_nan_with_none(self):
        frame = pd.DataFrame({"model": ["RF"], "auc": [np.nan]})
        self.assertEqual(clean_table(frame, ["model", "auc"]),
                         [{"model": "RF", "auc": None}])

    def test_research_summary_and_actual_rows(self):
        self.data["comparison"] = pd.DataFrame({
            "model": ["Random Forest"], "accuracy": [0.8874],
            "f1": [0.7835], "auc": [0.9436]})
        self.data["loso"] = pd.DataFrame({
            "subject": ["S2"], "accuracy": [0.9],
            "f1": [0.8], "n_test": [2]})
        self.data["personalized"] = pd.DataFrame({
            "subject": ["S2"], "generic_accuracy": [0.80],
            "personal_accuracy": [0.81], "improvement_pp": [1.0]})
        payload = research_payload(self.data)
        self.assertEqual(payload["run_id"], "test-run")
        self.assertEqual(payload["comparison"][0]["f1"], 0.7835)
        self.assertEqual(payload["personalized"][0]["improvement_pp"], 1.0)


    def test_reliability_report_requires_matching_original_run(self):
        report={
            "main_run_id":"test-run",
            "features_sha256":"valid-feature-hash",
            "full_overlapping_windows":{
                "n_windows":608,"n_subjects":15,
                "brier_score":0.08369,"ece_equal_frequency_5_bins":0.04095,
                "pooled_accuracy":0.8865,
                "calibration_bins":[{"n":122,"mean_predicted":0.1,"observed_stress_rate":0.1}]*5,
                "selective_curve":[
                    {"threshold":0.5+0.05*i,"coverage":1.0 if i==0 else 0.8,
                     "n_accepted":608 if i==0 else 486,"accuracy":0.8865 if i==0 else 0.94}
                    for i in range(10)]
            },
            "non_overlapping_sensitivity":{"n_windows":321,"selective_curve":[{"accuracy":0.9}]}
        }
        self.data["summary"]["features_sha256"]="valid-feature-hash"
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"experiment_c_summary.json"
            path.write_text(json.dumps(report),encoding="utf-8")
            actual=reliability_payload(self.data,path)
            self.assertEqual(actual["n_windows"],608)
            self.assertEqual(actual["run_id"],"test-run")
            report["main_run_id"]="different-run"
            path.write_text(json.dumps(report),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"do not belong"):
                reliability_payload(self.data,path)
            report["main_run_id"]="test-run"
            report["features_sha256"]="wrong-hash"
            path.write_text(json.dumps(report),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"do not belong"):
                reliability_payload(self.data,path)


    def test_global_shap_uses_verified_saved_importance(self):
        self.data["shap"] = {
            "run_id":"test-run", "model_used":"Random Forest",
            "n_windows_explained":608, "value_units":"probability of stress",
            "global_importance":{"EDA_mean":0.045, "HRV_RMSSD":0.123},
        }
        response=global_shap_payload(self.data)
        self.assertEqual(response["features"], [
            {"feature":"HRV_RMSSD", "importance":0.123},
            {"feature":"EDA_mean", "importance":0.045},
        ])
        self.assertEqual(response["evaluation_type"],"in_sample_final_model_explanation")
        self.data["shap"]["run_id"]="another-run"
        with self.assertRaisesRegex(ValueError,"does not match"):
            global_shap_payload(self.data)

    def test_global_shap_unavailable_or_invalid(self):
        with self.assertRaises(FileNotFoundError):
            global_shap_payload(self.data)
        self.data["shap"]={
            "run_id":"test-run", "model_used":"Random Forest",
            "n_windows_explained":608, "value_units":"probability",
            "global_importance":{"HR_mean":float("nan")},
        }
        with self.assertRaisesRegex(ValueError,"Invalid SHAP"):
            global_shap_payload(self.data)


    def test_prerecorded_shap_requires_matching_provenance(self):
        self.data["summary"]["features_sha256"] = "feature-hash"
        payload = {
            "run_id": "test-run", "features_sha256": "feature-hash",
            "model": "Random Forest", "n_windows": 608,
            "by_subject": {"S10": [[{"feature": "HR_mean", "value": 0.12}]]},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "window_shap.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = prerecorded_shap_payload(self.data, "S10", 0, path)
            self.assertEqual(result["contributions"][0]["feature"], "HR_mean")
            self.assertEqual(result["run_id"], "test-run")
            payload["features_sha256"] = "different"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                prerecorded_shap_payload(self.data, "S10", 0, path)

    def test_prerecorded_shap_rejects_missing_and_nonfinite_values(self):
        self.data["summary"]["features_sha256"] = "feature-hash"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "window_shap.json"
            with self.assertRaises(FileNotFoundError):
                prerecorded_shap_payload(self.data, "S10", 0, path)
            payload = {
                "run_id": "test-run", "features_sha256": "feature-hash",
                "model": "Random Forest", "n_windows": 608,
                "by_subject": {"S10": [[{"feature": "HR_mean", "value": float("nan")}]]},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid"):
                prerecorded_shap_payload(self.data, "S10", 0, path)


if __name__ == "__main__":
    unittest.main()
