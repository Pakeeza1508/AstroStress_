"""Fast synthetic tests for the recorded-window replay API (no WESAD needed)."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from astrostress_outputs import file_sha256
from astrostress_preprocessing import FoldPreprocessor
from astrostress_inference import (
    ModelArtifactError, load_verified_model, predict_recorded_window
)


class TestRecordedWindowInference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.features = pd.DataFrame({
            "subject": ["S2", "S2", "S3", "S3", "S4", "S4"],
            "label_binary": [0, 0, 1, 1, 0, 1],
            "HR_mean": [65.0, 70.0, 98.0, 105.0, 68.0, 110.0],
            "EDA_mean": [1.0, 1.4, 9.0, 11.0, 2.0, 12.0],
        })
        self.features.to_csv(self.out / "features.csv", index=False)
        h = file_sha256(self.out / "features.csv")
        self.summary = {
            "run_id": "test-run-1", "best_model": "Random Forest",
            "features_sha256": h,
        }
        cols = ["HR_mean", "EDA_mean"]
        prep = FoldPreprocessor(cols).fit(self.features[cols])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(prep.transform(self.features[cols]), self.features["label_binary"])
        self.artifact = {
            "run_id": "test-run-1",
            "features_sha256": h,
            "model_name": "Random Forest",
            "feature_cols": cols,
            "selected_features": prep.selected_features_,
            "preprocessor": prep, "model": model,
        }
        joblib.dump(self.artifact, self.out / "trained_model.joblib")

    def test_real_saved_model_prediction(self):
        artifact = load_verified_model(self.out, self.summary)
        p, label = predict_recorded_window(artifact, self.features.iloc[[2]])
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)
        self.assertEqual(label, int(p >= 0.5))

    def test_feature_column_order_does_not_change_probability(self):
        artifact = load_verified_model(self.out, self.summary)
        row = self.features.iloc[[3]]
        a = predict_recorded_window(artifact, row)[0]
        b = predict_recorded_window(artifact, row[["EDA_mean", "subject", "HR_mean", "label_binary"]])[0]
        self.assertAlmostEqual(a, b)

    def test_missing_feature_does_not_silently_impute(self):
        artifact = load_verified_model(self.out, self.summary)
        with self.assertRaisesRegex(ValueError, "Missing feature"):
            predict_recorded_window(artifact, self.features[["HR_mean"]].iloc[[0]])

    def test_rejects_wrong_run(self):
        bad = dict(self.summary, run_id="different-run")
        with self.assertRaises(ModelArtifactError):
            load_verified_model(self.out, bad)

    def test_rejects_wrong_model_family(self):
        bad = dict(self.summary, best_model="XGBoost")
        with self.assertRaises(ModelArtifactError):
            load_verified_model(self.out, bad)

    def test_rejects_changed_features_file(self):
        self.features.iloc[:2].to_csv(self.out / "features.csv", index=False)
        with self.assertRaises(ModelArtifactError):
            load_verified_model(self.out, self.summary)

    def test_missing_model_has_clear_error(self):
        (self.out / "trained_model.joblib").unlink()
        with self.assertRaisesRegex(ModelArtifactError, "missing"):
            load_verified_model(self.out, self.summary)


if __name__ == "__main__":
    unittest.main()
