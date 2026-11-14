"""Signal-loader unit tests: synthetic arrays only, no WESAD download needed."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import astrostress_signals as signals


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({
            "subject": ["S10", "S10"],
            "label": [0, 1],
            "window_start_s": [10.0, 70.0],
            "window_end_s": [130.0, 190.0],
        })
        self.recorded_ecg = np.arange(250 * signals.ECG_HZ, dtype=float) / 1000
        self.recorded_eda = np.arange(250 * signals.EDA_HZ, dtype=float) / 100

    @patch("astrostress_signals._cached_subject")
    @patch("astrostress_signals.source_description")
    def test_windows_match_feature_timestamps(self, source, cached):
        source.return_value = {"available": True, "kind": "directory", "path": "trusted"}
        cached.return_value = (self.recorded_ecg, self.recorded_eda)
        item = signals.waveform_payload(self.features, "S10", 1)
        self.assertEqual(item["window_start_s"], 70.0)
        self.assertEqual(item["window_index"], 1)
        self.assertEqual(len(item["ecg"]["samples"]), 7000)
        self.assertEqual(len(item["eda"]["samples"]), 480)
        self.assertEqual(item["ecg"]["samples"][0], 49.0)  # 70 * 700 / 1000
        self.assertEqual(item["eda"]["samples"][0], 2.8)  # 70 * 4 / 100

    def test_unknown_recording_rejected_before_read(self):
        with self.assertRaises(ValueError):
            signals.waveform_payload(self.features, "../../S2", 0)

    @patch("astrostress_signals.source_description", return_value={"available":False,"kind":None,"note":"No files"})
    def test_no_source_gives_clear_error(self, source):
        with self.assertRaisesRegex(FileNotFoundError, "No files"):
            signals.waveform_payload(self.features, "S10", 0)

    @patch("astrostress_signals._cached_subject")
    @patch("astrostress_signals.source_description")
    def test_window_out_of_bounds_rejected(self, source, cached):
        source.return_value={"available":True,"kind":"directory","path":"trusted"}
        cached.return_value=(self.recorded_ecg,self.recorded_eda)
        with self.assertRaises(ValueError):
            signals.waveform_payload(self.features,"S10",2)


if __name__ == "__main__":
    unittest.main()
