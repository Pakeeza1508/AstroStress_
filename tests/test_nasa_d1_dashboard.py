"""Public NASA D1 dashboard is descriptive and uses SHA-pinned aggregate data only."""
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NasaD1DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.source = json.loads((ROOT / "docs/research/nasa_d1_public_summary.json").read_text(encoding="utf-8"))
        match = re.search(r'<script id="nasa-d1-data" type="application/json">(.*?)</script>', cls.html, re.S)
        assert match, "Missing embedded public aggregate data"
        cls.embedded = json.loads(match.group(1))

    def test_embedding_matches_source_and_is_nonmedical(self):
        self.assertEqual(self.embedded, self.source)
        self.assertFalse(self.embedded["is_astronaut_validation"])
        self.assertFalse(self.embedded["is_health_assessment"])
        self.assertEqual(self.embedded["exploratory_co2_threshold_ppm"], 4000)

    def test_original_telemetry_summary_and_hashes(self):
        expected = {
            "RR-1": (12087, 993, 996, 2, 6, 33, "6f252984805e8e9a1e7d19927f88b70160ebc30fa3cd7e842572f2b806e5a805"),
            "RR-6": (17572, 95, 96, 2, 12, 1392, "7c8d8f9c2c61856e749c3ac64abb3cdbaabf4f35254c2d5198e1c202e5e0f68f"),
        }
        for mission, values in expected.items():
            with self.subTest(mission=mission):
                r = self.embedded["missions"][mission]
                self.assertEqual(tuple(r[k] for k in (
                    "rows", "co2_above_4000_rows", "co2_at_or_above_4000_rows",
                    "gaps_over_10m", "short_intervals_under_5m",
                    "largest_gap_minutes", "sha256")), values)
                self.assertTrue(r["weekly_medians"])
                self.assertTrue(all(len(item) == 4 for item in r["weekly_medians"]))

    def test_rr9_is_summary_only(self):
        rr9 = self.embedded["missions"]["RR-9"]
        self.assertEqual(rr9["rows"], 10)
        self.assertFalse(rr9["timeline_available"])
        self.assertNotIn("weekly_medians", rr9)

    def test_interactive_navigation_and_no_api_export(self):
        for expected in ('data-page="nasa"', 'id="page-nasa"', 'id="nasa-mission-select"',
                         'id="nasa-metric-select"', 'id="chart-nasa-d1"',
                         "if (page === 'nasa') initNasaD1Chart();",
                         "document.getElementById('nasa-mission-select').addEventListener('change',initNasaD1Chart);",
                         "document.getElementById('nasa-metric-select').addEventListener('change',initNasaD1Chart);"):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.html)
        self.assertNotIn("window_start_s", json.dumps(self.embedded))
        self.assertNotIn("window_end_s", json.dumps(self.embedded))
        self.assertIn("Individual WESAD-derived recordings are not published", self.html)


if __name__ == "__main__":
    unittest.main()
