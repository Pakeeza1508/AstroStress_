"""Check public MVP fallback remains interactive without WESAD record access."""
import unittest
from pathlib import Path


class PublicMvpUXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (Path(__file__).resolve().parents[1] / "index.html").read_text(encoding="utf-8")

    def test_public_mode_has_aggregate_participant_explorer(self):
        for marker in (
            "function renderPublicResearchMode()",
            "id = 'public-ai-subject'",
            "researchFromServer?.loso",
            "select.addEventListener('change', update)",
            "renderPublicResearchMode();",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.html)

    def test_manual_daily_checkin_stays_enabled(self):
        for marker in ('id="health-stress" required', 'id="health-sleep" required',
                       'id="health-feeling" required',
                       "healthForm.addEventListener('submit'", 
                       "formTitle.textContent = '1. Your self-check"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.html)
        self.assertIn("if (recordedCard) recordedCard.style.display = 'none';", self.html)

    def test_no_public_window_access_reenabled(self):
        self.assertIn("if (!windowRecordsPublic) applyRestrictedWindowMode(status);", self.html)
        self.assertIn("Individual WESAD-derived recordings are not published", self.html)
        self.assertNotIn("ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS = '1'", self.html)


if __name__ == "__main__":
    unittest.main()
