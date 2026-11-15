"""Static link/navigation checks for the copy-first Overview and public-safe UI."""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


HTML = (Path(__file__).resolve().parents[1] / "index.html").read_text(encoding="utf-8")


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = []
        self.links = []
        self.pages = set()
        self.nav = set()
        self.sections = {}
    def handle_starttag(self, tag, attrs):
        props = dict(attrs)
        if props.get("id", "").startswith("page-"):
            self.pages.add(props["id"][5:])
        if tag == "button":
            self.buttons.append(props)
            if "nav-item" in props.get("class", "").split():
                self.nav.add(props.get("data-page"))
        if tag == "a":
            self.links.append(props)
        if tag == "section":
            self.sections[props.get("class", "")] = props


class OverviewLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = Controls()
        cls.doc.feed(HTML)

    def test_all_navigational_buttons_resolve_to_visible_pages(self):
        self.assertEqual(self.doc.nav, {"overview", "physiology", "analysis", "checkin", "explainability", "results", "nasa"})
        for control in self.doc.buttons:
            target = control.get("data-go")
            if target:
                with self.subTest(target=target):
                    self.assertIn(target, self.doc.pages)
                    self.assertIn(target, self.doc.nav)

    def test_overview_results_and_nasa_buttons_are_visible(self):
        intro = HTML.split('<div class="page active" id="page-overview">', 1)[1].split('<!-- Daily Health Check-in:', 1)[0]
        actions = re.search(r'<div class="mission-actions"([^>]*)>(.*?)</div>', intro, re.S)
        self.assertIsNotNone(actions)
        self.assertNotIn("hidden", actions.group(1))
        self.assertIn('data-go="results"', actions.group(2))
        self.assertIn('data-go="nasa"', actions.group(2))
        self.assertIn("document.querySelectorAll('[data-go]').forEach", HTML)
        self.assertIn("destination.click()", HTML)

    def test_inactive_hero_prediction_card_is_hidden(self):
        self.assertRegex(HTML, r'<section class="overview-demo"[^>]*hidden[^>]*style="display:none"')
        self.assertIn('id="overview-demo-run"', HTML)  # Keep API callback compatible.
        self.assertIn("What if astronauts' stress could be", HTML)

    def test_overview_return_restores_previous_scroll_position(self):
        self.assertIn("const pageScrollPositions = new Map", HTML)
        self.assertIn("pageScrollPositions.set(currentPage, window.scrollY)", HTML)
        self.assertIn("pageScrollPositions.get('overview')", HTML)
        self.assertIn("window.scrollTo({ top: restoreY, behavior: 'instant' })", HTML)
        self.assertNotIn("window.scrollTo({ top: 0, behavior: 'smooth' })", HTML)

    def test_datasets_used_sources_and_public_mode(self):
        self.assertIn('data-page="physiology">Datasets Used</button>', HTML)
        self.assertIn('id="page-physiology"', HTML)
        self.assertIn('Prediction Confidence &amp; Reliability', HTML)
        self.assertIn('NASA Environmental Telemetry Audit', HTML)
        self.assertEqual(HTML.count('class="card dataset-source-card"'), 3)
        self.assertNotIn('If WESAD is already downloaded on your computer', HTML)
        self.assertNotIn('Why are individual ECG and EDA traces unavailable?', HTML)
        self.assertIn("for (const el of [physSelector, physChart, physTables])", HTML)

    def test_public_access_disables_individual_predictions(self):
        self.assertIn("function applyRestrictedWindowMode(status)", HTML)
        self.assertIn("renderPublicResearchMode();", HTML)
        self.assertIn("const recordedCard = byId('checkin-example-button')?.closest('.card');", HTML)
        self.assertIn("if (recordedCard) recordedCard.style.display = 'none';", HTML)

    def test_external_links_do_not_use_unverified_local_paths(self):
        self.assertEqual(len(self.doc.links), 2)
        for anchor in self.doc.links:
            with self.subTest(href=anchor.get("href")):
                self.assertTrue(anchor.get("href", "").startswith("https://"))
                self.assertIn("noopener", anchor.get("rel", ""))
        self.assertIn("https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html", HTML)
        self.assertIn("https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx", HTML)


if __name__ == "__main__":
    unittest.main()
