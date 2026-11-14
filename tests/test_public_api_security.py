"""Regression coverage: public routes do not redistribute WESAD window records.

No WESAD, model, or NASA data is loaded by these access-control checks.
"""
import os
import unittest
from unittest.mock import patch

import app
import app_server


class PublicWindowAccessTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_flask_denies_sensitive_routes_without_artifacts(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(app, "verified_context", side_effect=AssertionError("must not load artifacts")):
            for path in (
                "/api/window?subject=S2&index=0",
                "/api/explain?subject=S2&index=0",
                "/api/physiology-trend?subject=S2",
            ):
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.get_json()["code"], "RESTRICTED_DATA")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_flask_deny_is_default_even_when_env_is_zero(self):
        with patch.dict(os.environ, {"ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS": "0"}):
            self.assertFalse(app.public_window_access_enabled())

    def test_explicit_opt_in_is_required(self):
        with patch.dict(os.environ, {"ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS": "1"}):
            self.assertTrue(app.public_window_access_enabled())

    def test_standalone_public_server_denies_sensitive_routes(self):
        for path in (
            "/api/window?subject=S2&index=0",
            "/api/explain?subject=S2&index=0",
            "/api/physiology-trend?subject=S2",
        ):
            with self.subTest(path=path):
                handler = object.__new__(app_server.Handler)
                handler.path = path
                results = []
                handler._json = lambda payload, status=200: results.append((status, payload))
                handler.context = None
                with patch.object(app_server, "PUBLIC_MODE", True), patch.object(app_server, "ALLOW_PUBLIC_WESAD_WINDOWS", False):
                    handler.do_GET()
                self.assertEqual(results[0][0], 403)
                self.assertEqual(results[0][1]["code"], "RESTRICTED_DATA")


if __name__ == "__main__":
    unittest.main()
