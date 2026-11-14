"""Vercel build hook: only the SHA256-pinned bundle, never a public default URL.

Uses synthetic ZIPs in temporary directories; no WESAD data or model is read.
"""
import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_render_bundle  # noqa: E402
import fetch_vercel_bundle  # noqa: E402

import app  # noqa: E402


def synthetic_zip(names=None):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name in sorted(names or fetch_render_bundle.ALLOWED):
            archive.writestr(name, "synthetic " + name)
    return buffer.getvalue()


class VerifiedZipInstallTests(unittest.TestCase):
    def test_installs_exact_allowed_files_when_hash_matches(self):
        payload = synthetic_zip()
        with tempfile.TemporaryDirectory() as tmp:
            fetch_render_bundle.install_verified_zip(payload, hashlib.sha256(payload).hexdigest(), tmp)
            for name in fetch_render_bundle.ALLOWED:
                self.assertTrue((Path(tmp) / name).is_file(), name)

    def test_rejects_hash_mismatch_without_writing(self):
        payload = synthetic_zip()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fetch_render_bundle.install_verified_zip(payload, "0" * 64, tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_rejects_unexpected_members_without_writing(self):
        payload = synthetic_zip(set(fetch_render_bundle.ALLOWED) | {"extra/secret.txt"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fetch_render_bundle.install_verified_zip(payload, hashlib.sha256(payload).hexdigest(), tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class VercelBuildHookTests(unittest.TestCase):
    def test_no_public_default_url(self):
        source = (ROOT / "scripts" / "fetch_vercel_bundle.py").read_text(encoding="utf-8")
        self.assertNotIn("releases/download", source)
        self.assertNotIn("https://", source)

    def test_fails_loudly_without_any_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                fetch_vercel_bundle.main(tmp, environ={})

    def test_local_bundle_must_match_pinned_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / fetch_vercel_bundle.LOCAL_BUNDLE).write_bytes(synthetic_zip())
            # An environment-provided hash must not override the pinned one.
            env = {"ASTROSTRESS_ARTIFACT_SHA256": hashlib.sha256(synthetic_zip()).hexdigest()}
            with self.assertRaises(ValueError):
                fetch_vercel_bundle.main(tmp, environ=env)
            self.assertFalse((Path(tmp) / "astrostress_output").exists())

    def test_private_url_is_checked_against_pinned_hash(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(fetch_vercel_bundle, "download", return_value=synthetic_zip()) as fake:
            with self.assertRaises(ValueError):
                fetch_vercel_bundle.main(tmp, environ={"ASTROSTRESS_ARTIFACT_URL": "https://example.invalid/b.zip"})
            fake.assert_called_once()

    def test_function_bundle_excludes_raw_zip(self):
        self.assertIn("astrostress-render-artifacts.zip", (ROOT / "vercel.json").read_text(encoding="utf-8"))


class StatusFlagTests(unittest.TestCase):
    def test_status_reports_window_records_restricted_by_default(self):
        context = ({"summary": {"run_id": "run-x"}}, {"model_name": "Random Forest"})
        client = app.app.test_client()
        with patch.dict(os.environ, {}, clear=True), patch.object(app, "verified_context", return_value=context):
            payload = client.get("/api/status").get_json()
        self.assertIs(payload["window_records_public"], False)
        with patch.dict(os.environ, {"ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS": "1"}), \
                patch.object(app, "verified_context", return_value=context):
            payload = client.get("/api/status").get_json()
        self.assertIs(payload["window_records_public"], True)


if __name__ == "__main__":
    unittest.main()
