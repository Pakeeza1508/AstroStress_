"""Fetch a user-approved, SHA256-pinned artifact bundle at Render build time.

Environment: ASTROSTRESS_ARTIFACT_URL (HTTPS), ASTROSTRESS_ARTIFACT_SHA256
Never download or unpickle arbitrary model files without verifying their source.
"""
import hashlib
import io
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "astrostress_output/features.csv",
    "astrostress_output/summary.json",
    "astrostress_output/model_comparison.csv",
    "astrostress_output/loso_results.csv",
    "astrostress_output/personalized_results.csv",
    "astrostress_output/shap_analysis.json",
    "astrostress_output/trained_model.joblib",
    "astrostress_output/window_shap.json",
    "experiment_c_output/experiment_c_summary.json",
}
MAX_BYTES = 12 * 1024 * 1024


def check_sha256(expected):
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise SystemExit("Artifact SHA256 must be exactly 64 hexadecimal digits.")


def install_verified_zip(payload, expected, root=ROOT):
    """Refuse anything but the exact pinned ZIP with exactly the allowed files."""
    check_sha256(expected)
    if len(payload) > MAX_BYTES:
        raise ValueError("Artifact archive is unexpectedly large")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("Artifact SHA256 mismatch — refusing deployment")
    with ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        if names != ALLOWED:
            raise ValueError("Artifact ZIP must contain exactly the expected verified output files")
        for item in archive.infolist():
            if item.is_dir() or item.file_size > MAX_BYTES:
                raise ValueError("Invalid artifact archive member")
        for item in archive.infolist():
            target = Path(root) / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))


def download(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise SystemExit("Artifact URL must use HTTPS without embedded credentials.")
    request = Request(url, headers={"User-Agent": "AstroStress-Artifact-Installer/1.0"})
    with urlopen(request, timeout=45) as response:
        if urlparse(response.url).scheme != "https":
            raise ValueError("Artifact redirect must remain HTTPS")
        return response.read(MAX_BYTES + 1)


def main():
    url = os.environ.get("ASTROSTRESS_ARTIFACT_URL", "").strip()
    expected = os.environ.get("ASTROSTRESS_ARTIFACT_SHA256", "").strip().lower()
    if not url or not expected:
        raise SystemExit("Set ASTROSTRESS_ARTIFACT_URL and ASTROSTRESS_ARTIFACT_SHA256 on Render before deploying.")
    check_sha256(expected)
    install_verified_zip(download(url), expected)
    print("Verified artifact ZIP installed; model provenance rechecked at server startup.")


if __name__ == "__main__":
    main()
