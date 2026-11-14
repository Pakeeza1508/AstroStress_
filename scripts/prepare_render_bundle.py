"""Build a small, explicit ZIP of trusted local outputs for Render deployment.

Usage: python scripts/prepare_render_bundle.py
Only share the bundle where WESAD dataset terms allow it. Never add the
original WESAD raw recordings or untrusted model files.
"""
from pathlib import Path
from hashlib import sha256
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "astrostress_output"
EXPERIMENT = ROOT / "experiment_c_output"
FILES = (
    "features.csv", "summary.json", "model_comparison.csv",
    "loso_results.csv", "personalized_results.csv",
    "shap_analysis.json", "trained_model.joblib", "window_shap.json",
)


def main():
    missing = [name for name in FILES if not (MAIN / name).is_file()]
    if not (EXPERIMENT / "experiment_c_summary.json").is_file():
        missing.append("experiment_c_output/experiment_c_summary.json")
    if missing:
        raise SystemExit("Missing verified outputs: " + ", ".join(missing))
    destination = ROOT / "astrostress-render-artifacts.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=7) as archive:
        for name in FILES:
            archive.write(MAIN / name, arcname="astrostress_output/" + name)
        archive.write(EXPERIMENT / "experiment_c_summary.json",
                      arcname="experiment_c_output/experiment_c_summary.json")
    digest = sha256(destination.read_bytes()).hexdigest()
    print(f"Artifact bundle: {destination} ({destination.stat().st_size} bytes)")
    print(f"SHA256: {digest}")
    print("Keep the ZIP out of Git. Upload only to storage whose sharing and WESAD terms you approve.")


if __name__ == "__main__":
    main()
