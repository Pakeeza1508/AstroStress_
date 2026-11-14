"""Precompute verified per-window SHAP values for a lightweight public demo.

Run locally after the verified model and pipeline outputs exist:
    python scripts/build_window_shap.py
The output is an in-sample explanation of saved WESAD predictions, NOT LOSO.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import shap

from astrostress_inference import load_verified_model
from astrostress_outputs import load_dashboard_data

OUTPUT = ROOT / "astrostress_output"


def main():
    data = load_dashboard_data(str(OUTPUT))
    if data.get("issues") or not data.get("summary") or "features" not in data:
        raise SystemExit("Missing or unverified pipeline outputs: " + str(data.get("issues")))
    summary = data["summary"]
    artifact = load_verified_model(OUTPUT, summary)
    frame = data["features"]
    X = artifact["preprocessor"].transform(frame[artifact["feature_cols"]].to_numpy(dtype=float))
    sv = shap.TreeExplainer(artifact["model"]).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    names = list(artifact["selected_features"])
    if sv.shape != (len(frame), len(names)) or not np.isfinite(sv).all():
        raise ValueError("SHAP output does not match verified model features")
    by_subject = {}
    for subject, row in zip(frame["subject"].astype(str), sv):
        rows = sorted(({"feature": key, "value": float(value)}
                       for key, value in zip(names, row)),
                      key=lambda pair: abs(pair["value"]), reverse=True)
        by_subject.setdefault(subject, []).append(rows[:10])
    output = {
        "run_id": summary["run_id"],
        "features_sha256": summary["features_sha256"],
        "model": artifact["model_name"],
        "evaluation_type": "in_sample_final_model_explanation",
        "n_windows": len(frame),
        "by_subject": by_subject,
    }
    filename = OUTPUT / "window_shap.json"
    filename.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {filename}; {len(frame)} verified in-sample SHAP explanations")


if __name__ == "__main__":
    main()
