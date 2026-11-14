"""AstroStress verified research demo API — standard-library HTTP, no JS build step.

Run: python app_server.py
Visit: http://127.0.0.1:8000/

The default mode binds localhost only. Set ASTROSTRESS_PUBLIC=1 for trusted
public deployments with published, provenance-checked, prebuilt artifacts.
Only load joblib artifacts generated locally by the trusted pipeline.
"""

import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from astrostress_inference import load_verified_model, predict_recorded_window, explain_recorded_window
from astrostress_outputs import load_dashboard_data
from astrostress_signals import source_description, waveform_payload

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("ASTROSTRESS_OUTPUT_DIR", ROOT / "astrostress_output"))
PORT = int(os.environ.get("PORT", os.environ.get("ASTROSTRESS_PORT", "8000")))
PUBLIC_MODE = os.environ.get("ASTROSTRESS_PUBLIC", "0") == "1"
HOST = "0.0.0.0" if PUBLIC_MODE else "127.0.0.1"


def get_context(output_dir=OUTPUT_DIR):
    """Refuse serving predictions from missing, stale or mixed analysis runs."""
    data = load_dashboard_data(str(output_dir))
    required = ("features", "summary", "comparison", "loso", "personalized")
    missing = [key for key in required if key not in data]
    if data.get("issues") or missing:
        raise ValueError("Research outputs are not fully verified: " +
                         "; ".join(data.get("issues", []) + [f"Missing {k}" for k in missing]))
    artifact = load_verified_model(output_dir, data["summary"])
    return data, artifact


def reliability_payload(data, report_path=None):
    """Read Experiment C only when tied to the verified original research run.

    The experiment remains separate and optional; never silently replace the
    main model's published metrics with filtered (selective) accuracy.
    """
    path = Path(report_path) if report_path else ROOT / "experiment_c_output" / "experiment_c_summary.json"
    if not path.is_file():
        raise FileNotFoundError("Experiment C results not found. Run python experiment_c.py first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    original = data["summary"]
    if (report.get("main_run_id") != original["run_id"]
            or report.get("features_sha256") != original["features_sha256"]):
        raise ValueError("Experiment C results do not belong to the current verified research run")
    full = report.get("full_overlapping_windows", {})
    nonoverlap = report.get("non_overlapping_sensitivity", {})
    if (full.get("n_windows") != int(original["n_windows"])
            or full.get("n_subjects") != int(original["n_subjects"])
            or not isinstance(full.get("calibration_bins"), list)
            or len(full["calibration_bins"]) != 5
            or not isinstance(full.get("selective_curve"), list)
            or len(full["selective_curve"]) != 10
            or nonoverlap.get("n_windows", 0) <= 0):
        raise ValueError("Experiment C report is incomplete or has a different sample count")
    first = full["selective_curve"][0]
    if abs(first["coverage"] - 1.0) > 1e-8 or first["n_accepted"] != full["n_windows"]:
        raise ValueError("Experiment C confidence sweep is incomplete")
    for row in full["selective_curve"]:
        if not (0 <= row["coverage"] <= 1 and 0 <= row["n_accepted"] <= full["n_windows"]):
            raise ValueError("Invalid Experiment C coverage")
    return {
        "run_id": original["run_id"],
        "source": "held-out Leave-One-Subject-Out evaluation (not saved-model demo replay)",
        "n_windows": full["n_windows"],
        "nonoverlap_windows": nonoverlap["n_windows"],
        "brier": full["brier_score"],
        "ece": full["ece_equal_frequency_5_bins"],
        "pooled_accuracy": full["pooled_accuracy"],
        "calibration_bins": full["calibration_bins"],
        "selective_curve": full["selective_curve"],
        "nonoverlap_curve": nonoverlap["selective_curve"],
        "note": "Thresholds are descriptive, not independently validated. Selection of the model family on the same evaluation and correlated windows limit conclusions.",
    }


def prerecorded_shap_payload(data, subject, index, path=None):
    """Precomputed SHAP values tied to the trusted model and original feature hash."""
    archive = Path(path) if path is not None else OUTPUT_DIR / "window_shap.json"
    if not archive.is_file():
        raise FileNotFoundError("Precomputed SHAP explanations missing. Run python scripts/build_window_shap.py locally.")
    payload = json.loads(archive.read_text(encoding="utf-8"))
    summary = data["summary"]
    if (payload.get("run_id") != summary["run_id"]
            or payload.get("features_sha256") != summary["features_sha256"]
            or payload.get("model") != summary["best_model"]
            or payload.get("n_windows") != int(summary["n_windows"])):
        raise ValueError("Precomputed SHAP values do not match verified model run")
    rows = payload.get("by_subject", {}).get(subject, [])
    if index < 0 or index >= len(rows):
        raise ValueError("SHAP recording index out of range")
    contributions = rows[index]
    if (not isinstance(contributions, list) or not contributions
            or any(not isinstance(v.get("feature"), str)
                   or not isinstance(v.get("value"), (int, float))
                   or not math.isfinite(float(v["value"]))
                   for v in contributions)):
        raise ValueError("Precomputed SHAP values invalid")
    return {"run_id": summary["run_id"], "subject": subject,
            "window_index": index, "contributions": contributions,
            "note": "Verified in-sample model attribution, not a causal explanation."}


def subjects_payload(data):
    """Expose non-identifying participant IDs and their window counts."""
    features = data["features"]
    return [{"id": str(s), "windows": int(n)}
            for s, n in sorted(features.groupby("subject").size().items(),
                               key=lambda pair: int("".join(c for c in str(pair[0]) if c.isdigit()) or 0))]



def physiology_trend_payload(data, subject):
    """Derived laboratory features across one participant's recorded windows.

    These window summaries are not original ECG/EDA waveforms and must never
    be plotted as raw sensor samples. Uses the already verified features.csv.
    """
    features = data["features"]
    selected = features[features["subject"].astype(str) == subject]
    if selected.empty:
        raise ValueError("Unknown WESAD participant")

    def finite_feature(row, name):
        if name not in row:
            return None
        value = pd.to_numeric(row[name], errors="coerce")
        return float(value) if pd.notna(value) and math.isfinite(float(value)) else None

    windows = [
        {
            "index": int(i),
            "window_start_s": finite_feature(row, "window_start_s"),
            "HR_mean": finite_feature(row, "HR_mean"),
            "EDA_mean": finite_feature(row, "EDA_mean"),
        }
        for i, (_, row) in enumerate(selected.iterrows())
    ]
    return {
        "run_id": data["summary"]["run_id"],
        "subject": subject,
        "source": "verified WESAD processed physiological features",
        "is_raw_waveform": False,
        "window_count": len(windows),
        "units": {"HR_mean": "BPM", "EDA_mean": "µS"},
        "windows": windows,
        "note": "Derived mean heart rate and skin conductance per recorded window, not raw ECG/EDA, live monitoring, or astronaut telemetry. Windows may overlap.",
    }

def window_payload(data, artifact, subject, window_index):
    """One indexed, recorded feature window; index is per participant, zero-based."""
    if not isinstance(window_index, int) or window_index < 0:
        raise ValueError("Window index must be a nonnegative integer")
    features = data["features"]
    if subject not in set(features["subject"].astype(str)):
        raise ValueError("Unknown WESAD participant")
    selected = features[features["subject"].astype(str) == subject]
    if window_index >= len(selected):
        raise ValueError("Window index outside available recorded windows")
    row = selected.iloc[[window_index]]
    probability, predicted = predict_recorded_window(artifact, row)
    label_col = "label" if "label" in row else "label_binary"
    actual = int(row[label_col].iloc[0])
    feature_values = {}
    for col in artifact["feature_cols"]:
        val = pd.to_numeric(row[col], errors="coerce").iloc[0]
        feature_values[col] = float(val) if pd.notna(val) and math.isfinite(float(val)) else None
    payload = {
        "run_id": data["summary"]["run_id"],
        "source": "WESAD laboratory recording",
        "evaluation_type": "in_sample_final_model_replay",
        "is_live": False,
        "subject": subject, "window_index": window_index,
        "window_count": int(len(selected)),
        "prediction": {"stress_probability": probability, "class": predicted,
                       "actual_class": actual, "matches_protocol_label": bool(predicted == actual)},
        "model": artifact["model_name"],
        "features": feature_values,
        "note": "The saved model was fitted on all WESAD windows. This is an in-sample replay, not held-out validation or astronaut monitoring.",
    }
    for column in ("window_start_s", "window_end_s"):
        if column in row:
            x = row[column].iloc[0]
            payload[column] = float(x) if pd.notna(x) else None
    return payload


def clean_table(frame, columns):
    """JSON-safe subset of a verified CSV, including missing measurements."""
    records = frame[columns].to_dict(orient="records")
    for row in records:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif hasattr(value, "item"):
                row[key] = value.item()
    return records


def global_shap_payload(data):
    """Verified in-sample global SHAP importance for the final fitted model."""
    shap_data = data.get("shap")
    summary = data["summary"]
    if not shap_data:
        raise FileNotFoundError("Verified global SHAP results unavailable. Run python astrostress_v2.py to generate them.")
    if shap_data.get("run_id") != summary["run_id"]:
        raise ValueError("SHAP analysis run does not match the verified research run")
    if shap_data.get("model_used") != summary["best_model"]:
        raise ValueError("SHAP analysis model differs from the selected research model")
    importance = shap_data.get("global_importance")
    if not isinstance(importance, dict) or not importance:
        raise ValueError("Global SHAP importance is missing or invalid")
    rows = []
    for name, raw in importance.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid SHAP feature name")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError("Invalid SHAP importance value")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid SHAP importance value")
        rows.append({"feature": name, "importance": value})
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return {
        "run_id": summary["run_id"],
        "model": shap_data["model_used"],
        "n_windows_explained": int(shap_data["n_windows_explained"]),
        "value_units": shap_data["value_units"],
        "features": rows,
        "evaluation_type": "in_sample_final_model_explanation",
        "note": "Mean absolute SHAP attribution from the final model refitted on WESAD. These values describe model behaviour, not medical causes or held-out performance.",
    }


def research_payload(data):
    s = data["summary"]
    return {
        "run_id": s["run_id"], "best_model": s["best_model"],
        "accuracy": float(s["best_accuracy"]), "f1": float(s["best_f1"]),
        "auc": float(s["best_auc"]), "n_subjects": int(s["n_subjects"]),
        "n_windows": int(s["n_windows"]), "n_features": int(s["n_candidate_features"]),
        "comparison": clean_table(data["comparison"], ["model", "accuracy", "f1", "auc"]),
        "loso": clean_table(data["loso"], ["subject", "accuracy", "f1", "n_test"]),
        "personalized": clean_table(data["personalized"], ["subject", "generic_accuracy", "personal_accuracy", "improvement_pp"]),
        "source": "Verified corrected WESAD laboratory run",
    }


class Handler(BaseHTTPRequestHandler):
    context = None

    def _json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/healthz":
            self._json({"ok": self.context is not None, "service": "astrostress-api"})
            return
        if url.path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not url.path.startswith("/api/"):
            self._json({"error": "Not found"}, 404)
            return
        try:
            data, artifact = self.context
            if url.path == "/api/status":
                self._json({"ok": True, "run_id": data["summary"]["run_id"],
                            "model": artifact["model_name"]})
            elif url.path == "/api/research":
                self._json(research_payload(data))
            elif url.path == "/api/global-shap":
                self._json(global_shap_payload(data))
            elif url.path == "/api/reliability":
                self._json(reliability_payload(data))
            elif url.path == "/api/subjects":
                self._json({"subjects": subjects_payload(data)})
            elif url.path == "/api/signal-source":
                source = source_description()
                self._json({"available": source["available"], "kind": source["kind"], "note": source.get("note", "")})
            elif url.path == "/api/signals":
                if PUBLIC_MODE:
                    raise FileNotFoundError("Raw WESAD waveforms are not enabled on this public demo. Local-only recordings are required.")
                q = parse_qs(url.query)
                subject = q.get("subject", [""])[0]
                index = int(q.get("index", ["0"])[0])
                payload = waveform_payload(data["features"], subject, index)
                payload["run_id"] = data["summary"]["run_id"]
                self._json(payload)
            elif url.path == "/api/explain":
                q = parse_qs(url.query)
                subject = q.get("subject", [""])[0]
                index = int(q.get("index", ["0"])[0])
                payload = window_payload(data, artifact, subject, index)
                if PUBLIC_MODE:
                    self._json(prerecorded_shap_payload(data, subject, index))
                else:
                    rows = data["features"][data["features"]["subject"].astype(str) == subject]
                    shap = explain_recorded_window(artifact, rows.iloc[[index]], top_n=10)
                    self._json({"run_id": payload["run_id"], "subject": subject,
                                "window_index": index, "contributions":
                                [{"feature": name, "value": value} for name, value in shap],
                                "note": "In-sample model attribution, not a causal explanation."})
            elif url.path == "/api/physiology-trend":
                q = parse_qs(url.query)
                subject = q.get("subject", [""])[0]
                self._json(physiology_trend_payload(data, subject))
            elif url.path == "/api/window":
                q = parse_qs(url.query)
                subject = q.get("subject", [""])[0]
                index = int(q.get("index", ["0"])[0])
                self._json(window_payload(data, artifact, subject, index))
            else:
                self._json({"error": "Not found"}, 404)
        except FileNotFoundError as exc:
            self._json({"error": str(exc), "code": "SOURCE_UNAVAILABLE"}, 404)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception:
            import traceback
            traceback.print_exc()
            self._json({"error": "The model request failed; see server logs."}, 500)


def main():
    print(f"Loading locally verified outputs from: {OUTPUT_DIR}")
    Handler.context = get_context(OUTPUT_DIR)
    print("Model and research artifacts verified.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"AstroStress listening on {HOST}:{PORT} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
