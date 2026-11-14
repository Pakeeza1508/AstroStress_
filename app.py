"""Vercel Flask adapter for the original AstroStress demo.

The existing index.html and app_server.py remain unchanged on this branch.
Only recorded WESAD windows and verified precomputed SHAP data are served.
"""
import logging
import os
from functools import lru_cache

from flask import Flask, jsonify, request, send_file
from app_server import (
    ROOT, get_context, global_shap_payload, prerecorded_shap_payload,
    reliability_payload, research_payload, subjects_payload, window_payload,
    physiology_trend_payload,
)

app = Flask(__name__, static_folder=None)


def public_window_access_enabled():
    """Only an explicit opt-in allows distribution of WESAD-derived windows."""
    return os.environ.get("ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS") == "1"


def restricted_response():
    return jsonify(error="Individual WESAD-derived records are not publicly available.",
                   code="RESTRICTED_DATA"), 403


@lru_cache(maxsize=1)
def verified_context():
    """Load the small SHA256-pinned research/model bundle once per warm worker."""
    return get_context()


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/")
@app.get("/index.html")
def dashboard():
    return send_file(ROOT / "index.html", mimetype="text/html")


@app.get("/healthz")
def health():
    try:
        verified_context()
        return jsonify(ok=True, service="astrostress-api")
    except Exception:
        app.logger.exception("Verified model is unavailable")
        return jsonify(ok=False, service="astrostress-api"), 503


@app.get("/api/<endpoint>")
def api(endpoint):
    try:
        if endpoint in ("window", "physiology-trend", "explain") and not public_window_access_enabled():
            return restricted_response()
        data, artifact = verified_context()

        if endpoint == "status":
            return jsonify(ok=True, run_id=data["summary"]["run_id"],
                           model=artifact["model_name"],
                           window_records_public=public_window_access_enabled())
        if endpoint == "research":
            return jsonify(research_payload(data))
        if endpoint == "global-shap":
            return jsonify(global_shap_payload(data))
        if endpoint == "reliability":
            return jsonify(reliability_payload(data))
        if endpoint == "physiology-trend":
            return jsonify(physiology_trend_payload(data, request.args.get("subject", "")))
        if endpoint == "subjects":
            return jsonify(subjects=subjects_payload(data))
        if endpoint == "signal-source":
            return jsonify(available=False, kind=None,
                           note="Raw WESAD waveforms are not available in the public demo.")
        if endpoint == "signals":
            return jsonify(error="Raw WESAD waveforms are not enabled on this public demo.",
                           code="SOURCE_UNAVAILABLE"), 404

        if endpoint in ("window", "explain"):
            subject = request.args.get("subject", "")
            index = int(request.args.get("index", "0"))
            if endpoint == "window":
                return jsonify(window_payload(data, artifact, subject, index))
            # Validate the selected window before returning its prerecorded SHAP.
            window_payload(data, artifact, subject, index)
            return jsonify(prerecorded_shap_payload(data, subject, index))

        return jsonify(error="Not found"), 404
    except FileNotFoundError as exc:
        return jsonify(error=str(exc), code="SOURCE_UNAVAILABLE"), 404
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("AstroStress API request failed")
        return jsonify(error="The model request failed; see server logs."), 500
