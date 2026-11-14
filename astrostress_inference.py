"""Verified, offline inference on recorded WESAD feature windows.

The saved final model is trained on *all* WESAD windows. Predictions made on
rows from features.csv are therefore IN-SAMPLE demonstrations, not held-out
LOSO estimates. Never present them as prospective accuracy or astronaut data.

Only load joblib files generated locally by this project's trusted pipeline:
joblib/pickle deserialization can execute arbitrary code.
"""

from pathlib import Path
import joblib
import pandas as pd

from astrostress_outputs import MODEL_FILE, FEATURES_FILE, file_sha256
from astrostress_preprocessing import predict_stress_proba


class ModelArtifactError(ValueError):
    """Saved artifact missing, stale, or inconsistent with current evaluation."""


def load_verified_model(output_dir, summary):
    """Load the trusted model only when its run, features, and type match summary.

    The caller must first obtain summary from load_dashboard_data(), which checks
    the evaluation schema and the features.csv content hash.
    """
    if not summary or not summary.get("run_id"):
        raise ModelArtifactError("No verified evaluation summary is available.")
    out = Path(output_dir)
    model_path = out / MODEL_FILE
    features_path = out / FEATURES_FILE
    if not model_path.is_file():
        raise ModelArtifactError("Trained model is missing. Run python astrostress_v2.py locally.")
    if not features_path.is_file():
        raise ModelArtifactError("features.csv is missing.")
    features_hash = file_sha256(features_path)
    if not summary.get("features_sha256") or summary["features_sha256"] != features_hash:
        raise ModelArtifactError("The model's source features do not match the verified summary.")

    # IMPORTANT: load only this project's locally generated trusted joblib file.
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ModelArtifactError("Unexpected saved-model artifact format.")
    for key in ("run_id", "features_sha256", "model_name", "feature_cols",
                "selected_features", "preprocessor", "model"):
        if key not in artifact:
            raise ModelArtifactError(f"Saved-model artifact lacks {key}.")
    if artifact["run_id"] != summary["run_id"]:
        raise ModelArtifactError("Saved model belongs to a different analysis run.")
    if artifact["features_sha256"] != features_hash:
        raise ModelArtifactError("Saved model belongs to a different features.csv.")
    if artifact["model_name"] != summary["best_model"]:
        raise ModelArtifactError("Saved model type does not match the selected model.")
    if len(artifact["feature_cols"]) == 0 or len(artifact["selected_features"]) == 0:
        raise ModelArtifactError("Saved model has no usable features.")
    if list(artifact["selected_features"]) != list(artifact["preprocessor"].selected_features_):
        raise ModelArtifactError("Saved model selected-feature order is inconsistent.")
    return artifact


def predict_recorded_window(artifact, window):
    """Return (stress_probability, predicted_label) for exactly one feature row.

    No fallback values are created. Input features must be present by NAME.
    """
    if not isinstance(window, pd.DataFrame) or len(window) != 1:
        raise ValueError("Provide exactly one recorded window as a DataFrame.")
    probability = float(predict_stress_proba(artifact, window)[0])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("The model returned an invalid stress probability.")
    return probability, int(probability >= 0.5)


def explain_recorded_window(artifact, window, top_n=5):
    """Return approximate per-feature SHAP contributions for one recorded row.

    Explains the *in-sample saved model*, not causal physiology. Units depend on
    the tree-model family. Uses the same SHAP output normalizer as the pipeline.
    """
    # Keep deployment inference independent of training-only model families.
    import numpy as np
    import shap

    def compute_shap_values(model, X):
        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        if sv.shape != X.shape:
            raise ValueError(f"SHAP output shape {sv.shape} does not match input {X.shape}")
        return sv

    cols = artifact["feature_cols"]
    missing = [name for name in cols if name not in window.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    X = artifact["preprocessor"].transform(window[cols].to_numpy(dtype=float))
    sv = compute_shap_values(artifact["model"], X)
    names = list(artifact["selected_features"])
    if sv.shape != (1, len(names)):
        raise ValueError("SHAP contributions do not align with model features.")
    pairs = sorted(zip(names, (float(v) for v in sv[0])), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:max(1, int(top_n))]
