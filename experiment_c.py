"""AstroStress Experiment C — held-out prediction reliability.

Run AFTER the verified main analysis:
    python experiment_c.py

Uses the existing features.csv and the exact Random Forest configuration and
fold-local preprocessing from astrostress_v2.py. Refits ONLY the 15 LOSO folds
needed to recover probabilities. Does not alter astrostress_output/ artifacts,
the final saved classifier, the website, or previously reported evaluation.

All confidence thresholds are descriptive, NOT independently validated actions.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

import astrostress_outputs as ao
from astrostress_preprocessing import preprocess_fold
from astrostress_v2 import build_model, get_models

OUTPUT_DIR = Path("astrostress_output")
RESULT_DIR = Path("experiment_c_output")
THRESHOLDS = np.round(np.arange(.50, .951, .05), 2)
COLUMNS = ["run_id", "subject", "window_index", "window_start_s", "window_end_s",
           "y_true", "y_pred", "p_stress", "confidence", "is_correct"]


def collect_held_out_probabilities(frame, feature_cols, model_config, run_id):
    """Use separate held-out subjects with the original fold-local preprocessing."""
    if "subject" not in frame or "label" not in frame:
        raise ValueError("Expected subject and binary label in features.csv")
    rows = []
    subjects = sorted(frame["subject"].astype(str).unique(),
                      key=lambda s: int(s[1:]))
    for subject in subjects:
        train = frame[frame["subject"].astype(str) != subject]
        test = frame[frame["subject"].astype(str) == subject]
        if len(test) == 0 or len(train) == 0 or len(set(train["label"])) < 2:
            raise ValueError(f"Cannot evaluate the held-out subject {subject}")
        X_train, X_test, _, _, _ = preprocess_fold(
            train[feature_cols].to_numpy(), test[feature_cols].to_numpy(),
            feature_cols)
        model = build_model(model_config, train["label"].to_numpy())
        model.fit(X_train, train["label"].to_numpy())
        probs = model.predict_proba(X_test)
        classes = list(model.classes_)
        if 0 not in classes or 1 not in classes:
            raise ValueError(f"Both classes required in training fold {subject}")
        positive = np.asarray(probs[:, classes.index(1)], dtype=float)
        predicted = np.asarray(model.predict(X_test), dtype=int)
        actual = test["label"].to_numpy(dtype=int)
        for window_index, (_, record) in enumerate(test.iterrows()):
            index = len(rows)
            p = float(positive[window_index])
            if not math.isfinite(p) or p < 0 or p > 1:
                raise ValueError(f"Bad probability at {subject}, window {window_index}")
            rows.append({
                "run_id": run_id, "subject": subject,
                "window_index": window_index,
                "window_start_s": float(record["window_start_s"]),
                "window_end_s": float(record["window_end_s"]),
                "y_true": int(actual[window_index]),
                "y_pred": int(predicted[window_index]),
                "p_stress": p,
                "confidence": max(p, 1-p),
                "is_correct": int(predicted[window_index] == actual[window_index]),
            })
        print(f"  {subject}: {len(test)} held-out windows")
    result = pd.DataFrame.from_records(rows, columns=COLUMNS)
    if len(result) != len(frame) or result[["subject", "window_index"]].duplicated().any():
        raise ValueError("Held-out coverage or window indexing mismatch")
    return result


def equal_frequency_calibration(predictions, bins=5):
    """Five near-equal sized bins by rank, robust to tied RF probabilities."""
    ordered = predictions.sort_values("p_stress", kind="stable").reset_index(drop=True)
    n = len(ordered)
    if n < bins:
        raise ValueError("Insufficient held-out predictions for calibration bins")
    groups = np.floor(np.arange(n) * bins / n).astype(int)
    rows = []
    for idx in range(bins):
        block = ordered.iloc[np.where(groups == idx)[0]]
        rows.append({
            "bin": idx+1, "n": len(block),
            "mean_predicted": float(block["p_stress"].mean()),
            "observed_stress_rate": float(block["y_true"].mean()),
        })
    ece = sum(r["n"]/n * abs(r["mean_predicted"]-r["observed_stress_rate"])
              for r in rows)
    return rows, float(ece)


def selective_curve(predictions, thresholds=THRESHOLDS):
    """Descriptive sweep only: no threshold is optimized or claimed validated."""
    output = []
    count = len(predictions)
    for threshold in thresholds:
        accepted = predictions[predictions["confidence"] >= float(threshold)-1e-12]
        coverage = len(accepted)/count if count else 0.0
        output.append({
            "threshold": round(float(threshold), 2),
            "n_accepted": int(len(accepted)),
            "coverage": float(coverage),
            "accuracy": float(accuracy_score(accepted["y_true"], accepted["y_pred"])) if len(accepted) else None,
            "stress_f1": float(f1_score(accepted["y_true"], accepted["y_pred"], zero_division=0)) if len(accepted) else None,
        })
    return output


def non_overlapping_subset(predictions):
    """Greedily select windows that share no raw time samples per participant.

    'Every other row' would be wrong when the feature extractor skips windows.
    """
    chosen = []
    for subject, group in predictions.groupby("subject", sort=False):
        previous_end = -math.inf
        for index, window in group.sort_values(["window_start_s", "window_end_s"]).iterrows():
            start, end = float(window["window_start_s"]), float(window["window_end_s"])
            if not (math.isfinite(start) and math.isfinite(end) and end > start):
                raise ValueError(f"Invalid window interval for {subject}")
            if start >= previous_end-1e-8:
                chosen.append(index)
                previous_end = end
    return predictions.loc[chosen].copy()


def summary_metrics(predictions):
    p = predictions["p_stress"].to_numpy(dtype=float)
    y = predictions["y_true"].to_numpy(dtype=int)
    calibration, ece = equal_frequency_calibration(predictions)
    return {
        "n_windows": int(len(predictions)),
        "n_subjects": int(predictions["subject"].nunique()),
        "pooled_accuracy": float(accuracy_score(y, predictions["y_pred"])),
        "pooled_stress_f1": float(f1_score(y, predictions["y_pred"], zero_division=0)),
        "pooled_roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_equal_frequency_5_bins": ece,
        "calibration_bins": calibration,
        "selective_curve": selective_curve(predictions),
    }


def assert_reproduces_original(predictions, summary):
    """Refuse publication if recovered LOSO predictions disagree with saved run."""
    acc = float(np.mean([accuracy_score(g["y_true"], g["y_pred"])
                         for _, g in predictions.groupby("subject")]))
    f1 = float(np.mean([f1_score(g["y_true"], g["y_pred"], zero_division=0)
                        for _, g in predictions.groupby("subject")]))
    auc = float(roc_auc_score(predictions["y_true"], predictions["p_stress"]))
    expected = {"best_accuracy": acc, "best_f1": f1, "best_auc": auc}
    for key, observed in expected.items():
        reported = float(summary[key])
        if abs(observed-reported) > .00011:  # previous results rounded to 4dp
            raise ValueError(
                f"EXPERIMENT C STOPPED: reproduced {key}={observed:.6f} "
                f"does not match verified main run {reported:.6f}. "
                "Do not use or publish these probabilities until resolved.")
    return {"mean_subject_accuracy": acc, "mean_subject_f1": f1, "pooled_auc": auc}


def main():
    data = ao.load_dashboard_data(str(OUTPUT_DIR))
    if data.get("issues"):
        raise ValueError("Verified main research run is unavailable: "+
                         "; ".join(data["issues"]))
    if any(name not in data for name in ["features", "summary", "comparison", "loso"]):
        raise ValueError("Missing verified main evaluation files")
    summary = data["summary"]
    if summary["best_model"] != "Random Forest":
        raise ValueError("Experiment C only supports the originally selected Random Forest")
    frame = data["features"].copy()
    if "label" not in frame:
        frame = frame.rename(columns={"label_binary": "label"})
    feature_cols = ao.feature_columns(frame)
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    if not {"window_start_s", "window_end_s"}.issubset(frame):
        raise ValueError("Feature timestamps required for the non-overlap check")
    print("Recovering held-out Leave-One-Subject-Out probabilities:")
    predictions = collect_held_out_probabilities(
        frame, feature_cols, get_models()["Random Forest"], summary["run_id"])
    reproduced = assert_reproduces_original(predictions, summary)
    overall = summary_metrics(predictions)
    independent = non_overlapping_subset(predictions)
    conservative = summary_metrics(independent)
    by_subject = []
    for subject, g in predictions.groupby("subject"):
        by_subject.append({
            "subject": subject, "n_windows": len(g),
            "accuracy": float(accuracy_score(g["y_true"], g["y_pred"])),
            "mean_confidence": float(g["confidence"].mean()),
            "wrong_mean_confidence": float(g.loc[g["is_correct"] == 0, "confidence"].mean()) if (g["is_correct"] == 0).any() else None,
        })
    by_subject.sort(key=lambda x: int(x["subject"][1:]))
    result = {
        "experiment": "C — descriptive held-out prediction reliability",
        "main_run_id": summary["run_id"],
        "features_sha256": summary["features_sha256"],
        "model": "Random Forest (same fold-local setup as astrostress_v2)",
        "evaluation": "Leave-One-Subject-Out; all test subjects excluded from training",
        "reproduced_main_metrics": reproduced,
        "full_overlapping_windows": overall,
        "non_overlapping_sensitivity": conservative,
        "by_subject": by_subject,
        "notes": [
            "Brier and ECE use pooled held-out predictions; correlated windows limit precision.",
            "Calibration bins use rank-based equal-sized groups (ties may split across bins).",
            "Threshold sweep is descriptive, not trained or validated for clinical decisions.",
            "Pooled accuracy is not the same as mean per-subject LOSO accuracy.",
            "The model family was chosen on the same LOSO evaluation; results are exploratory.",
            "No astronaut, medical, live monitoring, or validated error-detection claims.",
        ],
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(RESULT_DIR/"experiment_c_predictions.csv", index=False)
    independent.to_csv(RESULT_DIR/"experiment_c_nonoverlap.csv", index=False)
    (RESULT_DIR/"experiment_c_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("\nExperiment C saved separately in", RESULT_DIR)
    print("Main evaluation reproduced:", reproduced)
    print("Held-out Brier:", round(overall["brier_score"], 4))
    print("Full / non-overlap N:", len(predictions), "/", len(independent))
    print("No website or main evaluation files changed.")


if __name__ == "__main__":
    main()
