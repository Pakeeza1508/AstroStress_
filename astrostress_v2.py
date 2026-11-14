"""
AstroStress — Corrected Pipeline
NASA Space Apps Challenge 2026

What it does:
  1. Leave-One-Subject-Out (LOSO) comparison of 4 model families; imputation,
     feature selection and scaling are fitted inside each fold on training
     subjects only (astrostress_preprocessing.py)
  2. Fair personalization: generic vs personalized on identical evaluation
     windows; evaluation windows overlapping calibration windows are purged
  3. Final model = the model family with the highest mean LOSO F1, refitted
     on all windows and saved with its preprocessing and feature schema
     (trained_model.joblib). It is the best model under this procedure, not
     an independently validated production model.
  4. SHAP on that final model (in-sample; explains model behaviour, not
     physiological causation)
  5. Isolation Forest anomaly flags; the flagged fraction is set by the
     `contamination` parameter, and flags are statistical outliers, not
     clinical events

Run AFTER astrostress_complete.py has generated features.csv
Usage:
    python astrostress_v2.py

Output schemas and provenance (run_id, features hash) are documented in
astrostress_outputs.py.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score, confusion_matrix)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  xgboost not installed — using GradientBoosting as substitute")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  lightgbm not installed — using GradientBoosting variant as substitute")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  shap not installed — will skip SHAP analysis")


# preprocess_fold(X_train, X_test, feature_cols) — impute, filter
# near-zero-variance, scale; all fitted on X_train only. Shared with
# astrostress_complete.py; also handles inf values, columns that are entirely
# missing in training, and folds where no feature survives.
from astrostress_preprocessing import (preprocess_fold, NoUsableFeaturesError,
                                       FoldPreprocessor, calibration_split)
import astrostress_outputs as ao

OUTPUT_DIR = "astrostress_output"


def load_features_raw():
    """Load raw features WITHOUT any global preprocessing.
    All cleaning happens inside each LOSO fold to avoid leakage."""
    path = os.path.join(OUTPUT_DIR, "features.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found! Run astrostress_complete.py first.")
        sys.exit(1)

    df = pd.read_csv(path)

    # Standardize label column
    if 'label_binary' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'label_binary': 'label'})

    # Identify ALL potential feature columns (no filtering here)
    # Labels, subject and window timing are metadata, not model features
    feature_cols = ao.feature_columns(df)

    # Only replace inf with nan — do NOT fill or filter here
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    return df, feature_cols


def get_models():
    """Return dict of model configs to compare."""
    models = {
        "Random Forest": {
            'class': RandomForestClassifier,
            'params': dict(n_estimators=200, max_depth=15, min_samples_leaf=3,
                          class_weight='balanced', random_state=42, n_jobs=-1)
        },
        "Gradient Boosting": {
            'class': GradientBoostingClassifier,
            'params': dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                          min_samples_leaf=3, random_state=42)
        },
    }

    if HAS_XGB:
        models["XGBoost"] = {
            'class': xgb.XGBClassifier,
            'params': dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                          min_child_weight=3, random_state=42,
                          eval_metric='logloss', verbosity=0)
            # NOTE: scale_pos_weight computed per fold from training labels
        }

    if HAS_LGB:
        models["LightGBM"] = {
            'class': lgb.LGBMClassifier,
            'params': dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                          min_child_samples=3, class_weight='balanced',
                          random_state=42, verbose=-1)
        }

    return models


def build_model(model_config, y_train):
    """Instantiate a model; class weighting comes from THESE training labels only.

    RF and LightGBM use class_weight='balanced' (computed by the library at
    fit time from y_train); XGBoost gets scale_pos_weight = n_neg / n_pos of
    y_train; Gradient Boosting is unweighted (as configured).
    """
    params = model_config['params'].copy()
    if HAS_XGB and model_config['class'] is xgb.XGBClassifier:
        y_train = np.asarray(y_train)
        n_neg, n_pos = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
        params['scale_pos_weight'] = n_neg / n_pos if n_pos > 0 else 1.0
    return model_config['class'](**params)


def loso_evaluate(df, feature_cols, model_config):
    """Run leakage-free LOSO cross-validation.

    ALL preprocessing (imputation, feature selection, scaling) happens
    inside each fold, fitted only on training data.
    """
    subjects = sorted(df['subject'].unique())
    results = []
    all_true, all_pred, all_prob = [], [], []

    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]

        if len(test) == 0 or len(train) == 0:
            continue

        X_train_raw = train[feature_cols].values
        y_train = train['label'].values
        X_test_raw = test[feature_cols].values
        y_test = test['label'].values

        if len(set(y_train)) < 2:
            continue

        # Preprocessing inside the fold — no leakage
        try:
            X_train, X_test, sel_features, scaler, imputer = \
                preprocess_fold(X_train_raw, X_test_raw, feature_cols)
        except NoUsableFeaturesError as e:
            print(f"    {test_subj}: skipped ({e})")
            continue

        # Fold-specific class weighting from this fold's training labels only
        model = build_model(model_config, y_train)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        results.append({
            'subject': test_subj,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'n_test': len(y_test),
            'n_stress': int(sum(y_test == 1)),
        })
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)

    if not results:
        return results, 0, 0, 0, [], [], []

    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    # AUC is pooled over all held-out predictions (per-subject AUC is
    # undefined for subjects whose windows contain a single class)
    auc = roc_auc_score(all_true, all_prob) if len(set(all_true)) > 1 else 0

    return results, avg_acc, avg_f1, auc, all_true, all_pred, all_prob


def run_model_comparison(df, feature_cols):
    """Compare multiple models with leakage-free LOSO."""
    print("\n" + "="*60)
    print("MULTI-MODEL COMPARISON (Leakage-Free LOSO)")
    print("="*60)

    models = get_models()
    comparison = []
    best_model_name = None
    best_f1 = -1
    best_results = None
    best_all_true, best_all_pred, best_all_prob = None, None, None

    for name, config in models.items():
        print(f"\n  Testing {name}...")

        results, avg_acc, avg_f1, auc, all_true, all_pred, all_prob = \
            loso_evaluate(df, feature_cols, config)

        # Compute standard deviations for uncertainty reporting
        accs = [r['accuracy'] for r in results]
        f1s = [r['f1'] for r in results]

        entry = {
            'model': name,
            'accuracy': round(avg_acc, 4),
            'accuracy_std': round(np.std(accs), 4),
            'precision': round(float(np.mean([r['precision'] for r in results])), 4) if results else 0,
            'recall': round(float(np.mean([r['recall'] for r in results])), 4) if results else 0,
            'f1': round(avg_f1, 4),
            'f1_std': round(np.std(f1s), 4),
            'auc': round(auc, 4),
            'n_subjects': len(results),
            'per_subject': results
        }
        comparison.append(entry)

        print(f"    Accuracy: {avg_acc:.3f} ± {np.std(accs):.3f} | "
              f"F1: {avg_f1:.3f} ± {np.std(f1s):.3f} | AUC: {auc:.3f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_name = name
            best_results = results
            best_all_true = all_true
            best_all_pred = all_pred
            best_all_prob = all_prob

    print(f"\n  Best Model (by F1): {best_model_name} (F1={best_f1:.3f})")

    return comparison, best_model_name, best_results, best_all_true, best_all_pred, best_all_prob


def run_fair_personalization(df, feature_cols, best_model_config):
    """Fair personalization comparison on IDENTICAL evaluation windows.

    For each subject, windows are split chronologically (calibration_split):
    the first 30% are calibration windows; the remaining windows that do not
    share raw signal with any calibration window are the evaluation windows.

    - Generic model: trained on the other subjects only
    - Personalized model: trained on the other subjects + calibration windows
    - Both are evaluated on the same evaluation windows; preprocessing for
      each is fitted on its own training rows only.
    """
    print("\n" + "="*60)
    print("FAIR PERSONALIZATION COMPARISON")
    print("(Both models tested on identical, non-overlapping evaluation windows)")
    print("="*60)

    subjects = sorted(df['subject'].unique())
    results = []

    for test_subj in subjects:
        others = df[df['subject'] != test_subj]
        subj_data = df[df['subject'] == test_subj]

        if len(subj_data) < 6:
            print(f"  {test_subj}: Skipped (only {len(subj_data)} windows)")
            continue

        cal_data, test_data, n_purged = calibration_split(subj_data)
        if len(test_data) < 2:
            print(f"  {test_subj}: Skipped (only {len(test_data)} evaluation windows)")
            continue

        y_test = test_data['label'].values
        cal_classes = len(set(cal_data['label'].values))

        train_personal = pd.concat([others, cal_data])
        y_train_generic = others['label'].values
        y_train_personal = train_personal['label'].values
        if len(set(y_train_generic)) < 2:
            print(f"  {test_subj}: Skipped (single-class training data)")
            continue

        X_eval_raw = test_data[feature_cols].values  # identical windows for both models
        try:
            X_train_g, X_test_g, _, _, _ = preprocess_fold(
                others[feature_cols].values, X_eval_raw, feature_cols)
            X_train_p, X_test_p, _, _, _ = preprocess_fold(
                train_personal[feature_cols].values, X_eval_raw, feature_cols)
        except NoUsableFeaturesError as e:
            print(f"  {test_subj}: skipped ({e})")
            continue

        generic_model = build_model(best_model_config, y_train_generic)
        generic_model.fit(X_train_g, y_train_generic)
        y_pred_generic = generic_model.predict(X_test_g)
        generic_acc = accuracy_score(y_test, y_pred_generic)
        generic_f1 = f1_score(y_test, y_pred_generic, zero_division=0)

        personal_model = build_model(best_model_config, y_train_personal)
        personal_model.fit(X_train_p, y_train_personal)
        y_pred_personal = personal_model.predict(X_test_p)
        personal_acc = accuracy_score(y_test, y_pred_personal)
        personal_f1 = f1_score(y_test, y_pred_personal, zero_division=0)

        improvement = personal_acc - generic_acc

        results.append({
            'subject': test_subj,
            'generic_accuracy': round(generic_acc, 4),
            'generic_f1': round(generic_f1, 4),
            'personal_accuracy': round(personal_acc, 4),
            'personal_f1': round(personal_f1, 4),
            'improvement': round(improvement, 4),
            'improvement_pp': round(improvement * 100, 2),
            'n_calibration': len(cal_data),
            'n_purged': n_purged,
            'n_test': len(test_data),
            'cal_classes': cal_classes,
        })

        print(f"  {test_subj}: Generic={generic_acc:.3f} → Personal={personal_acc:.3f} "
              f"({improvement * 100:+.1f} pp; cal={len(cal_data)}, purged={n_purged}, "
              f"eval={len(test_data)}, cal_classes={cal_classes})")

    if results:
        avg_improvement = np.mean([r['improvement'] for r in results])
        improved_count = sum(1 for r in results if r['improvement'] > 0)
        print(f"\n  Average accuracy change: {avg_improvement * 100:+.1f} percentage points")
        print(f"  Subjects improved: {improved_count}/{len(results)}")
        print(f"  Note: Calibration class diversity varies by subject due to WESAD protocol order")

    return results


def train_final_model(df, feature_cols, model_config, model_name):
    """Refit the selected model family on ALL windows, with its preprocessing.

    This in-sample refit is what SHAP explains and what is saved to
    trained_model.joblib. Its performance estimate is the LOSO result; the
    refit itself is not independently validated.
    """
    import sklearn
    X_raw = df[feature_cols].to_numpy(dtype=float)
    y = df['label'].values
    prep = FoldPreprocessor(feature_cols).fit(X_raw)
    model = build_model(model_config, y)
    model.fit(prep.transform(X_raw), y)
    versions = {'scikit-learn': sklearn.__version__, 'numpy': np.__version__,
                'pandas': pd.__version__}
    if HAS_XGB:
        versions['xgboost'] = xgb.__version__
    if HAS_LGB:
        versions['lightgbm'] = lgb.__version__
    return {
        'artifact_version': 1,
        'model_name': model_name,
        'model': model,
        'preprocessor': prep,                                 # imputer -> variance filter -> scaler
        'feature_cols': list(feature_cols),                   # required input columns, in order
        'selected_features': list(prep.selected_features_),   # columns the model sees
        'label_meaning': {0: 'non-stress (WESAD baseline/amusement/meditation)',
                          1: 'stress (WESAD TSST condition)'},
        'trained_on': 'all windows of all subjects (in-sample refit after LOSO model selection)',
        'n_training_windows': int(len(y)),
        'library_versions': versions,
        'usage': 'astrostress_preprocessing.predict_stress_proba(artifact, features_df)',
    }


def compute_shap_values(model, X):
    """SHAP values for the positive (stress) class, shape (n_windows, n_features)."""
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):          # older SHAP / some models: [class0, class1]
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    if sv.shape != X.shape:
        raise ValueError(f"SHAP output shape {sv.shape} does not match input {X.shape}")
    return sv


def run_shap_on_best_model(df, feature_cols, best_model_config, best_model_name, artifact=None):
    """SHAP explanations for the selected model (the saved final model).

    In-sample: explains how the refitted model uses features on the WESAD
    windows. It does not establish physiological causes of stress.
    """
    if not HAS_SHAP:
        print("\n  SHAP not available — skipping")
        return None

    print("\n" + "="*60)
    print(f"SHAP EXPLAINABILITY (on {best_model_name})")
    print("="*60)
    print("  Note: SHAP computed in-sample for global feature importance.")
    print("  This explains model behavior, not biological causation.")

    if artifact is None:
        artifact = train_final_model(df, feature_cols, best_model_config, best_model_name)
    X = artifact['preprocessor'].transform(df[artifact['feature_cols']].to_numpy(dtype=float))
    features = artifact['selected_features']

    print("  Computing SHAP values (this may take a minute)...")
    try:
        sv = compute_shap_values(artifact['model'], X)
    except Exception as e:  # never crash the pipeline because of the explainer
        print(f"  SHAP failed for {best_model_name}: {type(e).__name__}: {e} — skipping")
        return None

    mean_abs = np.abs(sv).mean(axis=0)
    shap_importance = dict(sorted(((f, round(float(v), 4)) for f, v in zip(features, mean_abs)),
                                  key=lambda x: x[1], reverse=True))

    # Per-subject mean SHAP over that subject's stress-labelled windows
    subject_shap = {}
    labels = df['label'].to_numpy()
    subjects_arr = df['subject'].to_numpy()
    for subj in sorted(df['subject'].unique()):
        mask = (subjects_arr == subj) & (labels == 1)
        if mask.sum() > 0:
            means = sv[mask].mean(axis=0)
            drivers = sorted(((f, round(float(v), 4)) for f, v in zip(features, means)),
                             key=lambda x: abs(x[1]), reverse=True)[:5]
            subject_shap[subj] = drivers

    print(f"  Top SHAP features (from {best_model_name}):")
    for feat, val in list(shap_importance.items())[:5]:
        print(f"    {feat:20s}: {val:.4f}")

    explanations = generate_explanations(shap_importance, subject_shap, df, features)
    model_cls = type(artifact['model']).__name__

    return {
        'model_used': best_model_name,
        'model_class': model_cls,
        'note': ('SHAP computed in-sample on the final model refitted on all windows. '
                 'Explains model behaviour, not biological causation.'),
        'value_units': ('probability of stress' if model_cls == 'RandomForestClassifier'
                        else 'log-odds of stress'),
        'feature_order': list(features),
        'n_windows_explained': int(len(X)),
        'global_importance': shap_importance,
        'subject_drivers': subject_shap,
        'explanations': explanations,
    }


FEATURE_DESCRIPTIONS = {
    'HR_mean': 'heart rate',
    'HRV_MeanNN': 'mean interval between heartbeats',
    'HRV_MedianNN': 'median interval between heartbeats',
    'HRV_RMSSD': 'short-term heart rate variability (RMSSD)',
    'HRV_SDNN': 'overall heart rate variability (SDNN)',
    'HRV_pNN50': 'beat-to-beat heart rate changes (pNN50)',
    'HRV_LF': 'low-frequency HRV power',
    'HRV_HF': 'high-frequency HRV power',
    'HRV_LFHF': 'LF/HF ratio',
    'HRV_SampEn': 'heart rhythm complexity (sample entropy)',
    'HRV_SD1': 'short-term heart variability (SD1)',
    'HRV_SD2': 'long-term heart variability (SD2)',
    'HR_std': 'heart rate fluctuation',
    'EDA_mean': 'average skin conductance',
    'EDA_std': 'skin conductance variability',
    'EDA_min': 'minimum skin conductance',
    'EDA_max': 'peak skin conductance',
    'EDA_range': 'skin conductance range',
    'EDA_peaks': 'number of skin conductance peaks',
}


def _join_names(names):
    if len(names) <= 1:
        return ''.join(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def generate_explanations(shap_importance, subject_shap, df, feature_cols):
    """Plain-English summaries of SHAP output (model behaviour, not causation).

    Works with any number of surviving features, including fewer than three.
    """
    explanations = {}
    top = [FEATURE_DESCRIPTIONS.get(f, f) for f in list(shap_importance)[:3]]
    if top:
        explanations['global'] = (
            f"The features with the largest average influence on the model's stress "
            f"predictions (mean |SHAP|) are: {_join_names(top)}. SHAP describes how the model "
            f"uses these features on the WESAD laboratory data; it does not establish the "
            f"physiological causes of stress."
        )
    else:
        explanations['global'] = "No features were available to explain."

    for subj, drivers in subject_shap.items():
        if not drivers:
            continue
        subj_rows = df[df['subject'] == subj]
        stress_pct = (subj_rows['label'] == 1).mean() * 100 if len(subj_rows) else 0.0
        top_feat, top_val = drivers[0]
        top_desc = FEATURE_DESCRIPTIONS.get(top_feat, top_feat)
        direction = 'toward stress' if top_val > 0 else 'away from stress'
        explanations[subj] = {
            'summary': (f"Participant {subj}: {stress_pct:.0f}% of windows are labelled stress "
                        f"(WESAD protocol). On these windows the feature with the largest average "
                        f"influence on the model is {top_desc}, which pushes predictions {direction}."),
            'stress_pct': round(float(stress_pct), 1),
            'primary_driver': top_desc,
            'direction': direction,
            'top_drivers': [(FEATURE_DESCRIPTIONS.get(f, f), round(v, 4)) for f, v in drivers[:3]],
        }

    return explanations


def run_anomaly_detection(df, feature_cols, contamination=0.1):
    """Flag statistically unusual windows with Isolation Forest.

    `contamination` (default 0.1) is the configured fraction of windows the
    algorithm flags, so the overall anomaly rate is set by this parameter —
    it is NOT a discovered event rate. Flags are statistical outliers in the
    feature space (unsupervised, fitted on all windows), not validated
    clinical events or early warnings. Overlapping windows are not
    independent observations.
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION (Isolation Forest)")
    print(f"  Contamination threshold: {contamination} (configurable)")
    print("="*60)

    X_raw = df[feature_cols].to_numpy(dtype=float)
    try:
        prep = FoldPreprocessor(feature_cols).fit(X_raw)  # aligned names; NaN/inf/constant safe
    except NoUsableFeaturesError as e:
        print(f"  Skipped: {e}")
        return [], None
    X_scaled = prep.transform(X_raw)

    iso_forest = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    )
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)

    df_anomaly = df[['subject', 'label']].copy()
    df_anomaly['anomaly'] = (anomaly_labels == -1).astype(int)
    df_anomaly['anomaly_score'] = anomaly_scores

    anomaly_results = []
    for subj in sorted(df['subject'].unique()):
        subj_mask = df['subject'] == subj
        subj_anomalies = df_anomaly[subj_mask]
        n_anomalies = subj_anomalies['anomaly'].sum()
        n_total = len(subj_anomalies)
        stress_anomalies = ((subj_anomalies['anomaly'] == 1) & (subj_anomalies['label'] == 1)).sum()

        anomaly_results.append({
            'subject': subj,
            'total_windows': n_total,
            'anomalies': int(n_anomalies),
            'anomaly_pct': round(n_anomalies / n_total * 100, 1),
            'stress_anomalies': int(stress_anomalies),
            'avg_anomaly_score': round(float(subj_anomalies['anomaly_score'].mean()), 4)
        })

        if n_anomalies > 0:
            print(f"  {subj}: {n_anomalies}/{n_total} flagged "
                  f"({n_anomalies/n_total*100:.0f}%) — {stress_anomalies} during stress")

    total_anomalies = sum(r['anomalies'] for r in anomaly_results)
    print(f"\n  Total flagged: {total_anomalies}/{len(df)} "
          f"(≈{contamination*100:.0f}% expected by threshold)")
    print(f"  Note: These are statistical outliers, not validated clinical events.")
    print(f"  Note: Overlapping windows are not independent observations.")

    return anomaly_results, df_anomaly


def generate_recommendations(stress_prob, hr_mean=None, hrv_rmssd=None, eda_mean=None):
    """Generate recommendations based on stress level.
    These are illustrative protocol suggestions, not validated medical guidance."""
    recs = []

    if stress_prob < 0.3:
        recs.append({
            'level': 'NOMINAL', 'color': 'green', 'icon': '✅',
            'message': 'All vitals within normal range',
            'actions': ['Continue current activity', 'Next check-in: 30 minutes']
        })
    elif stress_prob < 0.6:
        recs.append({
            'level': 'ELEVATED', 'color': 'yellow', 'icon': '⚠️',
            'message': 'Mild stress indicators detected',
            'actions': [
                'Consider brief breathing exercise',
                'Reduce workload if possible',
                'Hydrate',
                'Next check-in: 15 minutes'
            ]
        })
    elif stress_prob < 0.8:
        recs.append({
            'level': 'HIGH', 'color': 'orange', 'icon': '🔶',
            'message': 'Significant stress response detected',
            'actions': [
                'Pause current task for break',
                'Guided breathing exercise recommended',
                'Light stretching or movement',
                'Log stressor in crew journal',
                'Next check-in: 10 minutes'
            ]
        })
    else:
        recs.append({
            'level': 'CRITICAL', 'color': 'red', 'icon': '🚨',
            'message': 'Critical stress level — action recommended',
            'actions': [
                'Pause current activity',
                'Move to rest area',
                'Initiate calming protocol',
                'Crew medical officer assessment recommended',
                'No EVA or critical tasks until cleared'
            ]
        })

    if hr_mean and hr_mean > 90:
        recs[0]['actions'].insert(0, f'Heart rate elevated ({hr_mean:.0f} BPM) — focus on slow breathing')
    if hrv_rmssd and hrv_rmssd < 25:
        recs[0]['actions'].insert(0, 'Low HRV detected — recovery period suggested')
    if eda_mean and eda_mean > 15:
        recs[0]['actions'].insert(0, 'High skin conductance — sympathetic activation detected')

    return recs


def save_all_results(comparison, personal_results, shap_data, anomaly_results, df,
                     best_model_name=None, feature_cols=None, features_sha256=None,
                     contamination=0.1, artifact=None):
    """Save everything for dashboard consumption.

    All artifacts follow the schemas in astrostress_outputs.py and share one
    run_id; summary.json is rebuilt from this run only and written last.
    The final model artifact (if given) is saved to trained_model.joblib.
    Returns the run_id.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = ao.new_run_id()
    if best_model_name is None:
        best_model_name = max(comparison, key=lambda x: x['f1'])['model']
    if feature_cols is None:
        feature_cols = ao.feature_columns(df)
    if features_sha256 is None:
        fpath = os.path.join(OUTPUT_DIR, ao.FEATURES_FILE)
        features_sha256 = ao.file_sha256(fpath) if os.path.exists(fpath) else None

    # 1. Model comparison (with uncertainty)
    comp_df = pd.DataFrame([{k: v for k, v in m.items() if k != 'per_subject'}
                            for m in comparison])
    comp_df.insert(0, 'run_id', run_id)
    ao.write_csv_atomic(os.path.join(OUTPUT_DIR, ao.MODEL_COMPARISON_FILE), comp_df,
                        ao.MODEL_COMPARISON_COLUMNS)

    # Per-subject results for each model (reference files)
    for m in comparison:
        model_name = m['model'].lower().replace(' ', '_')
        per = pd.DataFrame(m['per_subject'])
        per.insert(0, 'run_id', run_id)
        per.to_csv(os.path.join(OUTPUT_DIR, f"loso_{model_name}.csv"), index=False)

    # Per-subject LOSO results of the selected model (read by the dashboard)
    best_entry = next(m for m in comparison if m['model'] == best_model_name)
    ao.write_csv_atomic(os.path.join(OUTPUT_DIR, ao.LOSO_RESULTS_FILE),
                        ao.loso_results_frame(best_entry['per_subject'], best_model_name, run_id),
                        ao.LOSO_RESULTS_COLUMNS)

    # 2. Personalization results (generic vs personalized on identical windows)
    pers_path = os.path.join(OUTPUT_DIR, ao.PERSONALIZED_FILE)
    if personal_results:
        ao.write_csv_atomic(pers_path, ao.personalized_frame(personal_results, run_id),
                            ao.PERSONALIZED_COLUMNS)
    else:
        # Never leave an older run's file looking current: write an empty table
        ao.write_csv_atomic(pers_path, pd.DataFrame(columns=ao.PERSONALIZED_COLUMNS),
                            ao.PERSONALIZED_COLUMNS)

    # 3. SHAP data (or remove an older run's file)
    shap_path = os.path.join(OUTPUT_DIR, ao.SHAP_FILE)
    if shap_data:
        clean_shap = {
            'schema_version': ao.SCHEMA_VERSION,
            'run_id': run_id,
            'model_used': shap_data['model_used'],
            'model_class': shap_data['model_class'],
            'note': shap_data['note'],
            'value_units': shap_data['value_units'],
            'feature_order': shap_data['feature_order'],
            'n_windows_explained': shap_data['n_windows_explained'],
            'global_importance': shap_data['global_importance'],
            'explanations': {},
        }
        for key, val in shap_data['explanations'].items():
            if isinstance(val, dict):
                clean_val = {}
                for k, v in val.items():
                    if isinstance(v, list):
                        clean_val[k] = [(str(a), float(b)) for a, b in v]
                    elif isinstance(v, (np.integer, np.floating)):
                        clean_val[k] = float(v)
                    else:
                        clean_val[k] = v
                clean_shap['explanations'][key] = clean_val
            else:
                clean_shap['explanations'][key] = val
        ao.write_json_atomic(shap_path, clean_shap)
    # If SHAP was skipped, an older shap_analysis.json (different run_id) is
    # left untouched; the dashboard loader refuses it.

    # 4. Anomaly results
    anom_df = pd.DataFrame(anomaly_results, columns=ao.ANOMALY_COLUMNS[1:])
    anom_df.insert(0, 'run_id', run_id)
    ao.write_csv_atomic(os.path.join(OUTPUT_DIR, ao.ANOMALY_FILE), anom_df, ao.ANOMALY_COLUMNS)

    # 5. Recommendations template (documented schema: ao.RECOMMENDATION_KEYS)
    rec_template = {
        'schema_version': ao.SCHEMA_VERSION,
        'run_id': run_id,
        'note': 'Illustrative protocol suggestions. Not validated medical guidance.',
        'levels': ['NOMINAL', 'ELEVATED', 'HIGH', 'CRITICAL'],
        'thresholds': [0.3, 0.6, 0.8, 1.0],
        'sample_recommendations': {
            'NOMINAL': generate_recommendations(0.1)[0],
            'ELEVATED': generate_recommendations(0.4)[0],
            'HIGH': generate_recommendations(0.7)[0],
            'CRITICAL': generate_recommendations(0.9)[0],
        }
    }
    ao.write_json_atomic(os.path.join(OUTPUT_DIR, ao.RECOMMENDATIONS_FILE), rec_template)

    # 6. Final model artifact (model + preprocessing + feature schema)
    final_model_info = None
    if artifact is not None and HAS_JOBLIB:
        artifact = dict(artifact, run_id=run_id, features_sha256=features_sha256)
        model_path = os.path.join(OUTPUT_DIR, ao.MODEL_FILE)
        joblib.dump(artifact, model_path)
        final_model_info = {
            'file': ao.MODEL_FILE, 'model_name': artifact['model_name'],
            'n_input_features': len(artifact['feature_cols']),
            'n_selected_features': len(artifact['selected_features']),
            'trained_on': artifact['trained_on'],
        }
        print(f"  Final model saved to {model_path}")

    # 7. Summary — rebuilt from THIS run only (never merged with an older
    #    summary.json, so stale metrics cannot survive). Written last.
    summary = ao.build_summary(
        run_id=run_id, comparison=comparison, best_model_name=best_model_name,
        personal_results=personal_results, shap_data=shap_data,
        anomaly_results=anomaly_results, df=df, feature_cols=feature_cols,
        features_sha256=features_sha256, contamination=contamination,
        feature_meta=ao.read_feature_meta(OUTPUT_DIR, features_sha256),
        final_model_info=final_model_info,
    )
    ao.write_json_atomic(os.path.join(OUTPUT_DIR, ao.SUMMARY_FILE), summary)

    print(f"\n  All results saved to {OUTPUT_DIR}/ (run_id {run_id})")
    return run_id


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AstroStress — Corrected Analysis Pipeline              ║")
    print("║  NASA Space Apps Challenge 2026                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load data WITHOUT global preprocessing
    print("\nLoading features (raw — preprocessing happens inside folds)...")
    features_path = os.path.join(OUTPUT_DIR, ao.FEATURES_FILE)
    features_sha256 = ao.file_sha256(features_path) if os.path.exists(features_path) else None
    df, feature_cols = load_features_raw()
    print(f"  Loaded {len(df)} windows, {len(feature_cols)} candidate features, "
          f"{len(df['subject'].unique())} subjects")

    # 1. Leakage-free model comparison
    comparison, best_name, best_results, all_true, all_pred, all_prob = \
        run_model_comparison(df, feature_cols)
    if best_name is None:
        print("ERROR: no model produced LOSO results.")
        sys.exit(1)

    # 2. Selected model config; refit it on all windows (saved + explained)
    best_config = get_models()[best_name]
    artifact = train_final_model(df, feature_cols, best_config, best_name)

    # 3. Fair personalization comparison
    personal_results = run_fair_personalization(df, feature_cols, best_config)

    # 4. SHAP on the selected (final) model
    shap_data = run_shap_on_best_model(df, feature_cols, best_config, best_name,
                                       artifact=artifact)

    # 5. Anomaly detection
    contamination = 0.1
    anomaly_results, df_anomaly = run_anomaly_detection(df, feature_cols, contamination)

    # 6. Save everything
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    save_all_results(comparison, personal_results, shap_data, anomaly_results, df,
                     best_model_name=best_name, feature_cols=feature_cols,
                     features_sha256=features_sha256, contamination=contamination,
                     artifact=artifact)

    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    best = next(m for m in comparison if m['model'] == best_name)
    print(f"  Selected model (highest mean LOSO F1): {best['model']}")
    print(f"  Accuracy:  {best['accuracy']} ± {best.get('accuracy_std', 'N/A')}")
    print(f"  Precision: {best['precision']}  Recall: {best['recall']}")
    print(f"  F1:        {best['f1']} ± {best.get('f1_std', 'N/A')}")
    print(f"  AUC (pooled): {best['auc']}")
    print(f"  Models Compared: {len(comparison)}")
    print(f"  SHAP Model: {shap_data['model_used'] if shap_data else 'N/A'}")
    print(f"  Anomalies Flagged: {sum(r['anomalies'] for r in anomaly_results)} "
          f"(contamination = {contamination:.0%} of windows, by configuration)")
    if personal_results:
        avg_imp = np.mean([r['improvement'] for r in personal_results])
        print(f"  Mean personalization accuracy change: {avg_imp * 100:+.1f} pp "
              f"(identical evaluation windows)")
    print(f"\n  Limitations documented in summary.json")
    print("  Research artifacts saved; launch the React Mission Control frontend separately.")


if __name__ == "__main__":
    main()
