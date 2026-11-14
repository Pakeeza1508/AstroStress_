"""
╔══════════════════════════════════════════════════════════════╗
║  AstroStress — Astronaut Stress Detection from Wearables   ║
║  NASA Space Apps Challenge 2026                             ║
║                                                             ║
║  Complete pipeline: data loading → feature extraction →     ║
║  LOSO evaluation → personalized vs generic comparison →     ║
║  visualization report                                       ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
  python astrostress_complete.py --data_dir C:/Users/Pakeeza/Desktop/nasa-space/WESAD
  python astrostress_complete.py --data_dir ./WESAD

REQUIREMENTS:
  pip install neurokit2 scikit-learn pandas numpy matplotlib seaborn

DATA:
  WESAD dataset (Schmidt et al., 2018)
  Download: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx
  Expected structure:
    WESAD/
      S2/S2.pkl
      S3/S3.pkl
      ...
      S17/S17.pkl
"""

import pickle
import numpy as np
import pandas as pd
import warnings
import argparse
import os
import sys
import time

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astrostress_preprocessing import (FoldPreprocessor, NoUsableFeaturesError,
                                       preprocess_fold, calibration_split)
import astrostress_outputs as ao

META_COLS = list(ao.META_COLUMNS)  # labels, subject and window timing are not features

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_wesad_subject(data_dir, subject_id):
    """Load a single WESAD subject's pickle file."""
    pkl_path = os.path.join(data_dir, subject_id, f'{subject_id}.pkl')
    if not os.path.exists(pkl_path):
        print(f"  WARNING: {pkl_path} not found, skipping.")
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def get_available_subjects(data_dir):
    """Find all subject directories in WESAD folder."""
    subjects = []
    for d in sorted(os.listdir(data_dir)):
        if d.startswith('S') and os.path.isdir(os.path.join(data_dir, d)):
            pkl_path = os.path.join(data_dir, d, f'{d}.pkl')
            if os.path.exists(pkl_path):
                subjects.append(d)
    return subjects


# ============================================================
# 2. HRV FEATURE EXTRACTION
# ============================================================

def extract_hrv_features(ecg_segment, sampling_rate=700):
    """
    Extract HRV features from ECG using NeuroKit2.

    Features extracted:
    - Time-domain: MeanNN, SDNN, RMSSD, pNN50, MedianNN
    - Frequency-domain: LF, HF, LF/HF (only if segment >= 120s)
    - Non-linear: SampEn, SD1, SD2
    - Basic: HR mean, HR std

    Returns dict of features or None if extraction fails.
    """
    import neurokit2 as nk

    try:
        signals, info = nk.ecg_process(ecg_segment, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]

        if len(r_peaks) < 5:
            return None

        features = {}

        # Time-domain
        hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
        for col in ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MedianNN']:
            if col in hrv_time.columns:
                features[col] = hrv_time[col].values[0]

        # Frequency-domain (need >= 120s for LF reliability)
        segment_duration = len(ecg_segment) / sampling_rate
        if segment_duration >= 120:
            try:
                hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate)
                for col in ['HRV_LF', 'HRV_HF', 'HRV_LFHF']:
                    if col in hrv_freq.columns:
                        val = hrv_freq[col].values[0]
                        if np.isfinite(val):
                            features[col] = val
            except Exception:
                pass

        # Non-linear
        try:
            hrv_nl = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate)
            for col in ['HRV_SampEn', 'HRV_SD1', 'HRV_SD2']:
                if col in hrv_nl.columns:
                    val = hrv_nl[col].values[0]
                    if np.isfinite(val):
                        features[col] = val
        except Exception:
            pass

        # Heart rate stats from RR intervals
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # ms
        if len(rr_intervals) > 0:
            features['HR_mean'] = 60000 / np.mean(rr_intervals)
            features['HR_std'] = np.std(60000 / rr_intervals) if len(rr_intervals) > 1 else 0

        return features
    except Exception:
        return None


def extract_eda_features(eda_segment, sampling_rate=4):
    """Extract EDA (electrodermal activity) features."""
    features = {
        'EDA_mean': np.mean(eda_segment),
        'EDA_std': np.std(eda_segment),
        'EDA_min': np.min(eda_segment),
        'EDA_max': np.max(eda_segment),
        'EDA_range': np.max(eda_segment) - np.min(eda_segment),
    }
    if len(eda_segment) > 10:
        diff = np.diff(eda_segment)
        peaks = np.sum((diff[:-1] > 0) & (diff[1:] < 0))
        features['EDA_peaks'] = peaks
    return features


def segment_and_extract(data, window_sec=120, step_sec=60, chest_rate=700):
    """
    Segment data into windows, extract features per window.

    Args:
        data: WESAD subject dict
        window_sec: Window size in seconds (default 120 for frequency domain)
        step_sec: Step size in seconds (default 60 = 50% overlap)
        chest_rate: Chest sensor sampling rate (700 Hz for WESAD)

    Returns: list of feature dicts with labels
    """
    ecg = data['signal']['chest']['ECG'].flatten()
    labels = data['label'].flatten()
    subject_id = data['subject']
    eda_wrist = data['signal']['wrist']['EDA'].flatten()
    eda_rate = 4

    window_samples = window_sec * chest_rate
    step_samples = step_sec * chest_rate

    segments = []
    stats = {'total': 0, 'mixed': 0, 'failed': 0, 'undefined': 0, 'valid': 0}

    for start in range(0, len(ecg) - window_samples + 1, step_samples):
        end = start + window_samples
        stats['total'] += 1

        window_labels = labels[start:end]
        unique = set(window_labels)
        if 0 in unique:
            stats['undefined'] += 1
            continue
        if len(unique) != 1:
            stats['mixed'] += 1
            continue

        label = unique.pop()

        # Binary: stress=1, non-stress=0
        binary_label = 1 if label == 2 else 0

        # HRV from ECG
        hrv = extract_hrv_features(ecg[start:end], chest_rate)
        if hrv is None:
            stats['failed'] += 1
            continue

        # EDA from wrist
        eda_start = int(start * eda_rate / chest_rate)
        eda_end = int(end * eda_rate / chest_rate)
        if eda_end <= len(eda_wrist):
            eda = extract_eda_features(eda_wrist[eda_start:eda_end], eda_rate)
            hrv.update(eda)

        hrv['label_binary'] = binary_label
        hrv['label_multiclass'] = label
        hrv['subject'] = subject_id
        # Window timing (seconds from recording start) — used to keep
        # calibration and evaluation windows from sharing raw samples
        hrv['window_start_s'] = start / chest_rate
        hrv['window_end_s'] = end / chest_rate
        segments.append(hrv)
        stats['valid'] += 1

    print(f"  {subject_id}: {stats['valid']} valid / {stats['total']} total "
          f"(skip: {stats['mixed']} mixed, {stats['failed']} ECG fail, {stats['undefined']} undef)")

    return segments


# ============================================================
# 3. CLASSIFICATION — GENERIC (LOSO)
# ============================================================

def loso_generic(df, feature_cols, label_col='label_binary'):
    """
    Leave-One-Subject-Out for a generic (subject-independent) model.
    Imputation, variance-based feature selection and scaling are fitted on
    the training subjects of each fold only, then applied to the held-out subject.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

    subjects = sorted(df['subject'].unique())
    results = []
    all_true, all_pred, all_prob = [], [], []

    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]

        if len(test) == 0 or len(train) == 0:
            continue
        if len(train[label_col].unique()) < 2:
            continue

        y_train = train[label_col].values
        y_test = test[label_col].values

        # Fold-local preprocessing: fitted on training subjects only
        try:
            X_train, X_test, _, _, _ = preprocess_fold(
                train[feature_cols].values, test[feature_cols].values, feature_cols)
        except NoUsableFeaturesError as e:
            print(f"  {test_subj}: skipped ({e})")
            continue

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                      class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) == 2 else np.zeros(len(y_test))

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results.append({
            'subject': test_subj,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'fpr': fpr,
            'n_test': len(y_test),
            'n_stress': sum(y_test == 1),
        })

        r = results[-1]
        print(f"  {test_subj}: Acc={r['accuracy']:.3f} F1={r['f1']:.3f} "
              f"Rec={r['recall']:.3f} Prec={r['precision']:.3f} FPR={r['fpr']:.3f}")

    return results, np.array(all_true), np.array(all_pred), np.array(all_prob)


# ============================================================
# 4. CLASSIFICATION — PERSONALIZED
# ============================================================

def loso_personalized(df, feature_cols, label_col='label_binary'):
    """
    Personalized model: for each test subject in LOSO, train a
    subject-specific model using the generic model's predictions
    calibrated with a small amount of subject data.

    Method: Train on N-1 subjects plus the first 30% (chronological) of the
    test subject's windows, evaluate on the remaining windows. Evaluation
    windows that share raw signal with a calibration window (overlapping
    sliding windows) are purged — see calibration_split.
    This simulates a "calibration period" for personalization.

    Preprocessing (imputation, feature selection, scaling) is fitted on the
    other subjects + the calibration windows only; the evaluation windows
    are only transformed.

    For a fair comparison, a generic baseline (same model, trained on the
    other subjects only) is evaluated on the IDENTICAL evaluation windows and
    reported as generic_accuracy / generic_f1; `improvement` is
    accuracy - generic_accuracy on those windows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    subjects = sorted(df['subject'].unique())
    results = []

    for test_subj in subjects:
        others = df[df['subject'] != test_subj]
        subj_data = df[df['subject'] == test_subj].copy()

        if len(subj_data) < 6:
            continue
        if len(subj_data[label_col].unique()) < 2:
            continue
        if len(others[label_col].unique()) < 2:
            continue

        # Chronological split: first 30% calibration, remainder evaluation,
        # with evaluation windows that overlap calibration windows purged
        cal_data, test_data, n_purged = calibration_split(subj_data)

        if len(test_data) < 2 or len(test_data[label_col].unique()) < 1:
            continue

        # Combine others + calibration data for training
        train_combined = pd.concat([others, cal_data])

        if len(train_combined[label_col].unique()) < 2:
            continue

        y_train = train_combined[label_col].values
        y_test = test_data[label_col].values
        X_eval_raw = test_data[feature_cols].values  # identical windows for both models

        # Generic baseline: other subjects only; personalized: others +
        # calibration. Imputer, feature mask and scaler are fitted on each
        # model's training rows only; the evaluation windows are never used
        # for fitting.
        try:
            X_train_g, X_test_g, _, _, _ = preprocess_fold(
                others[feature_cols].values, X_eval_raw, feature_cols)
            X_train, X_test, _, _, _ = preprocess_fold(
                train_combined[feature_cols].values, X_eval_raw, feature_cols)
        except NoUsableFeaturesError as e:
            print(f"  {test_subj}: skipped ({e})")
            continue

        generic_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                             class_weight='balanced')
        generic_clf.fit(X_train_g, others[label_col].values)
        y_pred_generic = generic_clf.predict(X_test_g)

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                      class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        gen_acc = accuracy_score(y_test, y_pred_generic)
        gen_f1 = f1_score(y_test, y_pred_generic, zero_division=0)

        results.append({
            'subject': test_subj,
            'accuracy': acc,
            'f1': f1,
            'recall': rec,
            'precision': prec,
            'generic_accuracy': gen_acc,
            'generic_f1': gen_f1,
            'improvement': acc - gen_acc,
            'n_calibration': len(cal_data),
            'n_purged': n_purged,
            'n_test': len(test_data),
        })

        print(f"  {test_subj}: Acc={acc:.3f} F1={f1:.3f} "
              f"(generic on same windows: Acc={gen_acc:.3f}; "
              f"cal={len(cal_data)}, test={len(test_data)})")

    return results


# ============================================================
# 5. VISUALIZATION / REPORT
# ============================================================

def generate_report(df, feature_cols, generic_results, personal_results,
                    all_true, all_pred, all_prob, output_dir):
    """Generate all evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
    from sklearn.ensemble import RandomForestClassifier

    # --- Plot 1: Confusion Matrix + ROC ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Stress', 'Stress'])
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('LOSO Confusion Matrix\nGeneric RF Model', fontsize=12)

    fpr_curve, tpr_curve, _ = roc_curve(all_true, all_prob)
    roc_auc = auc(fpr_curve, tpr_curve)
    axes[1].plot(fpr_curve, tpr_curve, 'b-', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('LOSO ROC Curve', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_confusion_roc.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Feature Importance ---
    # Descriptive, in-sample importance on all windows (NOT an evaluation
    # metric), so preprocessing here is intentionally fitted on all data.
    y_all = df['label_binary'].values
    prep = FoldPreprocessor(feature_cols)
    X_s = prep.fit_transform(df[feature_cols].values)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_s, y_all)

    imp = sorted(zip(prep.selected_features_, clf.feature_importances_),
                 key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(15, len(imp))
    names = [x[0] for x in imp[:top_n]]
    vals = [x[1] for x in imp[:top_n]]
    colors = ['#e74c3c' if 'EDA' in n else '#3498db' if 'HRV' in n else '#2ecc71' for n in names]
    ax.barh(range(top_n-1, -1, -1), vals, color=colors)
    ax.set_yticks(range(top_n-1, -1, -1))
    ax.set_yticklabels(names)
    ax.set_xlabel('Importance (Gini)')
    ax.set_title('Top Features for Stress Detection', fontsize=13)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#e74c3c', label='EDA'),
                       Patch(facecolor='#3498db', label='HRV'),
                       Patch(facecolor='#2ecc71', label='HR')], loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Generic vs Personalized comparison ---
    # Both bars per subject are measured on the IDENTICAL evaluation windows
    # (last 70% of the subject), never full-subject LOSO vs subset.
    if personal_results:
        merged = pd.DataFrame(personal_results)

        if len(merged) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            x = range(len(merged))
            w = 0.35
            axes[0].bar([i - w/2 for i in x], merged['generic_f1'], w,
                       label='Generic', color='#3498db')
            axes[0].bar([i + w/2 for i in x], merged['f1'], w,
                       label='Personalized', color='#e74c3c')
            axes[0].set_xlabel('Subject')
            axes[0].set_ylabel('F1 Score')
            axes[0].set_title('Generic vs Personalized — F1 per Subject\n'
                              '(same evaluation windows)', fontsize=12)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(merged['subject'])
            axes[0].legend()
            axes[0].set_ylim(0, 1.1)
            axes[0].grid(True, alpha=0.3, axis='y')

            # Summary bars (paired, identical windows)
            gen_means = [merged['generic_accuracy'].mean(), merged['generic_f1'].mean()]
            per_means = [merged['accuracy'].mean(), merged['f1'].mean()]

            x2 = range(len(gen_means))
            axes[1].bar([i - w/2 for i in x2], gen_means, w,
                       label='Generic', color='#3498db')
            axes[1].bar([i + w/2 for i in x2], per_means, w,
                       label='Personalized', color='#e74c3c')
            axes[1].set_xticks(x2)
            axes[1].set_xticklabels(['Accuracy', 'F1'])
            axes[1].set_title('Average Metrics Comparison (same windows)', fontsize=12)
            axes[1].legend()
            axes[1].set_ylim(0, 1.1)
            axes[1].grid(True, alpha=0.3, axis='y')

            plt.suptitle('Generic vs Personalized Stress Detection', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig3_generic_vs_personalized.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

    # --- Plot 4: Class Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bc = df['label_binary'].value_counts().sort_index()
    axes[0].bar(['Non-Stress', 'Stress'], bc.values, color=['#3498db', '#e74c3c'])
    axes[0].set_title('Binary Classes', fontsize=12)
    axes[0].set_ylabel('Windows')
    for i, v in enumerate(bc.values):
        axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    mc = df['label_multiclass'].value_counts().sort_index()
    lbl_map = {1: 'Baseline', 2: 'Stress', 3: 'Amusement', 4: 'Meditation'}
    axes[1].bar([lbl_map.get(k, str(k)) for k in mc.index], mc.values,
                color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(mc)])
    axes[1].set_title('Multi-Class', fontsize=12)
    axes[1].set_ylabel('Windows')
    for i, v in enumerate(mc.values):
        axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    plt.suptitle('WESAD Window Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  All plots saved to {output_dir}/")
    return imp, roc_auc


# ============================================================
# 6. MAIN
# ============================================================

def prepare_feature_frame(df):
    """Identify candidate feature columns and replace +/-inf with NaN.

    No statistics are computed here (no imputation, no variance filter), so
    nothing depends on held-out subjects. Returns (df_copy, feature_cols).
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df, feature_cols


def main():
    parser = argparse.ArgumentParser(description='AstroStress — Complete Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to WESAD data directory (containing S2/, S3/, etc.)')
    parser.add_argument('--window', type=int, default=120, help='Window size (seconds)')
    parser.add_argument('--step', type=int, default=60, help='Step size (seconds)')
    parser.add_argument('--output', type=str, default='./astrostress_output',
                        help='Output directory for results')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AstroStress — Astronaut Stress Detection Pipeline     ║")
    print("║  NASA Space Apps Challenge 2026                        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Find subjects
    subjects = get_available_subjects(args.data_dir)
    print(f"\nFound {len(subjects)} subjects: {subjects}")

    if len(subjects) == 0:
        print("\nERROR: No WESAD subjects found!")
        print(f"Expected structure: {args.data_dir}/S2/S2.pkl, {args.data_dir}/S3/S3.pkl, ...")
        sys.exit(1)

    # Feature extraction
    print(f"\n{'='*55}")
    print(f"STEP 1: Feature Extraction (window={args.window}s, step={args.step}s)")
    print(f"{'='*55}")

    all_segments = []
    t0 = time.time()

    for sid in subjects:
        print(f"\n  Loading {sid}...")
        data = load_wesad_subject(args.data_dir, sid)
        if data is None:
            continue
        segments = segment_and_extract(data, args.window, args.step)
        all_segments.extend(segments)
        del data  # free this subject's raw signals before loading the next

    print(f"\n  Total: {len(all_segments)} windows in {time.time()-t0:.1f}s")

    if len(all_segments) < 10:
        print("\nERROR: Too few segments. Check data.")
        sys.exit(1)

    df = pd.DataFrame(all_segments)

    # Clean features: inf -> NaN only. Imputation and feature selection are
    # done inside each LOSO fold (fitted on training subjects only).
    df, feature_cols = prepare_feature_frame(df)

    print(f"\n  Candidate features ({len(feature_cols)}): {feature_cols}")
    print(f"  (imputation + variance filter applied per fold, not globally)")
    print(f"  Non-stress: {sum(df['label_binary']==0)}, Stress: {sum(df['label_binary']==1)}")

    # Save raw (un-imputed) features so downstream LOSO stays leakage-free
    os.makedirs(args.output, exist_ok=True)
    features_path = os.path.join(args.output, ao.FEATURES_FILE)
    df.to_csv(features_path, index=False)
    # Provenance for downstream summaries (tied to this exact features.csv)
    ao.write_json_atomic(os.path.join(args.output, ao.FEATURE_META_FILE), {
        'schema_version': ao.SCHEMA_VERSION,
        'generated_at': ao.utc_now_iso(),
        'features_sha256': ao.file_sha256(features_path),
        'window_sec': args.window,
        'step_sec': args.step,
        'chest_rate_hz': 700,
        'subjects': sorted(df['subject'].unique().tolist()),
        'n_windows': int(len(df)),
        'candidate_features': feature_cols,
    })

    # Generic LOSO
    print(f"\n{'='*55}")
    print(f"STEP 2: LOSO Cross-Validation — Generic Model")
    print(f"{'='*55}")

    gen_results, all_true, all_pred, all_prob = loso_generic(df, feature_cols)

    if gen_results:
        avg = lambda k: np.mean([r[k] for r in gen_results])
        print(f"\n  GENERIC AVERAGE:")
        print(f"    Accuracy:  {avg('accuracy'):.3f}")
        print(f"    F1:        {avg('f1'):.3f}")
        print(f"    Recall:    {avg('recall'):.3f}")
        print(f"    Precision: {avg('precision'):.3f}")
        print(f"    FPR:       {avg('fpr'):.3f}")

    # Personalized
    print(f"\n{'='*55}")
    print(f"STEP 3: Personalized Model (calibration-based)")
    print(f"{'='*55}")

    per_results = loso_personalized(df, feature_cols)

    if per_results:
        avg_p = lambda k: np.mean([r[k] for r in per_results])
        print(f"\n  PERSONALIZED AVERAGE:")
        print(f"    Accuracy:  {avg_p('accuracy'):.3f}")
        print(f"    F1:        {avg_p('f1'):.3f}")
        print(f"    Recall:    {avg_p('recall'):.3f}")
        print(f"    Precision: {avg_p('precision'):.3f}")

    # Comparison
    if per_results:
        print(f"\n{'='*55}")
        print(f"STEP 4: Generic vs Personalized Comparison (identical evaluation windows)")
        print(f"{'='*55}")
        gen_acc = np.mean([r['generic_accuracy'] for r in per_results])
        per_acc = np.mean([r['accuracy'] for r in per_results])
        gen_f1 = np.mean([r['generic_f1'] for r in per_results])
        per_f1 = np.mean([r['f1'] for r in per_results])
        print(f"  Generic Acc / F1:      {gen_acc:.3f} / {gen_f1:.3f}")
        print(f"  Personalized Acc / F1: {per_acc:.3f} / {per_f1:.3f}")
        print(f"  Accuracy change:       {(per_acc - gen_acc) * 100:+.1f} percentage points")
        print(f"  F1 change:             {(per_f1 - gen_f1) * 100:+.1f} percentage points")

    # Generate plots
    print(f"\n{'='*55}")
    print(f"STEP 5: Generating Report")
    print(f"{'='*55}")

    plots_dir = os.path.join(args.output, 'plots')
    imp, roc_auc = generate_report(df, feature_cols, gen_results, per_results,
                                    all_true, all_pred, all_prob, plots_dir)

    # Final summary
    print(f"\n{'='*55}")
    print(f"COMPLETE — Results Summary")
    print(f"{'='*55}")
    print(f"  Subjects:    {len(subjects)}")
    print(f"  Windows:     {len(df)}")
    print(f"  Features:    {len(feature_cols)} candidates (selected per fold)")
    if gen_results:
        print(f"  Generic Acc: {np.mean([r['accuracy'] for r in gen_results]):.3f}")
        print(f"  Generic F1:  {np.mean([r['f1'] for r in gen_results]):.3f}")
        print(f"  AUC:         {roc_auc:.3f}")
    if per_results:
        print(f"  Personal F1: {np.mean([r['f1'] for r in per_results]):.3f}")
    print(f"\n  Top 5 features:")
    for feat, importance in imp[:5]:
        print(f"    {feat:20s}: {importance:.4f}")
    print(f"\n  Output: {args.output}/")
    print(f"  Plots:  {plots_dir}/")
    print(f"\n✅ AstroStress pipeline complete!")


if __name__ == '__main__':
    main()
