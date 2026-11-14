"""
AstroStress — output artifact schemas, provenance and loading.

Single source of truth for the files `astrostress_v2.py` writes and
React/FastAPI consumers read. Pure pandas/json with no UI framework dependency.

Evaluation artifacts and their schemas (SCHEMA_VERSION = 3)
-----------------------------------------------------------
summary.json
    Built fresh on every run by `build_summary` (never merged with an older
    file). Keys: SUMMARY_KEYS (+ optional SUMMARY_OPTIONAL_KEYS).
model_comparison.csv
    One row per model, mean over LOSO folds. Columns: MODEL_COMPARISON_COLUMNS.
loso_results.csv
    Per-subject LOSO results of the selected (best) model; each subject's row
    covers ALL of that subject's windows. Columns: LOSO_RESULTS_COLUMNS.
personalized_results.csv
    Per-subject fair personalization comparison. `generic_*` and `personal_*`
    are measured on the IDENTICAL evaluation windows (the subject's last 70%).
    `improvement` = personal_accuracy - generic_accuracy (fraction);
    `improvement_pp` = the same in percentage points.
    Columns: PERSONALIZED_COLUMNS.

Provenance: all four files carry the same `run_id`; summary.json records the
SHA-256 of the features.csv the run used. `load_dashboard_data` only exposes
evaluation results that match the current schema, share the summary's run_id
and were produced from the current features.csv.
"""

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCHEMA_VERSION = 3

# Non-feature columns that may appear in features.csv
META_COLUMNS = ('label', 'label_binary', 'label_multiclass', 'subject', 'window_idx',
                'window_start_s', 'window_end_s')


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLUMNS]

FEATURES_FILE = 'features.csv'
FEATURE_META_FILE = 'feature_extraction.json'
SUMMARY_FILE = 'summary.json'
MODEL_COMPARISON_FILE = 'model_comparison.csv'
LOSO_RESULTS_FILE = 'loso_results.csv'
PERSONALIZED_FILE = 'personalized_results.csv'
SHAP_FILE = 'shap_analysis.json'
ANOMALY_FILE = 'anomaly_results.csv'
RECOMMENDATIONS_FILE = 'recommendations.json'
MODEL_FILE = 'trained_model.joblib'

MODEL_COMPARISON_COLUMNS = ['run_id', 'model', 'accuracy', 'accuracy_std', 'precision',
                            'recall', 'f1', 'f1_std', 'auc', 'n_subjects']
LOSO_RESULTS_COLUMNS = ['run_id', 'subject', 'model', 'accuracy', 'f1', 'recall',
                        'precision', 'fpr', 'n_test', 'n_stress']
PERSONALIZED_COLUMNS = ['run_id', 'subject', 'generic_accuracy', 'generic_f1',
                        'personal_accuracy', 'personal_f1', 'improvement', 'improvement_pp',
                        'n_calibration', 'n_purged', 'n_test', 'cal_classes']
ANOMALY_COLUMNS = ['run_id', 'subject', 'total_windows', 'anomalies', 'anomaly_pct',
                   'stress_anomalies', 'avg_anomaly_score']
SHAP_KEYS = ['schema_version', 'run_id', 'model_used', 'model_class', 'note', 'value_units',
             'feature_order', 'n_windows_explained', 'global_importance', 'explanations']
RECOMMENDATION_KEYS = ['schema_version', 'run_id', 'note', 'levels', 'thresholds',
                       'sample_recommendations']

SUMMARY_KEYS = [
    'schema_version', 'run_id', 'generated_at', 'pipeline_version', 'preprocessing',
    'dataset', 'evaluation', 'features_file', 'features_sha256',
    'n_subjects', 'n_windows', 'n_candidate_features', 'label_counts',
    'best_model', 'best_accuracy', 'best_accuracy_std', 'best_precision', 'best_recall',
    'best_f1', 'best_f1_std', 'best_auc', 'auc_method',
    'models_compared', 'model_selection', 'final_model',
    'has_shap', 'shap_model', 'has_anomaly_detection', 'anomaly_contamination_threshold',
    'total_anomalies', 'personalization_tested', 'limitations',
]
SUMMARY_OPTIONAL_KEYS = [
    'feature_extraction', 'shap_top_features', 'global_explanation',
    'personalization_n_subjects', 'personalization_mean_generic_accuracy',
    'personalization_mean_personal_accuracy', 'personalization_avg_improvement',
    'personalization_avg_improvement_pp', 'personalization_subjects_improved',
    'personalization_method',
]

LIMITATIONS = [
    'Validated on WESAD lab data only — not astronaut/microgravity data',
    'Best model is selected by mean LOSO F1 and reported on the same LOSO run '
    '(no separate held-out test set), so the reported best-model metrics are optimistic',
    'Anomaly counts reflect configured threshold, not discovered rate',
    'SHAP explains model predictions, not biological causation',
    'Personalization improvement preliminary — calibration diversity varies',
    'Sliding windows overlap by 50%, so LOSO windows are not independent observations; '
    'in the personalization comparison, evaluation windows that overlap calibration windows '
    'are purged',
    'The saved final model is refitted on all windows after model selection; its '
    'performance estimate is the LOSO result, not an independent validation',
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def new_run_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(_to_builtin(obj), f, indent=2)  # ASCII-safe for any reader encoding
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_csv_atomic(path, df, columns):
    """Write df with exactly `columns` (in order); raises if any is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing columns {missing}")
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.csv')
    os.close(fd)
    try:
        df[columns].to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ----------------------------------------------------------------------
# Building artifacts (used by astrostress_v2.py)
# ----------------------------------------------------------------------

def loso_results_frame(per_subject, model_name, run_id):
    df = pd.DataFrame(per_subject)
    df.insert(0, 'run_id', run_id)
    df['model'] = model_name
    return df


def personalized_frame(personal_results, run_id):
    df = pd.DataFrame(personal_results)
    df.insert(0, 'run_id', run_id)
    if 'improvement_pp' not in df.columns and 'improvement' in df.columns:
        df['improvement_pp'] = (df['improvement'] * 100).round(2)
    return df


def read_feature_meta(output_dir, features_sha256):
    """Feature-extraction metadata written by astrostress_complete.py, only if it
    describes the same features.csv (matching hash); otherwise None."""
    path = os.path.join(output_dir, FEATURE_META_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if meta.get('features_sha256') != features_sha256:
        return None
    return {k: meta[k] for k in ('window_sec', 'step_sec', 'chest_rate_hz', 'subjects',
                                 'generated_at') if k in meta}


def build_summary(*, run_id, comparison, best_model_name, personal_results, shap_data,
                  anomaly_results, df, feature_cols, features_sha256,
                  contamination=0.1, feature_meta=None, generated_at=None,
                  final_model_info=None):
    """Build summary.json from current-run values ONLY (no merging)."""
    best = next(m for m in comparison if m['model'] == best_model_name)
    label_col = 'label' if 'label' in df.columns else 'label_binary'
    s = {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'generated_at': generated_at or utc_now_iso(),
        'pipeline_version': 'corrected',
        'preprocessing': 'leakage-free (imputation + feature selection + scaling inside LOSO folds)',
        'dataset': 'WESAD',
        'evaluation': 'Leave-One-Subject-Out',
        'features_file': FEATURES_FILE,
        'features_sha256': features_sha256,
        'n_subjects': int(best.get('n_subjects', 0)),
        'n_windows': int(len(df)),
        'n_candidate_features': int(len(feature_cols)),
        'label_counts': {'non_stress': int((df[label_col] == 0).sum()),
                         'stress': int((df[label_col] == 1).sum())},
        'best_model': best['model'],
        'best_accuracy': best['accuracy'],
        'best_accuracy_std': best.get('accuracy_std', 0),
        'best_precision': best.get('precision', 0),
        'best_recall': best.get('recall', 0),
        'best_f1': best['f1'],
        'best_f1_std': best.get('f1_std', 0),
        'best_auc': best['auc'],
        'auc_method': 'pooled over all held-out LOSO predictions',
        'models_compared': len(comparison),
        'model_selection': ('highest mean LOSO F1 across subjects; best under this procedure, '
                            'not an independently validated production model'),
        'final_model': final_model_info,
        'has_shap': shap_data is not None,
        'shap_model': shap_data['model_used'] if shap_data else None,
        'has_anomaly_detection': anomaly_results is not None,
        'anomaly_contamination_threshold': contamination,
        'total_anomalies': int(sum(r['anomalies'] for r in anomaly_results)) if anomaly_results else 0,
        'personalization_tested': bool(personal_results),
        'limitations': list(LIMITATIONS),
    }
    if feature_meta:
        s['feature_extraction'] = feature_meta
    if shap_data:
        s['shap_top_features'] = dict(list(shap_data['global_importance'].items())[:5])
        s['global_explanation'] = shap_data['explanations'].get('global', '')
    if personal_results:
        imp = [r['improvement'] for r in personal_results]
        s['personalization_n_subjects'] = len(personal_results)
        s['personalization_mean_generic_accuracy'] = round(float(np.mean(
            [r['generic_accuracy'] for r in personal_results])), 4)
        s['personalization_mean_personal_accuracy'] = round(float(np.mean(
            [r['personal_accuracy'] for r in personal_results])), 4)
        s['personalization_avg_improvement'] = round(float(np.mean(imp)), 4)
        s['personalization_avg_improvement_pp'] = round(float(np.mean(imp)) * 100, 2)
        s['personalization_subjects_improved'] = f"{sum(1 for i in imp if i > 0)}/{len(imp)}"
        s['personalization_method'] = ('generic (other subjects only) vs personalized (other '
                                       'subjects + first 30% of the subject), both evaluated '
                                       'on the same last 70% of the subject\'s windows')
    unknown = set(s) - set(SUMMARY_KEYS) - set(SUMMARY_OPTIONAL_KEYS)
    assert not unknown, unknown
    return s


# ----------------------------------------------------------------------
# Loading for the dashboard
# ----------------------------------------------------------------------

def _read_csv(path):
    return pd.read_csv(path)


def load_dashboard_data(output_dir):
    """Load pipeline artifacts, exposing evaluation results only if verified.

    Returns dict with optional keys: features, summary, comparison, loso,
    personalized, shap, anomalies, recommendations — and always `issues`
    (list of human-readable messages about missing/legacy/stale artifacts).
    """
    data = {'issues': []}
    issues = data['issues']

    def p(name):
        return os.path.join(output_dir, name)

    # Non-evaluation artifacts (shown as before)
    features_sha = None
    if os.path.exists(p(FEATURES_FILE)):
        feats = _read_csv(p(FEATURES_FILE))
        if 'label_binary' in feats.columns and 'label' not in feats.columns:
            feats = feats.rename(columns={'label_binary': 'label'})
        data['features'] = feats
        features_sha = file_sha256(p(FEATURES_FILE))

    # Evaluation results: summary is the anchor
    summary = None
    if not os.path.exists(p(SUMMARY_FILE)):
        issues.append(f"{SUMMARY_FILE} not found — run `python astrostress_v2.py` to "
                      f"generate verified evaluation results.")
    else:
        with open(p(SUMMARY_FILE), encoding='utf-8') as f:
            raw = json.load(f)
        missing = [k for k in SUMMARY_KEYS if k not in raw]
        if raw.get('schema_version') != SCHEMA_VERSION or missing:
            issues.append(f"{SUMMARY_FILE} is in a legacy/unsupported format (not produced by "
                          f"the current pipeline) — its metrics are not shown. Re-run "
                          f"`python astrostress_v2.py`.")
        elif features_sha is not None and raw.get('features_sha256') != features_sha:
            issues.append(f"Evaluation results were generated from a different {FEATURES_FILE} "
                          f"than the one now on disk — metrics are not shown. Re-run "
                          f"`python astrostress_v2.py`.")
        else:
            summary = raw
    if summary is not None:
        data['summary'] = summary

    run_id = summary['run_id'] if summary else None
    for key, name, cols in (('comparison', MODEL_COMPARISON_FILE, MODEL_COMPARISON_COLUMNS),
                            ('loso', LOSO_RESULTS_FILE, LOSO_RESULTS_COLUMNS),
                            ('personalized', PERSONALIZED_FILE, PERSONALIZED_COLUMNS)):
        if not os.path.exists(p(name)):
            if summary is not None:
                issues.append(f"{name} not found.")
            continue
        df = _read_csv(p(name))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            issues.append(f"{name} is in a legacy/unsupported format (missing {missing}) — "
                          f"not shown.")
        elif run_id is None:
            pass  # no verified summary to anchor against; already reported
        elif set(df['run_id'].astype(str)) != {run_id}:
            issues.append(f"{name} belongs to a different pipeline run than {SUMMARY_FILE} — "
                          f"not shown.")
        else:
            data[key] = df

    # SHAP / anomaly / recommendation artifacts must belong to the same run
    for key, name, required in (('shap', SHAP_FILE, SHAP_KEYS),
                                ('recommendations', RECOMMENDATIONS_FILE, RECOMMENDATION_KEYS)):
        if not os.path.exists(p(name)):
            continue
        with open(p(name), encoding='utf-8') as f:
            obj = json.load(f)
        if any(k not in obj for k in required):
            issues.append(f"{name} is in a legacy/unsupported format — not shown.")
        elif run_id is not None and obj['run_id'] == run_id:
            data[key] = obj
        elif run_id is not None:
            issues.append(f"{name} belongs to a different pipeline run — not shown.")
    if os.path.exists(p(ANOMALY_FILE)):
        df = _read_csv(p(ANOMALY_FILE))
        if any(c not in df.columns for c in ANOMALY_COLUMNS):
            issues.append(f"{ANOMALY_FILE} is in a legacy/unsupported format — not shown.")
        elif run_id is not None and set(df['run_id'].astype(str)) == {run_id}:
            data['anomalies'] = df
        elif run_id is not None:
            issues.append(f"{ANOMALY_FILE} belongs to a different pipeline run — not shown.")
    return data


def headline_metrics(data):
    """Verified headline numbers, or None if no verified summary is available."""
    s = data.get('summary')
    if not s:
        return None
    return {
        'best_model': s['best_model'],
        'accuracy': s['best_accuracy'], 'accuracy_std': s['best_accuracy_std'],
        'f1': s['best_f1'], 'auc': s['best_auc'],
        'n_subjects': s['n_subjects'], 'n_windows': s['n_windows'],
        'models_compared': s['models_compared'],
    }


def personalization_view(pers_df):
    """Per-subject rows (all evaluated subjects) + aggregates from the fair
    comparison; generic and personalized on identical evaluation windows."""
    rows = pers_df.sort_values('subject', key=lambda s: s.str.extract(r'(\d+)')[0]
                               .astype(float).fillna(0)).reset_index(drop=True)
    pp = rows['personal_accuracy'] * 100 - rows['generic_accuracy'] * 100
    return {
        'rows': rows,
        'n_subjects': int(len(rows)),
        'mean_generic_accuracy': float(rows['generic_accuracy'].mean()),
        'mean_personal_accuracy': float(rows['personal_accuracy'].mean()),
        'mean_improvement_pp': float(pp.mean()),
        'n_improved': int((pp > 0).sum()),
        'n_worse': int((pp < 0).sum()),
        'largest_gain': rows.loc[pp.idxmax()] if len(rows) else None,
    }


