"""
Phase 1.1 tests: evaluation artifact integrity and provenance.

Synthetic data only. The real astrostress_output/ is never read or written:
every test works in its own temporary directory.
Run:  python -m unittest discover -s tests -v
"""

import base64
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with contextlib.redirect_stdout(io.StringIO()):
    import astrostress_outputs as ao
    import astrostress_preprocessing as prep_mod
    import astrostress_complete as complete
    import astrostress_v2 as v2
    import prepare_dashboard_data as pdd

FEATS = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MedianNN', 'HRV_LF',
         'HRV_HF', 'HRV_LFHF', 'HRV_SampEn', 'HRV_SD1', 'HRV_SD2', 'HR_mean', 'HR_std',
         'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_range', 'EDA_peaks']
SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6']
N_PER_SUBJECT = 30

# Values that only exist in the legacy hardcoded artifacts — must never render
LEGACY_STRINGS = ['88.8%', '93.9%', '91.1%', '95.5%', '&lt;2s', '<2s']


def synth_features(seed=0):
    """Features as astrostress_complete.py writes them (label_binary column)."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, s in enumerate(SUBJECTS):
        for w in range(N_PER_SUBJECT):
            y = int(10 <= w < 17)
            r = {f: rng.normal(0, 1) + (1.2 * y if f in ('HR_mean', 'EDA_mean', 'HRV_MeanNN') else 0)
                 + 0.2 * si for f in FEATS}
            if rng.random() < 0.1:
                r['HRV_LF'] = np.nan
            r.update(label_binary=y, label_multiclass=2 if y else 1, subject=s)
            rows.append(r)
    return pd.DataFrame(rows)


def run_v2_pipeline(out_dir, features_df):
    features_df.to_csv(os.path.join(out_dir, ao.FEATURES_FILE), index=False)
    with mock.patch.object(v2, 'OUTPUT_DIR', out_dir), \
            contextlib.redirect_stdout(io.StringIO()):
        v2.main()


def copy_dir(src):
    dst = tempfile.mkdtemp()
    shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def write_legacy_artifacts(d):
    """Artifacts in the formats the old hardcoded prepare_dashboard_data.py and
    the pre-Phase-1.1 pipeline produced."""
    synth_features().to_csv(os.path.join(d, ao.FEATURES_FILE), index=False)
    pd.DataFrame([{'subject': s, 'accuracy': 0.911, 'f1': 0.7, 'recall': 0.8,
                   'precision': 0.9, 'fpr': 0.05} for s in SUBJECTS]).to_csv(
        os.path.join(d, ao.LOSO_RESULTS_FILE), index=False)
    pd.DataFrame([{'subject': s, 'accuracy': 0.955, 'f1': 0.9, 'cal_size': 9, 'test_size': 21}
                  for s in SUBJECTS]).to_csv(os.path.join(d, ao.PERSONALIZED_FILE), index=False)
    pd.DataFrame([{'model': 'XGBoost', 'accuracy': 0.9106, 'f1': 0.7555, 'auc': 0.9548}]).to_csv(
        os.path.join(d, ao.MODEL_COMPARISON_FILE), index=False)
    with open(os.path.join(d, ao.SUMMARY_FILE), 'w') as f:
        json.dump({'n_subjects': 15, 'generic_accuracy': 0.8885, 'auc': 0.939,
                   'best_model': 'XGBoost', 'best_accuracy': 0.9106, 'best_f1': 0.7555,
                   'best_auc': 0.9548, 'feature_importance': {'HR_mean': 0.1419}}, f)


def decode_plotly_array(v):
    if isinstance(v, dict) and 'bdata' in v:
        return np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v['dtype'])).tolist()
    return list(v)


# One real (synthetic-data) pipeline run shared by the read-only tests
FRESH = None


def setUpModule():
    global FRESH
    FRESH = tempfile.mkdtemp()
    run_v2_pipeline(FRESH, synth_features())


# ----------------------------------------------------------------------
# CSV / JSON schema compatibility
# ----------------------------------------------------------------------

class TestSchemas(unittest.TestCase):

    def test_evaluation_csvs_match_documented_schemas(self):
        for name, cols in ((ao.MODEL_COMPARISON_FILE, ao.MODEL_COMPARISON_COLUMNS),
                           (ao.LOSO_RESULTS_FILE, ao.LOSO_RESULTS_COLUMNS),
                           (ao.PERSONALIZED_FILE, ao.PERSONALIZED_COLUMNS)):
            with self.subTest(file=name):
                df = pd.read_csv(os.path.join(FRESH, name))
                self.assertEqual(list(df.columns), cols)
                self.assertGreater(len(df), 0)

    def test_single_run_id_across_artifacts(self):
        with open(os.path.join(FRESH, ao.SUMMARY_FILE)) as f:
            s = json.load(f)
        for name in (ao.MODEL_COMPARISON_FILE, ao.LOSO_RESULTS_FILE, ao.PERSONALIZED_FILE):
            self.assertEqual(set(pd.read_csv(os.path.join(FRESH, name))['run_id']), {s['run_id']})

    def test_loso_results_are_real_per_subject_results_of_selected_model(self):
        with open(os.path.join(FRESH, ao.SUMMARY_FILE)) as f:
            s = json.load(f)
        loso = pd.read_csv(os.path.join(FRESH, ao.LOSO_RESULTS_FILE))
        self.assertEqual(sorted(loso['subject']), sorted(SUBJECTS))
        self.assertEqual(set(loso['model']), {s['best_model']})
        self.assertEqual(int(loso['n_test'].sum()), len(SUBJECTS) * N_PER_SUBJECT)
        per_model = pd.read_csv(os.path.join(
            FRESH, f"loso_{s['best_model'].lower().replace(' ', '_')}.csv"))
        merged = loso.merge(per_model, on='subject', suffixes=('', '_m'))
        np.testing.assert_allclose(merged['accuracy'], merged['accuracy_m'])
        # the summary's best accuracy is the mean of these per-subject rows
        self.assertAlmostEqual(s['best_accuracy'], round(loso['accuracy'].mean(), 4), places=4)

    def test_personalized_improvement_columns_consistent(self):
        p = pd.read_csv(os.path.join(FRESH, ao.PERSONALIZED_FILE))
        np.testing.assert_allclose(p['improvement'], p['personal_accuracy'] - p['generic_accuracy'],
                                   atol=1e-3)
        np.testing.assert_allclose(p['improvement_pp'], p['improvement'] * 100, atol=0.01)

    def test_dashboard_loader_accepts_fresh_run(self):
        data = ao.load_dashboard_data(FRESH)
        self.assertEqual(data['issues'], [])
        for key in ('summary', 'comparison', 'loso', 'personalized', 'features', 'shap',
                    'anomalies', 'recommendations'):
            self.assertIn(key, data)
        h = ao.headline_metrics(data)
        self.assertEqual(h['n_subjects'], len(SUBJECTS))


# ----------------------------------------------------------------------
# Summary integrity
# ----------------------------------------------------------------------

def minimal_run_inputs():
    per_subject = [{'subject': s, 'accuracy': 0.5 + i / 10, 'f1': 0.4, 'recall': 0.4,
                    'precision': 0.4, 'fpr': 0.1, 'n_test': 10, 'n_stress': 3}
                   for i, s in enumerate(SUBJECTS[:2])]
    comparison = [
        {'model': 'Random Forest', 'accuracy': 0.61, 'accuracy_std': 0.05, 'precision': 0.4,
         'recall': 0.4, 'f1': 0.4,
         'f1_std': 0.0, 'auc': 0.7, 'n_subjects': 2, 'per_subject': per_subject},
        {'model': 'Gradient Boosting', 'accuracy': 0.55, 'accuracy_std': 0.02, 'precision': 0.3,
         'recall': 0.3, 'f1': 0.3,
         'f1_std': 0.0, 'auc': 0.6, 'n_subjects': 2, 'per_subject': per_subject},
    ]
    personal = [{'subject': 'S2', 'generic_accuracy': 0.6, 'generic_f1': 0.5,
                 'personal_accuracy': 0.7, 'personal_f1': 0.6, 'improvement': 0.1,
                 'improvement_pp': 10.0, 'n_calibration': 3, 'n_purged': 1, 'n_test': 7, 'cal_classes': 1}]
    anomalies = [{'subject': 'S2', 'total_windows': 10, 'anomalies': 1, 'anomaly_pct': 10.0,
                  'stress_anomalies': 0, 'avg_anomaly_score': -0.4}]
    df = synth_features().rename(columns={'label_binary': 'label'})
    return comparison, personal, anomalies, df


class TestSummaryIntegrity(unittest.TestCase):

    def _save(self, out, personal='default'):
        comparison, pers, anomalies, df = minimal_run_inputs()
        if personal != 'default':
            pers = personal
        with mock.patch.object(v2, 'OUTPUT_DIR', out), contextlib.redirect_stdout(io.StringIO()):
            run_id = v2.save_all_results(comparison, pers, None, anomalies, df,
                                         best_model_name='Random Forest', feature_cols=FEATS,
                                         features_sha256='abc')
        with open(os.path.join(out, ao.SUMMARY_FILE)) as f:
            return run_id, json.load(f)

    def test_stale_keys_and_values_do_not_survive(self):
        out = tempfile.mkdtemp()
        write_legacy_artifacts(out)
        run_id, s = self._save(out)
        for stale in ('generic_accuracy', 'auc', 'feature_importance'):
            self.assertNotIn(stale, s)
        self.assertNotIn(0.9106, s.values())
        self.assertEqual(s['run_id'], run_id)
        self.assertEqual(s['best_accuracy'], 0.61)
        self.assertEqual(s['best_model'], 'Random Forest')
        self.assertEqual(s['n_subjects'], 2)

    def test_summary_keys_exactly_documented(self):
        _, s = self._save(tempfile.mkdtemp())
        self.assertTrue(set(ao.SUMMARY_KEYS) <= set(s))
        self.assertTrue(set(s) <= set(ao.SUMMARY_KEYS) | set(ao.SUMMARY_OPTIONAL_KEYS))
        self.assertEqual(s['schema_version'], ao.SCHEMA_VERSION)
        self.assertTrue(any('LOSO' in l or 'optimistic' in l for l in s['limitations']))
        self.assertEqual(s['personalization_avg_improvement_pp'], 10.0)

    def test_best_model_is_the_selected_one_not_recomputed(self):
        comparison, pers, anomalies, df = minimal_run_inputs()
        s = ao.build_summary(run_id='r', comparison=comparison, best_model_name='Gradient Boosting',
                             personal_results=pers, shap_data=None, anomaly_results=anomalies,
                             df=df, feature_cols=FEATS, features_sha256='x')
        self.assertEqual((s['best_model'], s['best_accuracy']), ('Gradient Boosting', 0.55))

    def test_no_personalization_results_is_explicit(self):
        out = tempfile.mkdtemp()
        write_legacy_artifacts(out)  # contains an old personalized_results.csv
        _, s = self._save(out, personal=[])
        self.assertFalse(s['personalization_tested'])
        self.assertNotIn('personalization_avg_improvement', s)
        p = pd.read_csv(os.path.join(out, ao.PERSONALIZED_FILE))
        self.assertEqual(list(p.columns), ao.PERSONALIZED_COLUMNS)
        self.assertEqual(len(p), 0)  # old rows cannot survive

    def test_feature_meta_only_when_hash_matches(self):
        out = tempfile.mkdtemp()
        meta = {'features_sha256': 'abc', 'window_sec': 120, 'step_sec': 60}
        ao.write_json_atomic(os.path.join(out, ao.FEATURE_META_FILE), meta)
        _, s = self._save(out)
        self.assertEqual(s['feature_extraction']['window_sec'], 120)
        ao.write_json_atomic(os.path.join(out, ao.FEATURE_META_FILE),
                             dict(meta, features_sha256='different'))
        _, s = self._save(out)
        self.assertNotIn('feature_extraction', s)

    def test_complete_writes_feature_meta_tied_to_features_csv(self):
        out = tempfile.mkdtemp()
        df = synth_features()
        argv = ['astrostress_complete.py', '--data_dir', 'unused', '--output', out]
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(complete, 'get_available_subjects', return_value=SUBJECTS), \
                mock.patch.object(complete, 'load_wesad_subject', side_effect=lambda d, s: s), \
                mock.patch.object(complete, 'segment_and_extract',
                                  side_effect=lambda s, w, st: df[df.subject == s].to_dict('records')), \
                contextlib.redirect_stdout(io.StringIO()):
            complete.main()
        with open(os.path.join(out, ao.FEATURE_META_FILE)) as f:
            meta = json.load(f)
        self.assertEqual(meta['features_sha256'], ao.file_sha256(os.path.join(out, ao.FEATURES_FILE)))
        self.assertEqual((meta['window_sec'], meta['step_sec']), (120, 60))
        self.assertEqual(meta['n_windows'], len(df))


# ----------------------------------------------------------------------
# Missing / legacy / stale results handling
# ----------------------------------------------------------------------

class TestMissingAndStaleResults(unittest.TestCase):

    def test_empty_directory(self):
        data = ao.load_dashboard_data(tempfile.mkdtemp())
        self.assertIsNone(ao.headline_metrics(data))
        self.assertTrue(any('summary.json not found' in m for m in data['issues']))

    def test_legacy_hardcoded_artifacts_are_rejected(self):
        d = tempfile.mkdtemp()
        write_legacy_artifacts(d)
        data = ao.load_dashboard_data(d)
        for key in ('summary', 'comparison', 'loso', 'personalized'):
            self.assertNotIn(key, data)
        self.assertIsNone(ao.headline_metrics(data))
        self.assertEqual(len(data['issues']), 4)

    def test_results_from_different_features_are_hidden(self):
        d = copy_dir(FRESH)
        feats = pd.read_csv(os.path.join(d, ao.FEATURES_FILE))
        feats.loc[0, 'HR_mean'] += 1.0
        feats.to_csv(os.path.join(d, ao.FEATURES_FILE), index=False)
        data = ao.load_dashboard_data(d)
        for key in ('summary', 'comparison', 'loso', 'personalized'):
            self.assertNotIn(key, data)
        self.assertTrue(any('different features.csv' in m for m in data['issues']))

    def test_artifact_from_other_run_is_hidden(self):
        d = copy_dir(FRESH)
        p = os.path.join(d, ao.LOSO_RESULTS_FILE)
        loso = pd.read_csv(p)
        loso['run_id'] = 'some-older-run'
        loso.to_csv(p, index=False)
        data = ao.load_dashboard_data(d)
        self.assertNotIn('loso', data)
        self.assertIn('summary', data)
        self.assertIn('personalized', data)

    def test_prepare_dashboard_data_is_read_only_and_has_no_numbers(self):
        d = tempfile.mkdtemp()
        write_legacy_artifacts(d)
        before = {f: ao.file_sha256(os.path.join(d, f)) for f in os.listdir(d)}
        with contextlib.redirect_stdout(io.StringIO()):
            rc = pdd.main(['--output_dir', d])
            rc_fresh = pdd.main(['--output_dir', FRESH])
        after = {f: ao.file_sha256(os.path.join(d, f)) for f in os.listdir(d)}
        self.assertEqual(before, after)
        self.assertEqual((rc, rc_fresh), (1, 0))
        for attr in ('loso_results', 'personalized_results', 'feature_importance'):
            self.assertFalse(hasattr(pdd, attr))


# ----------------------------------------------------------------------
# Generic vs personalized on identical test windows
# ----------------------------------------------------------------------

def record_test_windows(module, fn, *args):
    calls = []
    real = prep_mod.preprocess_fold

    def spy(X_train, X_test, feature_cols):
        calls.append(np.array(X_test, dtype=float))
        return real(X_train, X_test, feature_cols)

    with mock.patch.object(module, 'preprocess_fold', side_effect=spy), \
            contextlib.redirect_stdout(io.StringIO()):
        result = fn(*args)
    return result, calls


class TestIdenticalEvaluationWindows(unittest.TestCase):

    def _check_pairs(self, calls, results, n_eval_expected):
        self.assertEqual(len(calls), 2 * len(SUBJECTS))
        for k in range(len(SUBJECTS)):
            generic_X, personal_X = calls[2 * k], calls[2 * k + 1]
            np.testing.assert_array_equal(generic_X, personal_X)  # NaNs compare equal
            self.assertEqual(len(generic_X), n_eval_expected)
        self.assertTrue(all(r['n_test'] == n_eval_expected for r in results))

    def test_v2_fair_personalization(self):
        df = synth_features().rename(columns={'label_binary': 'label'})
        cfg = v2.get_models()['Random Forest']
        res, calls = record_test_windows(v2, v2.run_fair_personalization, df, FEATS, cfg)
        cal_n = max(int(N_PER_SUBJECT * 0.3), 2)
        # untimed rows: the window adjacent to calibration overlaps it and is purged
        self._check_pairs(calls, res, N_PER_SUBJECT - cal_n - 1)
        self.assertTrue(all(r['n_purged'] == 1 for r in res))
        expected_eval = df[df.subject == SUBJECTS[0]].iloc[cal_n + 1:][FEATS].to_numpy(dtype=float)
        np.testing.assert_array_equal(calls[0], expected_eval)

    def test_complete_personalized_has_paired_generic_baseline(self):
        df, cols = complete.prepare_feature_frame(synth_features())
        res, calls = record_test_windows(complete, complete.loso_personalized, df, cols)
        cal_n = max(int(N_PER_SUBJECT * 0.3), 2)
        self._check_pairs(calls, res, N_PER_SUBJECT - cal_n - 1)  # 1 overlapping window purged
        for r in res:
            self.assertAlmostEqual(r['improvement'], r['accuracy'] - r['generic_accuracy'])

    def test_personalization_view_uses_only_paired_columns(self):
        p = pd.read_csv(os.path.join(FRESH, ao.PERSONALIZED_FILE))
        pv = ao.personalization_view(p)
        self.assertEqual(pv['n_subjects'], len(p))
        self.assertAlmostEqual(pv['mean_improvement_pp'],
                               ((p['personal_accuracy'] - p['generic_accuracy']) * 100).mean())
        self.assertEqual(pv['n_improved'] + pv['n_worse'] <= pv['n_subjects'], True)


if __name__ == '__main__':
    unittest.main()
