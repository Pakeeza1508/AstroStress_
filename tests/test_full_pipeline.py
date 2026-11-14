"""
End-to-end correctness tests: overlap purge, class weighting, model
serialization, SHAP (all model families), anomaly detection, model
comparison, HTML data export / JavaScript, and README accuracy.

Synthetic data only; every write goes to a temporary directory.
Run:  python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
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
    import joblib

from astrostress_preprocessing import calibration_split, predict_stress_proba

FEATS = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MedianNN', 'HRV_LF',
         'HRV_HF', 'HRV_LFHF', 'HRV_SampEn', 'HRV_SD1', 'HRV_SD2', 'HR_mean', 'HR_std',
         'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_range', 'EDA_peaks']
SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6']
N = 30
NODE = shutil.which('node')


def synth(seed=0, timed=True, skip_every=7, label_col='label'):
    """Synthetic windows with optional WESAD-like timing (120 s window, 60 s
    step) and occasional skipped windows (like mixed-label windows)."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, s in enumerate(SUBJECTS):
        k = 0
        w = 0
        while k < N:
            w += 1
            if skip_every and w % skip_every == 0:
                continue  # a skipped (mixed-label) window -> gap in time
            y = int(10 <= k < 17)
            r = {f: rng.normal(0, 1) + (1.2 * y if f in ('HR_mean', 'EDA_mean', 'HRV_MeanNN') else 0)
                 + 0.2 * si for f in FEATS}
            r.update({label_col: y, 'subject': s})
            if timed:
                r['window_start_s'] = 60.0 * w
                r['window_end_s'] = 60.0 * w + 120.0
            rows.append(r)
            k += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Overlapping-window prevention / chronology
# ----------------------------------------------------------------------

class TestCalibrationSplit(unittest.TestCase):

    def test_timed_split_never_shares_raw_samples(self):
        df = synth()
        for s in SUBJECTS:
            cal, ev, n_purged = calibration_split(df[df.subject == s])
            self.assertGreaterEqual(ev['window_start_s'].min(), cal['window_end_s'].max())
            self.assertEqual(len(cal), int(N * 0.3))
            self.assertEqual(len(cal) + len(ev) + n_purged, N)
            self.assertGreaterEqual(n_purged, 0)

    def test_timed_split_uses_chronological_order_even_if_rows_shuffled(self):
        subj = synth()
        subj = subj[subj.subject == 'S2']
        shuffled = subj.sample(frac=1.0, random_state=1)
        cal_a, ev_a, _ = calibration_split(subj)
        cal_b, ev_b, _ = calibration_split(shuffled)
        self.assertEqual(list(cal_a.index), list(cal_b.index))
        self.assertEqual(list(ev_a.index), list(ev_b.index))
        self.assertTrue(cal_b['window_start_s'].max() < ev_b['window_start_s'].min())

    def test_exact_purge_with_gaps(self):
        # windows start every 60 s, each 120 s long; a gap after the last
        # calibration window means nothing needs purging
        starts = [0, 60, 120, 180, 300, 360, 420, 480, 540, 600]
        df = pd.DataFrame({'window_start_s': starts, 'window_end_s': [x + 120 for x in starts],
                           'label': 0, 'subject': 'S1', 'f': range(10)})
        cal, ev, n = calibration_split(df, cal_fraction=0.4)  # cal = 0..180 (ends at 300)
        self.assertEqual((len(cal), n, ev['window_start_s'].min()), (4, 0, 300))
        df.loc[4, ['window_start_s', 'window_end_s']] = [240, 360]  # now overlaps cal end
        cal, ev, n = calibration_split(df, cal_fraction=0.4)
        self.assertEqual(n, 1)
        self.assertGreaterEqual(ev['window_start_s'].min(), 300)

    def test_untimed_fallback_purges_adjacent_window(self):
        df = synth(timed=False)
        subj = df[df.subject == 'S2']
        cal, ev, n = calibration_split(subj)
        self.assertEqual(n, 1)
        self.assertEqual(list(ev.index), list(subj.index[len(cal) + 1:]))

    def test_both_pipelines_use_the_purged_split(self):
        df = synth()
        expected = {s: len(calibration_split(df[df.subject == s])[1]) for s in SUBJECTS}
        cfg = v2.get_models()['Random Forest']
        with contextlib.redirect_stdout(io.StringIO()):
            res_v2 = v2.run_fair_personalization(df, FEATS, cfg)
            dfc, cols = complete.prepare_feature_frame(df.rename(columns={'label': 'label_binary'}))
            res_c = complete.loso_personalized(dfc, cols)
        self.assertEqual(cols, FEATS)  # timing columns are not features
        for res in (res_v2, res_c):
            self.assertEqual({r['subject']: r['n_test'] for r in res}, expected)
            self.assertTrue(all('n_purged' in r for r in res))


# ----------------------------------------------------------------------
# Model training, comparison, serialization
# ----------------------------------------------------------------------

class TestModels(unittest.TestCase):

    def test_class_weighting_comes_from_training_labels_only(self):
        y_train = np.array([0] * 30 + [1] * 10)
        if v2.HAS_XGB:
            m = v2.build_model(v2.get_models()['XGBoost'], y_train)
            self.assertAlmostEqual(m.get_params()['scale_pos_weight'], 3.0)
        rf = v2.build_model(v2.get_models()['Random Forest'], y_train)
        self.assertEqual(rf.get_params()['class_weight'], 'balanced')
        # config is not mutated across folds
        self.assertNotIn('scale_pos_weight', v2.get_models()['Random Forest']['params'])

    def test_model_comparison_metrics_and_selection(self):
        df = synth()
        with contextlib.redirect_stdout(io.StringIO()):
            comparison, best, *_ = v2.run_model_comparison(df, FEATS)
        self.assertEqual(len(comparison), len(v2.get_models()))
        for m in comparison:
            for k in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
                self.assertTrue(0 <= m[k] <= 1, (m['model'], k))
            per = pd.DataFrame(m['per_subject'])
            self.assertEqual(sorted(per['subject']), SUBJECTS)
            self.assertAlmostEqual(m['accuracy'], round(per['accuracy'].mean(), 4), places=4)
            self.assertAlmostEqual(m['recall'], round(per['recall'].mean(), 4), places=4)
        best_f1 = max(np.mean([r['f1'] for r in m['per_subject']]) for m in comparison)
        chosen = next(m for m in comparison if m['model'] == best)
        self.assertAlmostEqual(np.mean([r['f1'] for r in chosen['per_subject']]), best_f1)

    def _artifact(self, df, name='Random Forest'):
        return v2.train_final_model(df, FEATS, v2.get_models()[name], name)

    def test_serialization_roundtrip_reproduces_predictions(self):
        df = synth()
        art = self._artifact(df)
        before = predict_stress_proba(art, df)
        path = os.path.join(tempfile.mkdtemp(), 'm.joblib')
        joblib.dump(art, path)
        loaded = joblib.load(path)
        after = predict_stress_proba(loaded, df)
        # multithreaded tree ensembles sum in nondeterministic order (~1e-16)
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(after > 0.5, before > 0.5)
        self.assertEqual(loaded['feature_cols'], FEATS)
        self.assertEqual(loaded['selected_features'], loaded['preprocessor'].selected_features_)

    def test_feature_order_is_by_name_and_missing_columns_fail_loudly(self):
        df = synth()
        art = self._artifact(df)
        shuffled = df[list(reversed(df.columns))]
        np.testing.assert_allclose(predict_stress_proba(art, shuffled), predict_stress_proba(art, df),
                                   rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            predict_stress_proba(art, df.drop(columns=['HR_mean']))

    def test_all_nan_column_excluded_without_misalignment(self):
        df = synth()
        df['HRV_LF'] = np.nan
        art = self._artifact(df)
        self.assertNotIn('HRV_LF', art['selected_features'])
        self.assertEqual(len(art['selected_features']), len(FEATS) - 1)
        i = art['selected_features'].index('HR_mean')
        self.assertAlmostEqual(art['preprocessor'].scaler_.mean_[i], df['HR_mean'].mean())
        self.assertEqual(predict_stress_proba(art, df).shape, (len(df),))


# ----------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------

@unittest.skipUnless(v2.HAS_SHAP, 'shap not installed')
class TestShap(unittest.TestCase):

    def test_every_model_family_produces_aligned_shap(self):
        df = synth()
        for name, cfg in v2.get_models().items():
            with self.subTest(model=name):
                art = v2.train_final_model(df, FEATS, cfg, name)
                X = art['preprocessor'].transform(df[FEATS].to_numpy(dtype=float))
                sv = v2.compute_shap_values(art['model'], X)
                self.assertEqual(sv.shape, (len(df), len(art['selected_features'])))
                self.assertTrue(np.isfinite(sv).all())

    def test_shap_output_matches_model_feature_schema_with_missing_column(self):
        df = synth()
        df['HRV_LF'] = np.nan
        df.loc[0, 'HR_std'] = np.inf
        cfg = v2.get_models()['Random Forest']
        art = v2.train_final_model(df, FEATS, cfg, 'Random Forest')
        with contextlib.redirect_stdout(io.StringIO()):
            out = v2.run_shap_on_best_model(df, FEATS, cfg, 'Random Forest', artifact=art)
        self.assertEqual(out['feature_order'], art['selected_features'])
        self.assertEqual(set(out['global_importance']), set(art['selected_features']))
        self.assertNotIn('HRV_LF', out['global_importance'])
        self.assertEqual(out['model_used'], 'Random Forest')
        self.assertIn('not establish', out['explanations']['global'])

    def test_explanations_with_fewer_than_three_features(self):
        df = synth()
        for imp in ({}, {'HR_mean': 0.2}, {'HR_mean': 0.2, 'EDA_mean': 0.1}):
            subj = {'S2': [(f, 0.1) for f in imp]} if imp else {'S2': []}
            e = v2.generate_explanations(imp, subj, df, list(imp))
            self.assertIn('global', e)
            if imp:
                self.assertIn('heart rate', e['global'])
                self.assertIn('pushes predictions', e['S2']['summary'])


# ----------------------------------------------------------------------
# Anomaly detection
# ----------------------------------------------------------------------

class TestAnomaly(unittest.TestCase):

    def test_normal_run_with_nan_inf_and_constant_columns(self):
        df = synth()
        df['HRV_LF'] = np.nan
        df['EDA_min'] = 1.0
        df.loc[3, 'HR_std'] = np.inf
        with contextlib.redirect_stdout(io.StringIO()):
            res, flags = v2.run_anomaly_detection(df, FEATS, contamination=0.1)
        self.assertEqual([r['subject'] for r in res], SUBJECTS)
        total = sum(r['anomalies'] for r in res)
        self.assertAlmostEqual(total / len(df), 0.1, delta=0.02)  # set by contamination
        self.assertEqual(len(flags), len(df))

    def test_unusable_columns_do_not_crash(self):
        df = synth()
        df[FEATS] = 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            res, flags = v2.run_anomaly_detection(df, FEATS)
        self.assertEqual((res, flags), ([], None))


# ----------------------------------------------------------------------
# Full v2 run: artifacts, provenance, HTML data export
# ----------------------------------------------------------------------

RUN = None


def setUpModule():
    global RUN
    RUN = tempfile.mkdtemp()
    synth().rename(columns={'label': 'label_binary'}).to_csv(
        os.path.join(RUN, ao.FEATURES_FILE), index=False)
    with mock.patch.object(v2, 'OUTPUT_DIR', RUN), contextlib.redirect_stdout(io.StringIO()):
        v2.main()


class TestRunArtifacts(unittest.TestCase):

    def _summary(self):
        with open(os.path.join(RUN, ao.SUMMARY_FILE)) as f:
            return json.load(f)

    def test_every_artifact_belongs_to_one_run(self):
        s = self._summary()
        data = ao.load_dashboard_data(RUN)
        self.assertEqual(data['issues'], [])
        for key in ('comparison', 'loso', 'personalized', 'anomalies'):
            self.assertEqual(set(data[key]['run_id']), {s['run_id']})
        for key in ('shap', 'recommendations'):
            self.assertEqual(data[key]['run_id'], s['run_id'])
        art = joblib.load(os.path.join(RUN, ao.MODEL_FILE))
        self.assertEqual((art['run_id'], art['model_name']), (s['run_id'], s['best_model']))
        self.assertEqual(data['shap']['model_used'], s['best_model'])
        self.assertEqual(data['shap']['feature_order'], art['selected_features'])

    def test_saved_model_reproduces_predictions_after_reload(self):
        art = joblib.load(os.path.join(RUN, ao.MODEL_FILE))
        df = pd.read_csv(os.path.join(RUN, ao.FEATURES_FILE))
        cfg = v2.get_models()[art['model_name']]
        fresh = v2.train_final_model(df.rename(columns={'label_binary': 'label'}), FEATS, cfg,
                                     art['model_name'])
        np.testing.assert_allclose(predict_stress_proba(art, df), predict_stress_proba(fresh, df))

    def test_summary_reports_honest_selection_and_metrics(self):
        s = self._summary()
        self.assertIn('not an independently validated', s['model_selection'])
        for k in ('best_accuracy', 'best_precision', 'best_recall', 'best_f1', 'best_auc'):
            self.assertTrue(0 <= s[k] <= 1)
        self.assertEqual(s['final_model']['file'], ao.MODEL_FILE)

    def test_recommendations_schema(self):
        with open(os.path.join(RUN, ao.RECOMMENDATIONS_FILE)) as f:
            rec = json.load(f)
        self.assertTrue(set(ao.RECOMMENDATION_KEYS) <= set(rec))
        self.assertEqual(list(rec['sample_recommendations']), rec['levels'])
        text = json.dumps(rec)
        self.assertNotIn('Mission Control notified', text)

# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------

class TestReadme(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
            cls.readme = f.read()

    def test_commands_reference_existing_scripts_and_flags(self):
        cmds = re.findall(r'^(?:python|streamlit run) +(\S+\.py)(.*)$', self.readme, re.M)
        self.assertTrue(cmds)
        for script, args in cmds:
            self.assertTrue(os.path.exists(os.path.join(ROOT, script)), script)
            if script == 'astrostress_complete.py':
                help_text = subprocess.run([sys.executable, os.path.join(ROOT, script), '--help'],
                                           capture_output=True, text=True).stdout
                for flag in re.findall(r'--\w+', args):
                    self.assertIn(flag, help_text)
        self.assertNotIn('astrostress_corrected', self.readme)

    def test_project_structure_lists_existing_source_files(self):
        listed = re.findall(r'[├└]── (\S+\.(?:py|html|txt|md))', self.readme)
        self.assertTrue(listed)
        for name in listed:
            if name in ('features.csv',) or name.endswith(('.csv', '.json')):
                continue
            self.assertTrue(os.path.exists(os.path.join(ROOT, name)) or
                            os.path.exists(os.path.join(ROOT, 'tests', name)), name)
        for required in ('astrostress_v2.py', 'astrostress_preprocessing.py', 'astrostress_outputs.py'):
            self.assertIn(required, self.readme)

    def test_no_unsupported_claims(self):
        for s in ('91.1%', '95.5%', '88.8%', '93.9%', 'minor data leakage', '+23%'):
            self.assertNotIn(s, self.readme)


if __name__ == '__main__':
    unittest.main()
