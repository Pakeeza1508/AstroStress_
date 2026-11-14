"""
Phase 1 leakage tests for AstroStress LOSO preprocessing.

Synthetic data only — no WESAD download required.
Run:  python -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with contextlib.redirect_stdout(io.StringIO()):
    import astrostress_preprocessing as prep_mod
    import astrostress_complete as complete
    import astrostress_v2 as v2

from astrostress_preprocessing import (FoldPreprocessor, NoUsableFeaturesError,
                                       preprocess_fold)

FEATURES = ['f_signal', 'f_noise', 'f_nan', 'f_const', 'f_only_s4_varies', 'f_inf']
SUBJECTS = ['S1', 'S2', 'S3', 'S4', 'S5']


def make_df(seed=0, n_per_subject=20, label_col='label_binary'):
    """Chronological windows per subject; stress block in the middle."""
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, subj in enumerate(SUBJECTS):
        for w in range(n_per_subject):
            y = 1 if n_per_subject * 0.4 <= w < n_per_subject * 0.7 else 0
            rows.append({
                'f_signal': y * 2.0 + rng.normal(0, 0.5) + s_i * 0.3,
                'f_noise': rng.normal(0, 1),
                'f_nan': np.nan if rng.random() < 0.3 else rng.normal(5, 1),
                'f_const': 1.0,
                # constant in every subject except S4 -> global std > 0,
                # but zero variance whenever S4 is held out
                'f_only_s4_varies': rng.normal(0, 1) if subj == 'S4' else 3.0,
                'f_inf': np.inf if rng.random() < 0.1 else rng.normal(0, 1),
                label_col: y,
                'subject': subj,
            })
    df = pd.DataFrame(rows)
    if label_col == 'label_binary':
        df['label_multiclass'] = np.where(df['label_binary'] == 1, 2, 1)
    return df


def perturb(df, mask):
    """Wildly change feature values (incl. NaN/inf) for the masked rows."""
    df = df.copy()
    rng = np.random.default_rng(123)
    vals = df.loc[mask, FEATURES].to_numpy(dtype=float) * 1000 + 500
    vals += rng.normal(0, 100, vals.shape)
    vals[0, 0] = np.nan
    vals[1, 1] = np.inf
    vals[2, 2] = -np.inf
    df.loc[mask, FEATURES] = vals
    return df


def fitted_state(X_train, X_test, feature_cols):
    """Run preprocess_fold and return everything fitted from training data."""
    out = preprocess_fold(X_train, X_test, feature_cols)
    _, _, sel, scaler, imputer = out
    return out, {
        'imputer_stats': imputer.statistics_.copy(),
        'selected': list(sel),
        'scaler_mean': scaler.mean_.copy(),
        'scaler_scale': scaler.scale_.copy(),
    }


def assert_states_equal(tc, a, b):
    tc.assertEqual(a['selected'], b['selected'])
    np.testing.assert_array_equal(a['imputer_stats'], b['imputer_stats'])
    np.testing.assert_array_equal(a['scaler_mean'], b['scaler_mean'])
    np.testing.assert_array_equal(a['scaler_scale'], b['scaler_scale'])


def states_equal(a, b):
    return (a['selected'] == b['selected']
            and np.array_equal(a['imputer_stats'], b['imputer_stats'])
            and np.array_equal(a['scaler_mean'], b['scaler_mean'])
            and np.array_equal(a['scaler_scale'], b['scaler_scale']))


def record_calls(module, fn, *args, **kwargs):
    """Run fn with module.preprocess_fold wrapped; return (result, [states])."""
    states = []
    real = prep_mod.preprocess_fold

    def spy(X_train, X_test, feature_cols):
        out, st = fitted_state(X_train, X_test, feature_cols)
        st['n_test'] = len(X_test)
        states.append(st)
        return out

    with mock.patch.object(module, 'preprocess_fold', side_effect=spy), \
            contextlib.redirect_stdout(io.StringIO()):
        result = fn(*args, **kwargs)
    return result, states


# ----------------------------------------------------------------------
# Unit-level: preprocess_fold / FoldPreprocessor
# ----------------------------------------------------------------------

class TestFoldPreprocessor(unittest.TestCase):

    def setUp(self):
        df = make_df()
        self.train = df[df.subject != 'S4']
        self.test = df[df.subject == 'S4']
        self.df = df

    def _states_for_test_variants(self):
        _, s1 = fitted_state(self.train[FEATURES].values, self.test[FEATURES].values, FEATURES)
        changed = perturb(self.df, self.df.subject == 'S4')
        _, s2 = fitted_state(self.train[FEATURES].values,
                             changed[changed.subject == 'S4'][FEATURES].values, FEATURES)
        return s1, s2

    def test_imputation_stats_ignore_test_values(self):
        s1, s2 = self._states_for_test_variants()
        np.testing.assert_array_equal(s1['imputer_stats'], s2['imputer_stats'])
        # medians equal the TRAINING medians (inf treated as missing)
        X = self.train[FEATURES].replace([np.inf, -np.inf], np.nan)
        expected = X[s1_observed_cols(X)].median().to_numpy()
        np.testing.assert_allclose(s1['imputer_stats'], expected)

    def test_feature_selection_ignores_test_values(self):
        s1, s2 = self._states_for_test_variants()
        self.assertEqual(s1['selected'], s2['selected'])
        self.assertNotIn('f_const', s1['selected'])
        # Varies only in held-out S4 -> must NOT be selected (the old global
        # filter would have kept it because of S4's values).
        self.assertGreater(self.df['f_only_s4_varies'].std(), 1e-10)
        self.assertNotIn('f_only_s4_varies', s1['selected'])

    def test_scaler_params_ignore_test_values(self):
        s1, s2 = self._states_for_test_variants()
        np.testing.assert_array_equal(s1['scaler_mean'], s2['scaler_mean'])
        np.testing.assert_array_equal(s1['scaler_scale'], s2['scaler_scale'])
        # scaler mean of the training output is ~0 (fitted on training rows)
        X_tr, _, _, _, _ = preprocess_fold(self.train[FEATURES].values,
                                           self.test[FEATURES].values, FEATURES)
        np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=1e-9)

    def test_nan_and_inf_handled(self):
        changed = perturb(self.df, self.df.subject == 'S4')
        X_tr, X_te, sel, _, _ = preprocess_fold(
            changed[changed.subject != 'S4'][FEATURES].values,
            changed[changed.subject == 'S4'][FEATURES].values, FEATURES)
        self.assertTrue(np.isfinite(X_tr).all())
        self.assertTrue(np.isfinite(X_te).all())
        self.assertEqual(X_tr.shape[1], len(sel))
        self.assertEqual(X_te.shape[1], len(sel))

    def test_all_missing_training_column_excluded_and_names_aligned(self):
        train = self.train[FEATURES].copy()
        train['f_noise'] = np.nan          # entirely missing in training
        test = self.test[FEATURES].copy()  # observed in test — must not be used
        p = FoldPreprocessor(FEATURES).fit(train.values)
        self.assertNotIn('f_noise', p.selected_features_)
        self.assertIn('f_signal', p.selected_features_)
        # names line up with columns: f_signal's scaler mean == its train mean
        i = p.selected_features_.index('f_signal')
        self.assertAlmostEqual(p.scaler_.mean_[i], train['f_signal'].mean())
        out = p.transform(test.values)
        self.assertEqual(out.shape, (len(test), len(p.selected_features_)))
        self.assertTrue(np.isfinite(out).all())

    def test_test_column_entirely_missing_is_imputed_from_train(self):
        test = self.test[FEATURES].copy()
        test['f_signal'] = np.nan
        _, X_te, _, _, _ = preprocess_fold(self.train[FEATURES].values, test.values, FEATURES)
        self.assertTrue(np.isfinite(X_te).all())

    def test_all_features_removed_raises(self):
        X = np.ones((10, 3))
        X[:, 1] = np.nan
        with self.assertRaises(NoUsableFeaturesError):
            preprocess_fold(X, X, ['a', 'b', 'c'])
        with self.assertRaises(NoUsableFeaturesError):
            preprocess_fold(np.full((5, 2), np.nan), np.zeros((2, 2)), ['a', 'b'])


def s1_observed_cols(X):
    return [c for c in X.columns if X[c].notna().any()]


# ----------------------------------------------------------------------
# Pipeline-level: real LOSO functions with a recording spy
# ----------------------------------------------------------------------

class TestCompletePipelineLOSO(unittest.TestCase):

    def setUp(self):
        self.df, self.feature_cols = complete.prepare_feature_frame(make_df())

    def test_prepare_feature_frame_no_global_stats(self):
        raw = make_df()
        df, cols = complete.prepare_feature_frame(raw)
        self.assertEqual(cols, FEATURES)
        self.assertFalse(np.isinf(df[cols].to_numpy(dtype=float)).any())
        # NaNs are preserved (no global imputation) and no column dropped
        self.assertEqual(int(df[cols].isna().sum().sum()),
                         int(raw[cols].replace([np.inf, -np.inf], np.nan).isna().sum().sum()))
        self.assertIn('f_const', cols)

    def test_generic_fold_stats_ignore_held_out_subject(self):
        (_, _, _, _), base = record_calls(complete, complete.loso_generic,
                                          self.df, self.feature_cols)
        changed = perturb(self.df, self.df.subject == 'S3')
        _, new = record_calls(complete, complete.loso_generic, changed, self.feature_cols)
        self.assertEqual(len(base), len(SUBJECTS))
        k = SUBJECTS.index('S3')
        assert_states_equal(self, base[k], new[k])
        # control: S3 is in training for the other folds, so those change
        self.assertFalse(all(states_equal(base[i], new[i]) for i in range(len(SUBJECTS)) if i != k))

    def test_personalized_fold_stats_ignore_evaluation_subset(self):
        # Two preprocess_fold calls per fold: generic baseline (2k), personalized (2k+1)
        _, base = record_calls(complete, complete.loso_personalized,
                               self.df, self.feature_cols)
        self.assertEqual(len(base), 2 * len(SUBJECTS))
        k = SUBJECTS.index('S3')
        s3_idx = self.df.index[self.df.subject == 'S3']
        cal_n = max(int(len(s3_idx) * 0.3), 2)
        eval_mask = self.df.index.isin(s3_idx[cal_n:])
        _, new = record_calls(complete, complete.loso_personalized,
                              perturb(self.df, eval_mask), self.feature_cols)
        assert_states_equal(self, base[2 * k], new[2 * k])
        assert_states_equal(self, base[2 * k + 1], new[2 * k + 1])
        # control: calibration windows ARE allowed to influence the personalized
        # preprocessing, but never the generic baseline
        cal_mask = self.df.index.isin(s3_idx[:cal_n])
        _, cal_changed = record_calls(complete, complete.loso_personalized,
                                      perturb(self.df, cal_mask), self.feature_cols)
        self.assertFalse(states_equal(base[2 * k + 1], cal_changed[2 * k + 1]))
        assert_states_equal(self, base[2 * k], cal_changed[2 * k])

    def test_functions_still_run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            gen, t, p, prob = complete.loso_generic(self.df, self.feature_cols)
            per = complete.loso_personalized(self.df, self.feature_cols)
        self.assertEqual([r['subject'] for r in gen], SUBJECTS)
        self.assertEqual(len(t), len(self.df))
        self.assertEqual(len(per), len(SUBJECTS))
        for r in gen + per:
            for m in ('accuracy', 'f1', 'recall', 'precision'):
                self.assertTrue(0.0 <= r[m] <= 1.0)

    def test_generate_report_runs_with_raw_features(self):
        with contextlib.redirect_stdout(io.StringIO()):
            gen, t, p, prob = complete.loso_generic(self.df, self.feature_cols)
            per = complete.loso_personalized(self.df, self.feature_cols)
            with tempfile.TemporaryDirectory() as d:
                imp, roc_auc = complete.generate_report(
                    self.df, self.feature_cols, gen, per, t, p, prob, d)
                self.assertTrue(os.path.exists(os.path.join(d, 'fig2_feature_importance.png')))
        names = [n for n, _ in imp]
        self.assertNotIn('f_const', names)
        self.assertTrue(0.0 <= roc_auc <= 1.0)


class TestV2PipelineLOSO(unittest.TestCase):

    def setUp(self):
        df = make_df(label_col='label')
        df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
        self.df = df
        self.rf = v2.get_models()['Random Forest']

    def test_load_features_raw_keeps_nans(self):
        raw = make_df()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(v2, 'OUTPUT_DIR', d):
            raw.to_csv(os.path.join(d, 'features.csv'), index=False)
            df, cols = v2.load_features_raw()
        self.assertIn('label', df.columns)
        self.assertEqual(set(cols), set(FEATURES))
        vals = df[cols].to_numpy(dtype=float)
        self.assertFalse(np.isinf(vals).any())
        self.assertTrue(np.isnan(vals).any())

    def test_loso_evaluate_fold_stats_ignore_held_out_subject(self):
        _, base = record_calls(v2, v2.loso_evaluate, self.df, FEATURES, self.rf)
        changed = perturb(self.df, self.df.subject == 'S2')
        _, new = record_calls(v2, v2.loso_evaluate, changed, FEATURES, self.rf)
        k = SUBJECTS.index('S2')
        assert_states_equal(self, base[k], new[k])
        self.assertFalse(all(states_equal(base[i], new[i]) for i in range(len(SUBJECTS)) if i != k))

    def test_fair_personalization_ignores_evaluation_subset(self):
        _, base = record_calls(v2, v2.run_fair_personalization, self.df, FEATURES, self.rf)
        self.assertEqual(len(base), 2 * len(SUBJECTS))  # generic + personal per fold
        k = SUBJECTS.index('S2')
        s_idx = self.df.index[self.df.subject == 'S2']
        cal_n = max(int(len(s_idx) * 0.3), 2)
        eval_mask = self.df.index.isin(s_idx[cal_n:])
        _, new = record_calls(v2, v2.run_fair_personalization,
                              perturb(self.df, eval_mask), FEATURES, self.rf)
        assert_states_equal(self, base[2 * k], new[2 * k])          # generic
        assert_states_equal(self, base[2 * k + 1], new[2 * k + 1])  # personalized

    def test_all_models_still_run(self):
        for name, cfg in v2.get_models().items():
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()):
                results, acc, f1, auc, t, p, prob = v2.loso_evaluate(self.df, FEATURES, cfg)
                self.assertEqual(len(results), len(SUBJECTS))
                self.assertTrue(0.0 <= acc <= 1.0 and 0.0 <= f1 <= 1.0 and 0.0 <= auc <= 1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            res = v2.run_fair_personalization(self.df, FEATURES, self.rf)
        self.assertEqual(len(res), len(SUBJECTS))

    def test_fold_skipped_when_no_usable_features(self):
        df = self.df.copy()
        df[FEATURES] = 1.0  # zero variance everywhere
        with contextlib.redirect_stdout(io.StringIO()):
            results, *_ = v2.loso_evaluate(df, FEATURES, self.rf)
            gen, *_ = complete.loso_generic(
                df.rename(columns={'label': 'label_binary'}), FEATURES)
        self.assertEqual(results, [])
        self.assertEqual(gen, [])


if __name__ == '__main__':
    unittest.main()
