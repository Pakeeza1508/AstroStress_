"""
AstroStress — fold-local preprocessing (shared by both pipelines).

Every statistic used to transform data (imputation medians, the variance-based
feature mask, scaler mean/std) is learned in ``FoldPreprocessor.fit`` from the
rows passed to it, and nothing else. Callers must pass ONLY the training rows of a
fold (for personalization: other subjects + the permitted calibration windows),
then call ``transform`` on held-out rows.

Steps, in order:
  1. +/-inf -> NaN (stateless, row-wise; safe before or after splitting)
  2. Drop columns that are entirely missing in the training rows
  3. Median imputation (SimpleImputer fitted on training rows)
  4. Near-zero-variance filter (std computed on imputed training rows)
  5. StandardScaler fitted on training rows
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Same threshold the original pipelines used for `std() > 1e-10`.
VARIANCE_THRESHOLD = 1e-10


class NoUsableFeaturesError(ValueError):
    """Raised when no feature survives fold-local cleaning on the training rows."""


def replace_inf_with_nan(X):
    """Return a float ndarray copy of X with +/-inf replaced by NaN.

    Stateless and element-wise, so it uses no information from other rows.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy()
    X = np.array(X, dtype=float, copy=True)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X[~np.isfinite(X)] = np.nan
    return X


class FoldPreprocessor:
    """Impute -> variance filter -> scale, fitted on training rows only."""

    def __init__(self, feature_names, variance_threshold=VARIANCE_THRESHOLD):
        self.feature_names = list(feature_names)
        self.variance_threshold = variance_threshold

    def fit(self, X_train):
        X = replace_inf_with_nan(X_train)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"X has {X.shape[1]} columns but {len(self.feature_names)} "
                             f"feature names were given")
        if X.shape[0] == 0:
            raise NoUsableFeaturesError("No training rows to fit preprocessing on")

        # 2. Columns with no observed training value cannot be imputed from
        #    training data, so they are excluded for this fold. (SimpleImputer
        #    would otherwise drop them silently and misalign feature names.)
        self.observed_mask_ = ~np.all(np.isnan(X), axis=0)
        if not self.observed_mask_.any():
            raise NoUsableFeaturesError("All feature columns are missing in the training rows")

        # 3. Median imputation from training rows only.
        self.imputer_ = SimpleImputer(strategy='median')
        X_imp = self.imputer_.fit_transform(X[:, self.observed_mask_])

        # 4. Variance filter from imputed training rows only.
        self.train_std_ = np.std(X_imp, axis=0)
        keep_observed = self.train_std_ > self.variance_threshold
        if not keep_observed.any():
            raise NoUsableFeaturesError(
                "All features have near-zero variance in the training rows")
        self.keep_observed_ = keep_observed

        # Mask over the ORIGINAL feature columns.
        self.keep_mask_ = np.zeros(len(self.feature_names), dtype=bool)
        self.keep_mask_[np.flatnonzero(self.observed_mask_)[keep_observed]] = True
        self.selected_features_ = [f for f, k in zip(self.feature_names, self.keep_mask_) if k]

        # 5. Scaler from training rows only.
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_imp[:, keep_observed])
        return self

    def transform(self, X):
        X = replace_inf_with_nan(X)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"X has {X.shape[1]} columns but {len(self.feature_names)} "
                             f"feature names were given")
        X_imp = self.imputer_.transform(X[:, self.observed_mask_])
        return self.scaler_.transform(X_imp[:, self.keep_observed_])

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)


TIMING_COLUMNS = ('window_start_s', 'window_end_s')


def calibration_split(subj_df, cal_fraction=0.3, min_calibration=2, purge_if_untimed=1):
    """Chronological calibration/evaluation split of ONE subject's windows.

    The first `cal_fraction` of the subject's windows (at least
    `min_calibration`) are the calibration set. Evaluation windows that share
    raw signal with any calibration window are dropped ("purged"):

    - If the frame has `window_start_s`/`window_end_s` (written by
      astrostress_complete.py), windows are ordered by start time and every
      candidate evaluation window starting before the last calibration window
      ends is purged — exact, even when some windows were skipped.
    - Otherwise row order is taken as chronological and the first
      `purge_if_untimed` candidate windows are purged (1 = the default
      120 s window / 60 s step, where only the adjacent window overlaps).

    Returns (calibration_df, evaluation_df, n_purged).
    """
    timed = all(c in subj_df.columns for c in TIMING_COLUMNS)
    df = subj_df.sort_values(TIMING_COLUMNS[0], kind='stable') if timed else subj_df
    n_cal = max(int(len(df) * cal_fraction), min_calibration)
    cal, rest = df.iloc[:n_cal], df.iloc[n_cal:]
    if timed:
        evaluation = rest[rest[TIMING_COLUMNS[0]] >= cal[TIMING_COLUMNS[1]].max()]
    else:
        evaluation = rest.iloc[purge_if_untimed:]
    return cal, evaluation, len(rest) - len(evaluation)


def predict_stress_proba(artifact, features_df):
    """Stress probability from a saved model artifact (see astrostress_v2.py).

    `features_df` must contain every column in artifact['feature_cols'] (by
    name; extra columns are ignored and order does not matter). Raises
    ValueError on missing columns instead of silently misaligning features.
    """
    cols = artifact['feature_cols']
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for this model: {missing}")
    X = artifact['preprocessor'].transform(features_df[cols].to_numpy(dtype=float))
    return artifact['model'].predict_proba(X)[:, 1]


def preprocess_fold(X_train, X_test, feature_cols):
    """Preprocessing INSIDE a fold: impute, filter near-zero-variance, scale.

    All statistics are fitted on X_train only and applied to X_test.

    Returns:
        X_train_clean, X_test_clean, selected_feature_names, scaler, imputer

    The returned imputer was fitted on the columns that had at least one
    observed training value (``FoldPreprocessor.observed_mask_``).
    Raises NoUsableFeaturesError if no feature survives cleaning.
    """
    prep = FoldPreprocessor(feature_cols).fit(X_train)
    X_train_clean = prep.transform(X_train)
    X_test_clean = prep.transform(X_test)
    return X_train_clean, X_test_clean, prep.selected_features_, prep.scaler_, prep.imputer_
