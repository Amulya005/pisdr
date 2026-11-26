"""Temporal validation. Time-dependent data is NEVER randomly split for the
headline results: random splits leak future information into training and
overstate performance (RQ1). Random splitting exists here only so the project
can measure exactly how large that overstatement is.

Headline evaluation uses THREE disjoint temporal periods:
  train      -> fit the model
  calibration-> fit the conformal threshold (never used for training)
  test       -> final, untouched evaluation (never used for fitting anything)
"""
import numpy as np


def temporal_split(df, date_col="order_date", test_frac=0.25):
    """Two-way temporal split (kept for simple comparisons)."""
    df = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def temporal_train_cal_test(df, date_col="order_date", cal_frac=0.20, test_frac=0.25):
    """Three-way temporal split: earliest block trains, middle block
    calibrates, latest block is the final test. The three periods are
    disjoint and strictly time-ordered."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_frac))
    cal_start = int(test_start * (1 - cal_frac / (1 - test_frac)))
    train = df.iloc[:cal_start].copy()
    cal = df.iloc[cal_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    return train, cal, test


def rolling_origin_splits(df, date_col="order_date", n_splits=3, test_frac=0.15):
    """Expanding-window splits: each fold trains on all data up to a cutoff
    and tests on the next block. Yields (train_idx, test_idx) pairs."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    test_n = int(n * test_frac)
    folds = []
    for k in range(n_splits):
        test_end = n - (n_splits - 1 - k) * test_n
        test_start = test_end - test_n
        folds.append((np.arange(0, test_start), np.arange(test_start, test_end)))
    return folds


def random_split(df, test_frac=0.25, seed=42):
    """Random split - used ONLY as the RQ1 comparison baseline."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
