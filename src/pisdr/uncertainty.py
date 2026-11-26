"""Uncertainty quantification: split-conformal prediction with the standard
finite-sample correction, calibrated ONLY on a dedicated calibration period
(never on the test period - see splits.temporal_train_cal_test).

Coverage honesty (docs/03): split-conformal's 1-alpha coverage guarantee
rests on exchangeability between calibration and test data. This project
deliberately studies temporal shift, under which exchangeability can fail -
so the project reports EMPIRICAL coverage on the future test period and its
degradation under shift, and never claims a formal guarantee under drift.

Prediction-set edge cases, all surfaced explicitly:
- both classes in set  -> genuinely uncertain; the policy layer may act.
- exactly one class    -> confident prediction.
- EMPTY set            -> the model confidently excluded BOTH classes; this
  signals an overconfident/anomalous case and is flagged separately, never
  silently treated as a normal prediction.
"""
import numpy as np


def conformal_threshold(cal_probs, cal_labels, alpha=0.10):
    """Finite-sample corrected split-conformal quantile: the
    ceil((n+1)(1-alpha))-th smallest nonconformity score. Scores:
    1 - p(true class). Standard convention: if the required rank exceeds
    the calibration sample size n, the threshold is INFINITE (every
    prediction set includes both classes) - clipping the rank to n would
    silently under-cover, so it is never done."""
    cal_probs = np.asarray(cal_probs, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=int)
    scores = np.where(cal_labels == 1, 1 - cal_probs, cal_probs)
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[max(rank, 1) - 1])


def prediction_sets(probs, q):
    """Returns (include_positive, include_negative) boolean arrays. Class c
    is included when its nonconformity score 1 - p(c) is <= q."""
    probs = np.asarray(probs, dtype=float)
    return (1 - probs) <= q, probs <= q


def classify_sets(probs, q):
    """Per-case set type: 'uncertain' (both), 'confident' (one),
    'empty' (none - overconfident/anomalous)."""
    inc_pos, inc_neg = prediction_sets(probs, q)
    both = inc_pos & inc_neg
    empty = ~inc_pos & ~inc_neg
    out = np.where(both, "uncertain", np.where(empty, "empty", "confident"))
    return out


def empirical_coverage(probs, labels, q):
    """Fraction of cases whose TRUE class is inside the prediction set."""
    inc_pos, inc_neg = prediction_sets(probs, q)
    labels = np.asarray(labels, dtype=int)
    covered = np.where(labels == 1, inc_pos, inc_neg)
    return float(covered.mean())
