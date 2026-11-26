"""Evaluation metrics: predictive (PR-AUC, Brier, calibration error),
uncertainty (interval coverage), reported per split design with the split
design always named alongside the number."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def classification_metrics(y_true, probs):
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "calibration_error": expected_calibration_error(y_true, probs),
    }


def expected_calibration_error(y_true, probs, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (probs > edges[b]) & (probs <= edges[b + 1])
        if mask.sum() > 0:
            ece += mask.mean() * abs(y_true[mask].mean() - probs[mask].mean())
    return float(ece)
