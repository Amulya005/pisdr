"""Uncertainty quantification: split-conformal prediction sets for the
classifier and calibration measurement. The project distinguishes uncertainty
in predictions from uncertainty in operational assumptions (docs/03)."""
import numpy as np


def conformal_threshold(cal_probs, cal_labels, alpha=0.1):
    """Split-conformal: returns the score quantile that guarantees ~1-alpha
    coverage of the true class under exchangeability."""
    scores = np.where(cal_labels == 1, 1 - cal_probs, cal_probs)
    q = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores),
                    method="higher")
    return float(q)


def conformal_predict_set(probs, q):
    """A shipment is 'flagged uncertain' when neither class clears the
    conformal threshold - the policy layer treats these as high-uncertainty."""
    return (probs < q) & ((1 - probs) < q)


def interval_coverage(probs, labels, q):
    """Empirical coverage: fraction of cases where the true class was inside
    the prediction set (i.e., not confidently excluded)."""
    excluded_positive = (labels == 1) & (probs < (1 - q))
    excluded_negative = (labels == 0) & ((1 - probs) < (1 - q))
    return float(1 - (excluded_positive | excluded_negative).mean())
