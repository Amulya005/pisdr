"""Distribution-shift measurement: Population Stability Index (PSI) per
feature between a reference window and a current window. PSI < 0.1 is
conventionally 'no material shift', 0.1-0.25 moderate, > 0.25 large. These
thresholds are heuristics and are reported as such."""
import numpy as np


def population_stability_index(expected, actual, bins=10):
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.quantile(e, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, bins=edges)[0] / max(len(e), 1)
    a_pct = np.histogram(a, bins=edges)[0] / max(len(a), 1)
    e_pct = np.clip(e_pct, 1e-4, None)
    a_pct = np.clip(a_pct, 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
