"""Recovery-action policies compared in the simulator (RQ2, RQ4, RQ5).

- rule_based: expedite any shipment with scheduled transit >= a fixed number
  of days (a simple, understandable operations rule; ignores the model).
- point_prediction: expedite when the model's predicted probability clears a
  fixed threshold.
- uncertainty_aware: expedite when the predicted probability clears the
  threshold OR the conformal layer flags the prediction as uncertain
  (borderline cases get the protective action).

Decision regret is measured against an oracle that expedites exactly the
shipments that would otherwise stock out.
"""
import numpy as np
from .simulator import simulate_impacts


def _evaluate(df, mask, **sim_kw):
    out = simulate_impacts(df, expedited_mask=mask, **sim_kw)
    return {
        "total_cost": float(out["cost"].sum()),
        "stockouts": int(out["stockout"].sum()),
        "expedited": int(mask.sum()),
        "service_level": float(1 - out["stockout"].mean()),
    }


def oracle_mask(df, **sim_kw):
    base = simulate_impacts(df, **sim_kw)
    return (base["stockout"]).to_numpy()


def compare_policies(df, probs, uncertain_mask, scheduled_days,
                     rule_min_days=7, threshold=0.5, **sim_kw):
    results = {}
    results["rule_based"] = _evaluate(df, scheduled_days >= rule_min_days, **sim_kw)
    results["point_prediction"] = _evaluate(df, probs >= threshold, **sim_kw)
    results["uncertainty_aware"] = _evaluate(
        df, (probs >= threshold) | uncertain_mask, **sim_kw)
    results["oracle"] = _evaluate(df, oracle_mask(df, **sim_kw), **sim_kw)
    results["no_action"] = _evaluate(df, np.zeros(len(df), dtype=bool), **sim_kw)
    oracle_cost = results["oracle"]["total_cost"]
    for name, res in results.items():
        res["decision_regret"] = float(res["total_cost"] - oracle_cost)
    return results
