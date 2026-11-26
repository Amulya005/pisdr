"""Recovery-action policies compared in the simulator (RQ2, RQ4, RQ5).

- rule_based: expedite any shipment with scheduled transit >= a fixed number
  of days (a simple, understandable operations rule; ignores the model).
- point_prediction: expedite when the model's predicted probability clears a
  fixed threshold.
- uncertainty_aware: expedite when the predicted probability clears the
  threshold OR the conformal layer flags the case as uncertain (both classes
  in the prediction set). Empty prediction sets are reported, not acted on
  silently - see compare_policies output.
- cost_minimizing_oracle: knows the simulated outcomes and, per shipment,
  expedites exactly when the expedite cost is LOWER than the stockout cost
  it averts. Under the simulator's independence assumption this is the true
  minimum-cost action set, so decision regret against it is well defined.
"""
import numpy as np
from .simulator import simulate_impacts, shipment_outcomes


def _evaluate(df, mask, expedite_cost=180.0, **sim_kw):
    out = simulate_impacts(df, expedite_cost=expedite_cost,
                           expedited_mask=mask, **sim_kw)
    return {
        "total_cost": float(out["cost"].sum()),
        "stockouts": int(out["stockout"].sum()),
        "expedited": int(np.asarray(mask).sum()),
        "service_level": float(1 - out["stockout"].mean()),
    }


def cost_minimizing_oracle_mask(df, expedite_cost=180.0, **sim_kw):
    """Expedite exactly the shipments whose avertable stockout cost exceeds
    the cost of expediting them."""
    out = shipment_outcomes(df, **sim_kw)
    return (out["avertable_cost"].to_numpy() > expedite_cost)


def compare_policies(df, probs, set_types, scheduled_days,
                     rule_min_days=7, threshold=0.5, expedite_cost=180.0,
                     **sim_kw):
    uncertain_mask = np.asarray(set_types) == "uncertain"
    results = {}
    results["rule_based"] = _evaluate(df, scheduled_days >= rule_min_days,
                                      expedite_cost, **sim_kw)
    results["point_prediction"] = _evaluate(df, probs >= threshold,
                                            expedite_cost, **sim_kw)
    results["uncertainty_aware"] = _evaluate(
        df, (probs >= threshold) | uncertain_mask, expedite_cost, **sim_kw)
    results["oracle"] = _evaluate(
        df, cost_minimizing_oracle_mask(df, expedite_cost, **sim_kw),
        expedite_cost, **sim_kw)
    results["no_action"] = _evaluate(df, np.zeros(len(df), dtype=bool),
                                     expedite_cost, **sim_kw)
    oracle_cost = results["oracle"]["total_cost"]
    for res in results.values():
        res["decision_regret"] = float(res["total_cost"] - oracle_cost)
    return results, {"empty_sets": int((np.asarray(set_types) == "empty").sum()),
                     "uncertain_sets": int(uncertain_mask.sum())}
