"""Improvements D + E + F - decision-quality analysis on the lot grid.

E (cost robustness / regret): decision regret vs the cost-minimizing
oracle for every policy in every one of the 36 sensitivity scenarios.
F (value of information): per scenario, value of the model's information
= cost(no action) - cost(point prediction); value of PERFECT information
= cost(no action) - cost(oracle); efficiency = model value / perfect
value where the denominator is positive.
D (recovery tail): the grid reports MEAN costs only. Here, at three
representative scenarios, each policy's total cost is re-simulated 300
times with fresh delay draws, and the P50/P90/P95/P99/max of the cost
distribution are reported - a policy can lose on average yet protect
the tail, or win on average with a dangerous tail. Both matter.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pisdr.data import load_dataco, clean_dataco, TARGET
from pisdr.features import dataco_preprocessor, FEATURES_DATACO, add_time_features
from pisdr.splits import temporal_train_cal_test
from pisdr.uncertainty import conformal_threshold, classify_sets
from pisdr.simulator import aggregate_lots, simulate_impacts
from pisdr.policies import cost_minimizing_oracle_mask

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def main():
    grid = pd.read_csv("reports/sensitivity_analysis.csv")
    # --- E: regret summary ---
    reg = (grid.groupby("policy")["decision_regret"]
           .agg(["mean", "median", "max"]).round(0)
           .rename(columns={"mean": "mean_regret", "median": "median_regret",
                            "max": "worst_regret"}))
    reg.to_csv("reports/policy_regret_summary.csv")
    log("E regret summary:\n" + reg.to_string())
    # --- F: value of information ---
    piv = grid.pivot_table(index=["expedite_cost", "daily_demand", "safety_stock"],
                           columns="policy", values="total_cost")
    voi = pd.DataFrame(index=piv.index)
    voi["value_model_info"] = piv["no_action"] - piv["point_prediction"]
    voi["value_uncertainty_info"] = piv["no_action"] - piv["uncertainty_aware"]
    voi["value_perfect_info"] = piv["no_action"] - piv["oracle"]
    ok = voi["value_perfect_info"] > 0
    voi["model_efficiency"] = np.where(ok, voi["value_model_info"] / voi["value_perfect_info"], np.nan)
    voi.reset_index().to_csv("reports/value_of_information.csv", index=False)
    log(f"F: model info has POSITIVE value in {(voi['value_model_info'] > 0).sum()}/36 scenarios; "
        f"uncertainty info in {(voi['value_uncertainty_info'] > 0).sum()}/36; "
        f"mean model efficiency where perfect info matters: {voi.loc[ok, 'model_efficiency'].mean():.2f}")

    # --- D: recovery tail at three scenarios ---
    df = add_time_features(clean_dataco(load_dataco("data/raw/DataCoSupplyChainDataset.csv")))
    train, cal, test = temporal_train_cal_test(df)
    model = Pipeline([("prep", dataco_preprocessor()),
                      ("gb", HistGradientBoostingClassifier(max_iter=300, random_state=42))])
    model.fit(train[FEATURES_DATACO], train[TARGET])
    q = conformal_threshold(model.predict_proba(cal[FEATURES_DATACO])[:, 1],
                            cal[TARGET].to_numpy(), alpha=0.10)
    test = test.copy()
    test["p_late"] = model.predict_proba(test[FEATURES_DATACO])[:, 1]
    test["set_type"] = classify_sets(test["p_late"].to_numpy(), q)
    lots = aggregate_lots(test)
    sim_df = pd.DataFrame({"late_delivery_risk": lots["late"].to_numpy(),
                           "units": lots["units"].to_numpy()})
    p = lots["p_late"].to_numpy()
    unc = (lots["set_type"] == "uncertain").to_numpy()
    scenarios = [(60.0, 50.0, 60.0), (120.0, 25.0, 120.0), (300.0, 10.0, 240.0)]
    rows = []
    for ec, dd, ss in scenarios:
        kw = dict(daily_demand=dd, safety_stock=ss)
        masks = {
            "no_action": np.zeros(len(sim_df), dtype=bool),
            "point_prediction": p >= 0.5,
            "uncertainty_aware": (p >= 0.5) | unc,
            "oracle": cost_minimizing_oracle_mask(sim_df, expedite_cost=ec, **kw),
        }
        for pol, mask in masks.items():
            totals = [simulate_impacts(sim_df, expedite_cost=ec, expedited_mask=mask,
                                       seed=1000 + r, **kw)["cost"].sum()
                      for r in range(300)]
            totals = np.array(totals)
            rows.append({"expedite_cost": ec, "daily_demand": dd, "safety_stock": ss,
                         "policy": pol, "mean_cost": round(totals.mean(), 0),
                         "p50": round(np.percentile(totals, 50), 0),
                         "p90": round(np.percentile(totals, 90), 0),
                         "p95": round(np.percentile(totals, 95), 0),
                         "p99": round(np.percentile(totals, 99), 0),
                         "max_cost": round(totals.max(), 0)})
            r = rows[-1]
            log(f"D ec={ec:.0f} dd={dd:.0f} ss={ss:.0f} {pol}: mean ${r['mean_cost']:,.0f} "
                f"P95 ${r['p95']:,.0f} P99 ${r['p99']:,.0f} max ${r['max_cost']:,.0f}")
    pd.DataFrame(rows).to_csv("reports/recovery_tail.csv", index=False)
    log("wrote reports/policy_regret_summary.csv, value_of_information.csv, recovery_tail.csv")

if __name__ == "__main__":
    main()
