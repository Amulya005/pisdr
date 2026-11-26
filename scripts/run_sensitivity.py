"""Stage: policy sensitivity analysis on REAL DataCo lots + RQ1 + drift.

1. Train the leakage-safe model on the temporal train period; calibrate
   conformal on the calibration period (as in run_dataco_experiment).
2. Aggregate TEST orders into replenishment lots (simulator.aggregate_lots)
   so simulator units match its demand parameters. Lot uncertainty = the
   worst constituent order's conformal set type.
3. Sensitivity grid over expedite_cost x daily_demand x safety_stock:
   which policy minimizes total simulated cost in each scenario?
4. RQ1 on real data: a model trained on a RANDOM split, evaluated (a) on a
   random held-out slice (the number usually reported) and (b) on the same
   temporal test set (the honest number).
5. PSI drift: train vs test for the numeric features and predicted risk.
"""
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pisdr.data import load_dataco, clean_dataco, TARGET
from pisdr.features import dataco_preprocessor, FEATURES_DATACO, add_time_features
from pisdr.splits import temporal_train_cal_test
from pisdr.evaluate import classification_metrics
from pisdr.uncertainty import conformal_threshold, classify_sets, empirical_coverage
from pisdr.simulator import aggregate_lots
from pisdr.policies import compare_policies
from pisdr.drift import population_stability_index

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def make_model():
    return Pipeline([("prep", dataco_preprocessor()),
                     ("gb", HistGradientBoostingClassifier(max_iter=300, random_state=42))])

def main():
    df = add_time_features(clean_dataco(load_dataco("data/raw/DataCoSupplyChainDataset.csv")))
    train, cal, test = temporal_train_cal_test(df)
    model = make_model().fit(train[FEATURES_DATACO], train[TARGET])
    p_cal = model.predict_proba(cal[FEATURES_DATACO])[:, 1]
    p_test = model.predict_proba(test[FEATURES_DATACO])[:, 1]
    log("temporal test metrics:", classification_metrics(test[TARGET], p_test))
    q = conformal_threshold(p_cal, cal[TARGET].to_numpy(), alpha=0.10)
    log("conformal q:", round(float(q), 3), "| coverage on test:",
        round(empirical_coverage(p_test, test[TARGET].to_numpy(), q), 3))

    # --- RQ1: random-split training, two evaluations ---
    # The random split is drawn ONLY from the pre-test pool (train +
    # calibration periods): both approaches train exclusively on
    # information available before the same untouched future test period.
    # (An earlier version permuted the full dataframe, letting the random
    # model train on test-period rows - leakage, corrected Nov 26, 2026.)
    pre_test = pd.concat([train, cal])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(pre_test))
    n_tr = len(train)  # same training size as the temporal model
    r_train = pre_test.iloc[idx[:n_tr]]
    r_test = pre_test.iloc[idx[n_tr:]]
    r_model = make_model().fit(r_train[FEATURES_DATACO], r_train[TARGET])
    log("RQ1 random-trained PR-AUC on RANDOM (pre-test) test:",
        classification_metrics(r_test[TARGET],
            r_model.predict_proba(r_test[FEATURES_DATACO])[:, 1])["pr_auc"])
    log("RQ1 random-trained PR-AUC on TEMPORAL test:",
        classification_metrics(test[TARGET],
            r_model.predict_proba(test[FEATURES_DATACO])[:, 1])["pr_auc"])

    # --- PSI drift, train vs test ---
    psi_rows = {}
    for col in ["Days for shipment (scheduled)", "Order Item Quantity",
                "Order Item Discount Rate", "Order Item Total", "Product Price"]:
        psi_rows[col] = round(population_stability_index(
            train[col].to_numpy(dtype=float), test[col].to_numpy(dtype=float)), 3)
    psi_rows["predicted_risk"] = round(population_stability_index(
        model.predict_proba(train[FEATURES_DATACO])[:, 1], p_test), 3)
    log("PSI train->test:", psi_rows)

    # --- Lots + sensitivity grid ---
    test_scored = test.copy()
    test_scored["p_late"] = p_test
    test_scored["set_type"] = classify_sets(p_test, q)
    lots = aggregate_lots(test_scored)
    log(f"lots: {len(lots):,} | median units {lots['units'].median()} | "
        f"lot late rate {lots['late'].mean():.3f} | "
        f"uncertain lots {(lots['set_type'] == 'uncertain').mean():.3f}")
    sim_df = pd.DataFrame({"late_delivery_risk": lots["late"].to_numpy(),
                           "units": lots["units"].to_numpy()})
    rows = []
    for ec, dd, ss in itertools.product([60, 120, 180, 300], [10, 25, 50], [60, 120, 240]):
        results, _ = compare_policies(
            sim_df, lots["p_late"].to_numpy(), lots["set_type"].to_numpy(),
            lots["scheduled_days"].to_numpy(),
            expedite_cost=float(ec), daily_demand=float(dd), safety_stock=float(ss))
        for policy, r in results.items():
            rows.append({"expedite_cost": ec, "daily_demand": dd, "safety_stock": ss,
                         "policy": policy, "total_cost": r["total_cost"],
                         "service_level": r["service_level"],
                         "decision_regret": r["decision_regret"]})
    grid = pd.DataFrame(rows)
    grid.to_csv("reports/sensitivity_analysis.csv", index=False)
    active = grid[grid["policy"] != "oracle"]
    wins = (active.loc[active.groupby(["expedite_cost", "daily_demand", "safety_stock"])
            ["total_cost"].idxmin()].groupby("policy").size())
    log("scenario wins (of 36) by active policy:", wins.to_dict())
    m = grid[grid["policy"] == "uncertainty_aware"].merge(
        grid[grid["policy"] == "point_prediction"],
        on=["expedite_cost", "daily_demand", "safety_stock"], suffixes=("_ua", "_pp"))
    log("uncertainty-aware cheaper than point-prediction in",
        int((m["total_cost_ua"] < m["total_cost_pp"]).sum()), "of 36 scenarios")
    best_active = active.groupby(["expedite_cost", "daily_demand", "safety_stock"])["total_cost"].min()
    na_cost = grid[grid["policy"] == "no_action"].set_index(
        ["expedite_cost", "daily_demand", "safety_stock"])["total_cost"]
    log("scenarios where the best active policy beats no-action:",
        int((best_active < na_cost).sum()), "of 36")
    print(grid.pivot_table(index=["expedite_cost", "daily_demand", "safety_stock"],
                           columns="policy", values="total_cost").round(0).to_string())
    log("wrote reports/sensitivity_analysis.csv")

if __name__ == "__main__":
    main()
