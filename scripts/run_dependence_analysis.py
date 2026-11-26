"""Improvement C - dependence-aware lot aggregation.

aggregate_lots() brackets lot risk with independence (1 - prod(1 - p_i))
and the perfect-dependence bound (max p_i). This script fills the space
between them with a Gaussian-copula Monte-Carlo estimate of
P(at least one order in the lot is late) at rho = 0 / 0.3 / 0.6 / 0.9,
then checks whether the aggregation assumption changes DECISIONS:
policy costs at one mid-grid scenario under independence vs rho = 0.6
aggregation, both evaluated on the true lot outcomes.
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
from pisdr.simulator import aggregate_lots, copula_any_late_probs
from pisdr.policies import compare_policies

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def main():
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

    ind = aggregate_lots(test, method="independence")
    mx = aggregate_lots(test, method="max")
    frame = ind.copy()
    frame["p_max"] = mx["p_late"].to_numpy()
    lot_keys = list(zip(pd.to_datetime(frame["order_date"]).dt.date,
                        frame["Order Region"], frame["Category Name"]))
    for rho in (0.0, 0.3, 0.6, 0.9):
        est = copula_any_late_probs(test, rho=rho, sims=200)
        frame[f"p_copula_{rho}"] = [est[k] for k in lot_keys]
        log(f"rho={rho}: mean lot P(any late) copula {frame[f'p_copula_{rho}'].mean():.3f} "
            f"| independence {frame['p_late'].mean():.3f} | max {frame['p_max'].mean():.3f} "
            f"| true lot late rate {frame['late'].mean():.3f}")

    # Decision consequence at one mid-grid scenario.
    kw = dict(expedite_cost=120.0, daily_demand=25.0, safety_stock=120.0)
    rows = []
    sim_df = pd.DataFrame({"late_delivery_risk": frame["late"].to_numpy(),
                           "units": frame["units"].to_numpy()})
    for label, col in [("independence", "p_late"), ("copula_rho_0.6", "p_copula_0.6")]:
        res, _info = compare_policies(sim_df, frame[col].to_numpy(),
                               frame["set_type"].to_numpy(),
                               frame["scheduled_days"].to_numpy(), **kw)
        for pol, r in res.items():
            rows.append({"aggregation": label, "policy": pol, **r})
            log(f"{label} / {pol}: cost ${r['total_cost']:,.0f} expedited {r['expedited']}")
    pd.DataFrame(rows).to_csv("reports/dependence_policy_check.csv", index=False)
    frame.to_csv("reports/dependence_lot_probs.csv", index=False)
    log("wrote reports/dependence_lot_probs.csv + dependence_policy_check.csv")

if __name__ == "__main__":
    main()
