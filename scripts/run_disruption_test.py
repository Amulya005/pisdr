"""Improvement I - severe disruption window test.

A disruption is INJECTED into the last 30 days of the test period: 25%
of on-time shipments in the window become late (seeded). The deployed
model and its conformal threshold are NOT refitted - that is the point:
what happens to coverage and to policy costs when the world breaks
inside a window the model never saw? Compared against the same window
with its original outcomes.
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
from pisdr.uncertainty import conformal_threshold, empirical_coverage, classify_sets
from pisdr.simulator import aggregate_lots
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

    cutoff = test["order_date"].max() - pd.Timedelta(days=30)
    in_win = (test["order_date"] >= cutoff).to_numpy()
    kw = dict(expedite_cost=120.0, daily_demand=25.0, safety_stock=120.0)
    win = test[in_win].copy()
    log(f"window {cutoff.date()}..{test['order_date'].max().date()}: {in_win.sum():,} orders; "
        f"model predicted mean p in window {test['p_late'].to_numpy()[in_win].mean():.3f} "
        f"(the deployed model cannot see the shock)")
    for dose in (0.25, 0.50):
        rng = np.random.RandomState(11)
        shocked = test[TARGET].to_numpy().copy()
        flip = in_win & (shocked == 0) & (rng.rand(len(test)) < dose)
        shocked[flip] = 1
        cov_in = empirical_coverage(test["p_late"].to_numpy()[in_win], shocked[in_win], q)
        log(f"dose {dose:.0%}: window late rate {test[TARGET].to_numpy()[in_win].mean():.3f} "
            f"-> {shocked[in_win].mean():.3f} | coverage inside window {cov_in:.3f}")
        w = win.copy(); w[TARGET] = shocked[in_win]
        lots = aggregate_lots(w)
        sim_df = pd.DataFrame({"late_delivery_risk": lots["late"].to_numpy(),
                               "units": lots["units"].to_numpy()})
        res, _ = compare_policies(sim_df, lots["p_late"].to_numpy(),
                                  lots["set_type"].to_numpy(),
                                  lots["scheduled_days"].to_numpy(), **kw)
        log(f"window lots dose {dose:.0%}: " + " | ".join(
            f"{k} ${v['total_cost']:,.0f}" for k, v in res.items()))
    cov0 = empirical_coverage(test["p_late"].to_numpy()[in_win],
                              test[TARGET].to_numpy()[in_win], q)
    log(f"baseline (no shock): coverage inside window {cov0:.3f}")

if __name__ == "__main__":
    main()
