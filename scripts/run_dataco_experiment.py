"""PISDR real-data experiment on DataCo (CC0).

Design (docs/03): three disjoint temporal periods - train / calibration /
test. Features are pre-outcome only (features.EXCLUDED_DATACO documents every
exclusion). The policy comparison runs the simulator on the test period's
REALIZED outcomes: DataCo supplies predictions and realized lateness; the
operational consequences remain simulated under stated assumptions - the two
are never conflated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pisdr.data import load_dataco, clean_dataco, TARGET
from pisdr.features import add_time_features, dataco_preprocessor, FEATURES_DATACO
from pisdr.splits import temporal_train_cal_test
from pisdr.evaluate import classification_metrics
from pisdr.uncertainty import conformal_threshold, classify_sets, empirical_coverage
from pisdr.policies import compare_policies


def main():
    import time
    t0 = time.time()
    def log(*a):
        print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
    df = clean_dataco(load_dataco("data/raw/DataCoSupplyChainDataset.csv"))
    df = add_time_features(df)
    log(f"clean rows: {len(df):,} | period {df['order_date'].min()} to {df['order_date'].max()}")
    train, cal, test = temporal_train_cal_test(df)
    log(f"train {len(train):,} | calibration {len(cal):,} | test {len(test):,}")

    # HistGradientBoosting: a 300-tree random forest is impractical on this
    # dataset size / hardware (a timed-out run is documented in the project
    # log); gradient boosting is the stronger and far faster tabular model.
    model = Pipeline([("prep", dataco_preprocessor()),
                      ("gb", HistGradientBoostingClassifier(
                          max_iter=300, random_state=42))])
    model.fit(train[FEATURES_DATACO], train[TARGET])
    p_cal = model.predict_proba(cal[FEATURES_DATACO])[:, 1]
    p_test = model.predict_proba(test[FEATURES_DATACO])[:, 1]

    log("test metrics:", {k: round(v, 3) for k, v in
                             classification_metrics(test[TARGET], p_test).items()})
    q = conformal_threshold(p_cal, cal[TARGET].to_numpy(), alpha=0.10)
    cov = empirical_coverage(p_test, test[TARGET].to_numpy(), q)
    set_types = classify_sets(p_test, q)
    log(f"conformal q={q:.3f} | empirical coverage on future test: {cov:.3f} "
          f"(nominal 0.90; no guarantee claimed under shift)")

    sim_df = pd.DataFrame({
        "late_delivery_risk": test[TARGET].to_numpy(),
        "units": test["Order Item Quantity"].to_numpy(),
    })
    results, set_report = compare_policies(
        sim_df, p_test, set_types,
        test["Days for shipment (scheduled)"].to_numpy())
    log("prediction sets on test:", set_report)
    table = pd.DataFrame(results).T
    print("\nPolicy comparison (simulated consequences of REAL test outcomes):")
    print(table.to_string())
    table.to_csv("reports/dataco_policy_comparison.csv")
    print("\nWrote reports/dataco_policy_comparison.csv")


if __name__ == "__main__":
    main()
