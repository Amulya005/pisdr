"""Improvement A - temporal and subgroup uncertainty reliability audit.

Question: does the healthy OVERALL conformal coverage on the future test
period (0.897 vs 0.90 nominal) hide subgroups where the prediction sets
are unreliable? Method: calibrate the conformal threshold on the
calibration period ONLY (unchanged from the main experiment), then break
test-period coverage down by chronological window, shipping mode, market,
order region, category, and predicted-risk band. Each row reports n,
empirical coverage, gap vs nominal, a Wilson 95% CI, mean set size, and
uncertain/empty shares. A subgroup is flagged when its CI excludes the
nominal level AND n >= 200 (smaller groups are reported but not judged).
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
from pisdr.uncertainty import conformal_threshold, empirical_coverage, coverage_by_group

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def main():
    df = add_time_features(clean_dataco(load_dataco("data/raw/DataCoSupplyChainDataset.csv")))
    train, cal, test = temporal_train_cal_test(df)
    model = Pipeline([("prep", dataco_preprocessor()),
                      ("gb", HistGradientBoostingClassifier(max_iter=300, random_state=42))])
    model.fit(train[FEATURES_DATACO], train[TARGET])
    p_cal = model.predict_proba(cal[FEATURES_DATACO])[:, 1]
    p_test = model.predict_proba(test[FEATURES_DATACO])[:, 1]
    q = conformal_threshold(p_cal, cal[TARGET].to_numpy(), alpha=0.10)
    y = test[TARGET].to_numpy()
    log(f"q={q:.3f} | overall test coverage {empirical_coverage(p_test, y, q):.3f} "
        f"(nominal 0.90) | n={len(test):,}")

    work = test.copy()
    work["p"] = p_test
    # Four equal-count chronological windows over the test period.
    ranks = work["order_date"].rank(method="first")
    work["time_window"] = pd.qcut(ranks, 4, labels=False)
    spans = work.groupby("time_window")["order_date"].agg(["min", "max"])
    work["time_window"] = work["time_window"].map(
        {i: f"W{i+1} {spans.loc[i,'min'].date()}..{spans.loc[i,'max'].date()}" for i in spans.index})
    work["risk_band"] = pd.cut(work["p"], [0, 0.3, 0.5, 0.7, 1.0],
                               labels=["p<0.3", "0.3-0.5", "0.5-0.7", "p>=0.7"],
                               include_lowest=True).astype(str)

    tables = []
    for dim in ["time_window", "Shipping Mode", "Market", "Order Region",
                "Category Name", "risk_band"]:
        t = coverage_by_group(work, dim, p_test, y, q)
        t.insert(0, "dimension", dim)
        tables.append(t)
        flagged = t[(t["ci_excludes_nominal"]) & (t["n"] >= 200)]
        log(f"{dim}: {len(t)} groups | flagged (CI excludes 0.90, n>=200): {len(flagged)}")
        for _, r in flagged.iterrows():
            log(f"   FLAG {r['group']}: n={r['n']:,} coverage={r['coverage']:.3f} "
                f"gap={r['gap']:+.3f} CI=({r['ci_lo']:.3f},{r['ci_hi']:.3f}) "
                f"uncertain={r['pct_uncertain']:.1%}")
    out = pd.concat(tables, ignore_index=True)
    out.to_csv("reports/coverage_audit.csv", index=False)
    log("wrote reports/coverage_audit.csv")

if __name__ == "__main__":
    main()
