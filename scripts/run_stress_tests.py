"""Improvement B - data-quality stress testing.

The model and conformal threshold are fitted ONCE on clean training and
calibration data (deployment reality: you cannot recalibrate on the
corrupted stream). Each stressor corrupts the TEST inputs only; true
labels are kept, so every number measures degradation of prediction,
calibration, coverage, and uncertainty behaviour under dirty data.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pisdr.data import load_dataco, clean_dataco, TARGET
from pisdr.features import (dataco_preprocessor, FEATURES_DATACO,
                            FEATURES_DATACO_NUMERIC, add_time_features)
from pisdr.splits import temporal_train_cal_test
from pisdr.evaluate import classification_metrics
from pisdr.uncertainty import conformal_threshold, empirical_coverage, classify_sets

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
rng = np.random.RandomState(7)

def main():
    df = add_time_features(clean_dataco(load_dataco("data/raw/DataCoSupplyChainDataset.csv")))
    train, cal, test = temporal_train_cal_test(df)
    model = Pipeline([("prep", dataco_preprocessor()),
                      ("gb", HistGradientBoostingClassifier(max_iter=300, random_state=42))])
    model.fit(train[FEATURES_DATACO], train[TARGET])
    q = conformal_threshold(model.predict_proba(cal[FEATURES_DATACO])[:, 1],
                            cal[TARGET].to_numpy(), alpha=0.10)
    y = test[TARGET].to_numpy()

    def assess(name, frame, labels):
        p = model.predict_proba(frame[FEATURES_DATACO])[:, 1]
        m = classification_metrics(labels, p)
        cov = empirical_coverage(p, labels, q)
        unc = float((classify_sets(p, q) == "uncertain").mean())
        log(f"{name}: PR-AUC {m['pr_auc']:.3f} ECE {m['calibration_error']:.3f} "
            f"coverage {cov:.3f} uncertain {unc:.1%}")
        return {"stressor": name, "n": len(frame), "pr_auc": m["pr_auc"],
                "ece": m["calibration_error"], "coverage": round(cov, 4),
                "pct_uncertain": round(unc, 4)}

    rows = [assess("baseline (clean)", test, y)]

    t1 = test.copy()  # 30% missing in every numeric feature
    for c in FEATURES_DATACO_NUMERIC:
        mask = rng.rand(len(t1)) < 0.30
        t1.loc[mask, c] = np.nan
    rows.append(assess("missing_numeric_30pct", t1, y))

    t2 = test.copy()  # 20% of shipping modes replaced by a wrong mode
    modes = t2["Shipping Mode"].unique()
    flip = rng.rand(len(t2)) < 0.20
    t2.loc[flip, "Shipping Mode"] = [rng.choice([m for m in modes if m != cur])
                                     for cur in t2.loc[flip, "Shipping Mode"]]
    rows.append(assess("shipping_mode_flip_20pct", t2, y))

    t3 = test.copy()  # 10% of categories replaced by a wrong category
    cats = t3["Category Name"].unique()
    flip = rng.rand(len(t3)) < 0.10
    t3.loc[flip, "Category Name"] = rng.choice(cats, size=int(flip.sum()))
    rows.append(assess("category_noise_10pct", t3, y))

    t4 = test.copy()  # timestamps delayed 30 days (features recomputed)
    t4["order_date"] = t4["order_date"] + pd.Timedelta(days=30)
    t4 = add_time_features(t4)
    rows.append(assess("timestamp_delay_30d", t4, y))

    dup_idx = rng.choice(len(test), size=len(test) // 10, replace=False)  # +10% duplicated rows
    t5 = pd.concat([test, test.iloc[dup_idx]], ignore_index=True)
    rows.append(assess("duplicates_plus_10pct", t5, t5[TARGET].to_numpy()))

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["d_pr_auc"] = (out["pr_auc"] - base["pr_auc"]).round(4)
    out["d_coverage"] = (out["coverage"] - base["coverage"]).round(4)
    out.to_csv("reports/stress_tests.csv", index=False)
    log("wrote reports/stress_tests.csv")

if __name__ == "__main__":
    main()
