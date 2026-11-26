"""Improvement H - explainability stability across time.

One fitted model, three periods. Permutation importance (average-
precision drop) is computed on a fixed-size sample from the TRAIN,
CALIBRATION, and TEST periods separately. If the features the model
relies on change rank across periods, explanations given to operators
are period-specific, not global truths - that instability is reported
via Spearman rank correlation of the importance rankings.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from pisdr.data import load_dataco, clean_dataco, TARGET
from pisdr.features import dataco_preprocessor, FEATURES_DATACO, add_time_features
from pisdr.splits import temporal_train_cal_test

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def main():
    df = add_time_features(clean_dataco(load_dataco("data/raw/DataCoSupplyChainDataset.csv")))
    train, cal, test = temporal_train_cal_test(df)
    model = Pipeline([("prep", dataco_preprocessor()),
                      ("gb", HistGradientBoostingClassifier(max_iter=300, random_state=42))])
    model.fit(train[FEATURES_DATACO], train[TARGET])
    imps = {}
    for name, part in [("train", train), ("calibration", cal), ("test", test)]:
        s = part.sample(n=min(4000, len(part)), random_state=42)
        r = permutation_importance(model, s[FEATURES_DATACO], s[TARGET],
                                   n_repeats=5, random_state=42,
                                   scoring="average_precision")
        imps[name] = pd.Series(r.importances_mean, index=FEATURES_DATACO)
        log(f"{name} top5: " + ", ".join(f"{k} {v:.3f}" for k, v in
                                         imps[name].sort_values(ascending=False).head(5).items()))
    table = pd.DataFrame(imps)
    table["rank_train"] = table["train"].rank(ascending=False)
    table["rank_test"] = table["test"].rank(ascending=False)
    for a, b in [("train", "calibration"), ("train", "test"), ("calibration", "test")]:
        rho, _ = spearmanr(table[a], table[b])
        log(f"Spearman importance-rank correlation {a} vs {b}: {rho:.3f}")
    table.round(4).to_csv("reports/importance_stability.csv")
    log("wrote reports/importance_stability.csv")

if __name__ == "__main__":
    main()
