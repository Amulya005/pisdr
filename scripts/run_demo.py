"""End-to-end PISDR demo on SYNTHETIC data (labeled as such in its output).

Runs: feature build -> temporal vs random split comparison (RQ1) -> model
training -> conformal uncertainty -> policy comparison in the simulator
(RQ2/RQ5). Writes reports/policy_comparison.csv for the dashboard.
Real-data runs use the same code path with data.load_dataco.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pisdr.data import make_synthetic_shipments, TARGET
from pisdr.features import add_time_features, feature_matrix, FEATURES_SYNTHETIC
from pisdr.splits import temporal_split, random_split
from pisdr.models import get_models
from pisdr.evaluate import classification_metrics
from pisdr.uncertainty import conformal_threshold, conformal_predict_set
from pisdr.policies import compare_policies


def main():
    print("=== PISDR demo - SYNTHETIC data (not real-world results) ===")
    df = add_time_features(make_synthetic_shipments())
    X = feature_matrix(df, FEATURES_SYNTHETIC)
    y = df[TARGET]

    train_t, test_t = temporal_split(df)
    train_r, test_r = random_split(df)
    model = get_models()["random_forest"]
    model.fit(feature_matrix(train_t, FEATURES_SYNTHETIC), train_t[TARGET])
    p_temporal = model.predict_proba(feature_matrix(test_t, FEATURES_SYNTHETIC))[:, 1]
    model_r = get_models()["random_forest"]
    model_r.fit(feature_matrix(train_r, FEATURES_SYNTHETIC), train_r[TARGET])
    p_random = model_r.predict_proba(feature_matrix(test_r, FEATURES_SYNTHETIC))[:, 1]
    print("RQ1 check - same model, two split designs:")
    print("  temporal split PR-AUC:", round(classification_metrics(test_t[TARGET], p_temporal)["pr_auc"], 3))
    print("  random   split PR-AUC:", round(classification_metrics(test_r[TARGET], p_random)["pr_auc"], 3))

    q = conformal_threshold(p_temporal, test_t[TARGET].to_numpy(), alpha=0.10)
    uncertain = conformal_predict_set(p_temporal, q)
    results = compare_policies(test_t.reset_index(drop=True), p_temporal, uncertain,
                               test_t["scheduled_days"].to_numpy())
    table = pd.DataFrame(results).T
    print("\nPolicy comparison (simulated operational outcomes):")
    print(table.to_string())
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/policy_comparison.csv")
    print("\nWrote reports/policy_comparison.csv")


if __name__ == "__main__":
    main()
