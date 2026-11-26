"""End-to-end PISDR demo on SYNTHETIC data (labeled as such in its output).

Evaluation design (docs/03): three disjoint temporal periods -
train (fit) / calibration (conformal threshold only) / test (final,
untouched). Reports empirical conformal coverage on the future test period,
split by the injected disruption window (shift) vs. normal periods, then
compares recovery policies in the simulator. Real-data runs use the same
code path with data.load_dataco.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pisdr.data import make_synthetic_shipments, TARGET
from pisdr.features import add_time_features, feature_matrix, FEATURES_SYNTHETIC
from pisdr.splits import temporal_train_cal_test, random_split
from pisdr.models import get_models
from pisdr.evaluate import classification_metrics
from pisdr.uncertainty import conformal_threshold, classify_sets, empirical_coverage
from pisdr.policies import compare_policies


def main():
    print("=== PISDR demo - SYNTHETIC data (not real-world results) ===")
    df = add_time_features(make_synthetic_shipments())
    train, cal, test = temporal_train_cal_test(df)
    print(f"periods: train {len(train)} | calibration {len(cal)} | test {len(test)}")

    model = get_models()["random_forest"]
    model.fit(feature_matrix(train, FEATURES_SYNTHETIC), train[TARGET])
    p_cal = model.predict_proba(feature_matrix(cal, FEATURES_SYNTHETIC))[:, 1]
    p_test = model.predict_proba(feature_matrix(test, FEATURES_SYNTHETIC))[:, 1]

    q = conformal_threshold(p_cal, cal[TARGET].to_numpy(), alpha=0.10)
    set_types = classify_sets(p_test, q)
    cov_all = empirical_coverage(p_test, test[TARGET].to_numpy(), q)
    in_shift = test["disruption_window"].to_numpy() == 1
    cov_shift = empirical_coverage(p_test[in_shift], test[TARGET].to_numpy()[in_shift], q) if in_shift.any() else float("nan")
    cov_norm = empirical_coverage(p_test[~in_shift], test[TARGET].to_numpy()[~in_shift], q) if (~in_shift).any() else float("nan")
    print(f"conformal q={q:.3f} (calibration period only)")
    print(f"empirical coverage on future test: {cov_all:.3f} overall | "
          f"{cov_shift:.3f} in disruption window | {cov_norm:.3f} normal periods")
    print("NOTE: coverage is reported empirically; no formal guarantee is "
          "claimed under temporal shift (docs/03).")
    print("test metrics:", {k: round(v, 3) for k, v in
                             classification_metrics(test[TARGET], p_test).items()})

    train_r, test_r = random_split(df)
    model_r = get_models()["random_forest"]
    model_r.fit(feature_matrix(train_r, FEATURES_SYNTHETIC), train_r[TARGET])
    p_rand = model_r.predict_proba(feature_matrix(test_r, FEATURES_SYNTHETIC))[:, 1]
    print("RQ1 check - random-split PR-AUC:",
          round(classification_metrics(test_r[TARGET], p_rand)["pr_auc"], 3),
          "vs temporal test PR-AUC:",
          round(classification_metrics(test[TARGET], p_test)["pr_auc"], 3))

    results, set_report = compare_policies(
        test.reset_index(drop=True), p_test, set_types,
        test["scheduled_days"].to_numpy())
    print("prediction sets on test:", set_report)
    table = pd.DataFrame(results).T
    print("\nPolicy comparison (simulated operational outcomes):")
    print(table.to_string())
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/policy_comparison.csv")
    print("\nWrote reports/policy_comparison.csv")


if __name__ == "__main__":
    main()
