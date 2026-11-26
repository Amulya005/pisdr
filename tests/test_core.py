import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pisdr.data import make_synthetic_shipments, TARGET
from pisdr.splits import (temporal_split, temporal_train_cal_test,
                          rolling_origin_splits)
from pisdr.drift import population_stability_index
from pisdr.simulator import simulate_impacts, shipment_outcomes
from pisdr.policies import compare_policies, cost_minimizing_oracle_mask
from pisdr.uncertainty import (conformal_threshold, classify_sets,
                               empirical_coverage)


def test_temporal_split_has_no_future_leakage():
    df = make_synthetic_shipments(n=500)
    train, test = temporal_split(df)
    assert train["order_date"].max() <= test["order_date"].min()


def test_three_way_split_disjoint_and_ordered():
    df = make_synthetic_shipments(n=1000)
    train, cal, test = temporal_train_cal_test(df)
    assert len(train) + len(cal) + len(test) == len(df)
    assert train["order_date"].max() <= cal["order_date"].min()
    assert cal["order_date"].max() <= test["order_date"].min()
    assert set(train.index) & set(cal.index) == set()
    assert set(cal.index) & set(test.index) == set()


def test_rolling_splits_are_expanding():
    df = make_synthetic_shipments(n=900)
    folds = rolling_origin_splits(df)
    starts = [tr.max() for tr, _ in folds]
    assert starts == sorted(starts)


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(0, 1, 2000)
    assert population_stability_index(x, x) < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(0, 1, 2000), rng.normal(1.5, 1, 2000)) > 0.25


def test_simulator_is_deterministic_with_seed():
    df = make_synthetic_shipments(n=300)
    a = simulate_impacts(df, seed=7)
    b = simulate_impacts(df, seed=7)
    assert a.equals(b)


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(3)
    n = 4000
    probs = rng.random(n)
    labels = (rng.random(n) < probs).astype(int)
    q = conformal_threshold(probs[:2000], labels[:2000], alpha=0.10)
    cov = empirical_coverage(probs[2000:], labels[2000:], q)
    assert 0.86 <= cov <= 0.94  # nominal 0.90, sampling tolerance


def test_conformal_empty_set_is_flagged():
    types = classify_sets(np.array([0.5]), q=0.3)
    assert types[0] == "empty"  # both classes excluded -> overconfident case


def test_conformal_uncertain_set_is_flagged():
    types = classify_sets(np.array([0.5]), q=0.9)
    assert types[0] == "uncertain"


def test_oracle_mask_matches_cost_rule():
    df = make_synthetic_shipments(n=400)
    out = shipment_outcomes(df)
    mask = cost_minimizing_oracle_mask(df, expedite_cost=180.0)
    assert (mask == (out["avertable_cost"].to_numpy() > 180.0)).all()


def test_oracle_is_cost_minimal_against_all_policies():
    df = make_synthetic_shipments(n=600).reset_index(drop=True)
    rng = np.random.default_rng(1)
    probs = rng.random(len(df))
    set_types = classify_sets(probs, q=0.7)
    results, _ = compare_policies(df, probs, set_types,
                                  df["scheduled_days"].to_numpy())
    oracle_cost = results["oracle"]["total_cost"]
    for name, res in results.items():
        assert res["total_cost"] >= oracle_cost - 1e-9, name
        assert abs(res["decision_regret"] - (res["total_cost"] - oracle_cost)) < 1e-6


def test_lot_aggregation():
    import pandas as pd
    from pisdr.simulator import aggregate_lots
    f = pd.DataFrame({
        "order_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        "Order Region": ["A", "A", "A"],
        "Category Name": ["X", "X", "X"],
        "Order Item Quantity": [2, 3, 5],
        "Days for shipment (scheduled)": [2, 4, 1],
        "late_delivery_risk": [0, 1, 0],
        "p_late": [0.5, 0.5, 0.2],
        "set_type": ["confident", "uncertain", "confident"],
    })
    lots = aggregate_lots(f)
    assert len(lots) == 2
    first = lots.iloc[0]
    assert first["units"] == 5 and first["scheduled_days"] == 4
    assert first["late"] == 1 and first["n_orders"] == 2
    assert abs(first["p_late"] - 0.75) < 1e-9  # 1 - (0.5 * 0.5)
    assert first["set_type"] == "uncertain"  # worst constituent set wins
