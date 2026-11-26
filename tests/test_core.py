import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pisdr.data import make_synthetic_shipments, TARGET
from pisdr.splits import temporal_split, rolling_origin_splits
from pisdr.drift import population_stability_index
from pisdr.simulator import simulate_impacts
from pisdr.policies import compare_policies


def test_temporal_split_has_no_future_leakage():
    df = make_synthetic_shipments(n=500)
    train, test = temporal_split(df)
    assert train["order_date"].max() <= test["order_date"].min()


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


def test_oracle_never_worse_than_no_action():
    df = make_synthetic_shipments(n=600).reset_index(drop=True)
    probs = np.full(len(df), 0.3)
    uncertain = np.zeros(len(df), dtype=bool)
    res = compare_policies(df, probs, uncertain, df["scheduled_days"].to_numpy())
    assert res["oracle"]["total_cost"] <= res["no_action"]["total_cost"]
    assert res["oracle"]["decision_regret"] == 0.0
