"""Data loading for PISDR.

Real data: DataCo Smart Supply Chain (Kaggle, CC0). The loader expects the raw
CSV in data/raw/ and performs only documented, reproducible cleaning.
Synthetic data: `make_synthetic_shipments` generates a clearly labeled
synthetic shipment table with the same schema the pipeline uses, so the full
pipeline and tests run without Kaggle credentials. Synthetic results are
never reported as real-world results (see docs/03-methodology.md).
"""
import numpy as np
import pandas as pd

# Columns that leak the DataCo late-delivery label (the label is derived from
# scheduled vs. actual shipping dates). Blacklisted from features - see the
# leakage audit in docs/02-data-dictionary.md.
LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",
    "shipping date (DateOrders)",
]

TARGET = "late_delivery_risk"


def load_dataco(path):
    """Load and minimally clean the DataCo CSV. Returns a DataFrame with a
    parsed order date and a binary target column named `late_delivery_risk`."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df = df.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)
    df[TARGET] = df["Late_delivery_risk"].astype(int)
    return df


def make_synthetic_shipments(n=4000, seed=42, start="2022-01-01", periods_days=720):
    """Synthetic shipment data with a KNOWN data-generating process, used for
    pipeline development, tests, and simulator experiments only.

    Late delivery probability rises with scheduled transit days, port
    congestion, and a mid-period 'disruption window' (distribution shift),
    plus noise. Nothing here is a real-world observation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, periods_days, n), unit="D")
    scheduled = rng.integers(1, 10, n)
    congestion = np.clip(rng.normal(0.45, 0.18, n), 0, 1)
    disruption = ((dates >= "2023-01-01") & (dates <= "2023-03-31")).astype(float)
    congestion = np.clip(congestion + 0.25 * disruption, 0, 1)
    logit = -2.2 + 0.28 * scheduled + 2.4 * congestion + 0.9 * disruption \
        + rng.normal(0, 0.5, n)
    p = 1 / (1 + np.exp(-logit))
    late = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "order_date": dates,
        "scheduled_days": scheduled,
        "port_congestion": congestion.round(3),
        "disruption_window": disruption,
        "order_value": np.round(rng.lognormal(4.5, 0.8, n), 2),
        "units": rng.integers(1, 40, n),
        TARGET: late,
    }).sort_values("order_date").reset_index(drop=True)
