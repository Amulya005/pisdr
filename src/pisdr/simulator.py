"""Downstream impact simulator.

No public dataset links a disruption to inventory outcomes, so impact is
estimated in a TRANSPARENT simulator with explicit assumptions (all parameters
below, all overridable from config). Simulator outputs are simulated
operational consequences - never real-world measured benefits.

Assumptions (v1, deliberately simple and stated):
- Each shipment carries `units` of one product toward a single stocking point.
- If a shipment is late, its units arrive `delay_days` late (drawn per event).
- Demand during the delay is constant at `daily_demand` units/day.
- A stockout occurs when delay-driven shortfall exceeds on-hand `safety_stock`.
- Costs: expediting a shipment costs `expedite_cost`; a stockout costs
  `stockout_cost_per_unit_day` per unit-day short.
"""
import numpy as np
import pandas as pd


def simulate_impacts(df, daily_demand=25.0, safety_stock=120.0, delay_days=6,
                     expedite_cost=180.0, stockout_cost_per_unit_day=4.0,
                     expedited_mask=None, seed=42):
    """Returns per-shipment simulated outcomes. `expedited_mask` marks
    shipments the active policy chose to expedite; expediting is assumed to
    prevent the lateness impact entirely at `expedite_cost` each."""
    rng = np.random.default_rng(seed)
    late = df["late_delivery_risk"].to_numpy() == 1
    units = df["units"].to_numpy(dtype=float)
    if expedited_mask is None:
        expedited_mask = np.zeros(len(df), dtype=bool)
    delays = np.where(late, rng.integers(1, delay_days + 1, len(df)), 0)
    shortfall = np.clip(delays * daily_demand - safety_stock, 0, None)
    shortfall_units = np.minimum(shortfall, units)
    stockout = (shortfall_units > 0) & late & ~expedited_mask
    cost = np.where(expedited_mask, expedite_cost, 0.0) \
        + np.where(stockout, shortfall_units * delays * stockout_cost_per_unit_day, 0.0)
    return pd.DataFrame({
        "late": late, "expedited": expedited_mask, "delay_days": delays,
        "stockout": stockout, "cost": cost,
    }, index=df.index)
