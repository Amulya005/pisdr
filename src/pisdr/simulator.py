"""Downstream impact simulator.

No public dataset links a disruption to inventory outcomes, so impact is
estimated in a TRANSPARENT simulator with explicit assumptions (all parameters
below, all overridable from config). Simulator outputs are simulated
operational consequences - never real-world measured benefits.

Assumptions (v1, deliberately simple and stated):
- Each shipment carries `units` of one product toward a single stocking point.
- Shipments are independent: one shipment's shortfall does not change
  another's. (This is what makes the per-shipment cost comparison in
  policies.cost_minimizing_oracle_mask exactly optimal - see docs/03.)
- If a shipment is late, its units arrive `delay_days` late (drawn per event).
- Demand during the delay is constant at `daily_demand` units/day.
- A stockout occurs when delay-driven shortfall exceeds on-hand `safety_stock`.
- Costs: expediting a shipment costs `expedite_cost` and is assumed to
  prevent that shipment's lateness impact entirely; a stockout costs
  `stockout_cost_per_unit_day` per unit-day short.
"""
import numpy as np
import pandas as pd


def shipment_outcomes(df, daily_demand=25.0, safety_stock=120.0, delay_days=6,
                      stockout_cost_per_unit_day=4.0, seed=42):
    """Per-shipment lateness consequences WITHOUT any action: delay,
    stockout flag, and the stockout cost that action could avert."""
    rng = np.random.default_rng(seed)
    late = df["late_delivery_risk"].to_numpy() == 1
    units = df["units"].to_numpy(dtype=float)
    delays = np.where(late, rng.integers(1, delay_days + 1, len(df)), 0)
    shortfall = np.clip(delays * daily_demand - safety_stock, 0, None)
    shortfall_units = np.minimum(shortfall, units)
    stockout = shortfall_units > 0
    avertable_cost = np.where(stockout, shortfall_units * delays
                              * stockout_cost_per_unit_day, 0.0)
    return pd.DataFrame({"late": late, "delay_days": delays,
                         "stockout": stockout, "avertable_cost": avertable_cost},
                        index=df.index)


def simulate_impacts(df, expedite_cost=180.0, expedited_mask=None, **kw):
    """Apply a policy's expedite decisions and total the costs. Expediting a
    shipment costs `expedite_cost` and removes its stockout cost."""
    out = shipment_outcomes(df, **kw)
    if expedited_mask is None:
        expedited_mask = np.zeros(len(df), dtype=bool)
    expedited_mask = np.asarray(expedited_mask, dtype=bool)
    incurred = np.where(expedited_mask, 0.0, out["avertable_cost"].to_numpy())
    cost = np.where(expedited_mask, expedite_cost, 0.0) + incurred
    stockout = out["stockout"].to_numpy() & ~expedited_mask
    return pd.DataFrame({"late": out["late"], "expedited": expedited_mask,
                         "delay_days": out["delay_days"], "stockout": stockout,
                         "cost": cost}, index=df.index)


def aggregate_lots(frame, prob_col="p_late", method="independence"):
    """Aggregate an order stream into replenishment lots.

    Real order data (DataCo) arrives as many small orders; the simulator's
    demand parameters describe a stocking point replenished in LOTS. A lot =
    all orders sharing (order day, Order Region, Category Name). For a lot:
      units          = sum of Order Item Quantity
      scheduled_days = max scheduled days among its orders
      late (actual)  = 1 if ANY constituent order was actually late
      p_late         = probability at least one order in the lot is late:
                       method="independence" -> 1 - prod(1 - p_i);
                       method="max" -> max p_i (perfect positive dependence,
                       the Frechet lower bound). Both are reported so the
                       dependence assumption is explicit, never hidden.
    This is a modeling definition, stated here so results can be challenged.
    """
    f = frame.copy()
    f["order_day"] = pd.to_datetime(f["order_date"]).dt.floor("D")
    grouped = f.groupby(["order_day", "Order Region", "Category Name"])
    lots = grouped.agg(
        units=("Order Item Quantity", "sum"),
        scheduled_days=("Days for shipment (scheduled)", "max"),
        late=("late_delivery_risk", "max"),
        n_orders=("late_delivery_risk", "size"),
    ).reset_index()
    if method == "independence":
        # P(at least one late) under conditional independence of orders.
        p = grouped[prob_col].apply(
            lambda s: 1.0 - float(np.prod(1.0 - s.to_numpy())))
    elif method == "max":
        # Perfect positive dependence bound (Frechet lower bound for the
        # union): if one shared disruption drives all orders in the lot,
        # P(at least one late) = max p_i. Independence and max bracket the
        # plausible dependence range together with the union bound
        # min(1, sum p_i); correlated-event simulation is future work.
        p = grouped[prob_col].max()
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    lots[prob_col] = p.to_numpy()
    if "set_type" in f.columns:
        rank = {"confident": 0, "empty": 1, "uncertain": 2}
        worst = grouped["set_type"].apply(
            lambda s: max(s, key=lambda v: rank[v]))
        lots["set_type"] = worst.to_numpy()
    lots = lots.rename(columns={"order_day": "order_date"})
    return lots.sort_values("order_date").reset_index(drop=True)


def copula_any_late_probs(frame, rho=0.5, sims=200, seed=42, prob_col="p_late"):
    """Monte-Carlo P(at least one order in the lot is late) under a
    Gaussian-copula dependence model (Improvement C): each order's latent
    variable shares a lot-level factor with weight sqrt(rho), so rho = 0
    reproduces independence and rho -> 1 approaches the max bound.
    Returns a dict keyed by the lot key (order day, region, category) -
    align by key, never by position.
    Simulation only - an estimate under a STATED dependence assumption,
    not a measured quantity."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    keys = list(zip(pd.to_datetime(frame["order_date"]).dt.date,
                    frame["Order Region"], frame["Category Name"]))
    lots = {}
    for i, k in enumerate(keys):
        lots.setdefault(k, []).append(i)
    p = frame[prob_col].to_numpy(dtype=float)
    estimates = {}
    for k, idxs in lots.items():
        idxs = np.asarray(idxs)
        z = rng.standard_normal((sims, 1))
        e = rng.standard_normal((sims, len(idxs)))
        u = norm.cdf(np.sqrt(rho) * z + np.sqrt(1 - rho) * e)
        late = u < p[idxs]
        estimates[k] = float(late.any(axis=1).mean())
    return estimates
