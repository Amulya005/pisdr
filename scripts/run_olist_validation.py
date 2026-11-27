"""Improvement J - independent-dataset validation on Olist (Brazil).

Compatibility check first: Olist orders carry a purchase timestamp, an
ESTIMATED delivery date (known at purchase time) and the ACTUAL delivery
date, so late = delivered after estimated is definable without leakage,
and order-time features exist (basket size, price, freight, promised
lead time, customer state, product category/weight). Delivered orders
only. This is a genuine out-of-distribution replication: different
country, marketplace, period (2016-2018), and a much lower late rate
than DataCo. Same protocol: chronological train/calibration/test split,
HistGradientBoosting, conformal threshold from the calibration period
only, coverage judged on the untouched future test period.
Data: HuggingFace mirror aviahYadler/Olist_Ecommerce_Dataset of the
Kaggle Olist public dataset (CC BY-NC-SA 4.0); files are gitignored.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pisdr.evaluate import classification_metrics
from pisdr.uncertainty import conformal_threshold, empirical_coverage, classify_sets

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
D = Path("data/external/olist")

def main():
    orders = pd.read_csv(D / "olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp",
                                      "order_delivered_customer_date",
                                      "order_estimated_delivery_date"])
    orders = orders[(orders["order_status"] == "delivered")
                    & orders["order_delivered_customer_date"].notna()].copy()
    orders["late"] = (orders["order_delivered_customer_date"]
                      > orders["order_estimated_delivery_date"]).astype(int)
    items = pd.read_csv(D / "olist_order_items_dataset.csv")
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        n_sellers=("seller_id", "nunique")).reset_index()
    first_item = items.sort_values("order_item_id").drop_duplicates("order_id")[["order_id", "product_id"]]
    products = pd.read_csv(D / "olist_products_dataset.csv")[
        ["product_id", "product_category_name", "product_weight_g"]]
    customers = pd.read_csv(D / "olist_customers_dataset.csv")[
        ["customer_id", "customer_state"]]
    df = (orders.merge(agg, on="order_id").merge(first_item, on="order_id")
          .merge(products, on="product_id", how="left")
          .merge(customers, on="customer_id", how="left"))
    df["est_lead_days"] = (df["order_estimated_delivery_date"]
                           - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["month"] = df["order_purchase_timestamp"].dt.month
    df["dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["hour"] = df["order_purchase_timestamp"].dt.hour
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    log(f"orders: {len(df):,} | late rate {df['late'].mean():.3f} | "
        f"{df['order_purchase_timestamp'].min().date()}..{df['order_purchase_timestamp'].max().date()}")

    num = ["est_lead_days", "n_items", "total_price", "total_freight",
           "n_sellers", "product_weight_g", "month", "dow", "hour"]
    cat = ["customer_state", "product_category_name"]
    X, y = df[num + cat], df["late"].to_numpy()
    n = len(df); n_tr, n_cal = int(n * 0.55), int(n * 0.75)
    # Date-cut boundaries (same rule as pisdr.splits): no purchase
    # timestamp appears on both sides of a boundary.
    t_tr = df["order_purchase_timestamp"].iloc[n_tr]
    t_cal = df["order_purchase_timestamp"].iloc[n_cal]
    itr = np.where((df["order_purchase_timestamp"] < t_tr).to_numpy())[0]
    ical = np.where(((df["order_purchase_timestamp"] >= t_tr)
                     & (df["order_purchase_timestamp"] < t_cal)).to_numpy())[0]
    ite = np.where((df["order_purchase_timestamp"] >= t_cal).to_numpy())[0]
    n_tr, n_cal = len(itr), len(itr) + len(ical)
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    model = Pipeline([("prep", prep),
                      ("gb", HistGradientBoostingClassifier(max_iter=300, random_state=42))])
    model.fit(X.iloc[:n_tr], y[:n_tr])
    p_cal = model.predict_proba(X.iloc[n_tr:n_cal])[:, 1]
    p_test = model.predict_proba(X.iloc[n_cal:])[:, 1]
    q = conformal_threshold(p_cal, y[n_tr:n_cal], alpha=0.10)
    m = classification_metrics(y[n_cal:], p_test)
    cov = empirical_coverage(p_test, y[n_cal:], q)
    unc = float((classify_sets(p_test, q) == "uncertain").mean())
    log(f"train {n_tr:,} cal {n_cal - n_tr:,} test {n - n_cal:,}")
    log(f"test metrics: PR-AUC {m['pr_auc']:.3f} ROC-AUC {m['roc_auc']:.3f} "
        f"Brier {m['brier']:.3f} ECE {m['calibration_error']:.3f}")
    log(f"conformal q={q:.3f} | test coverage {cov:.3f} (nominal 0.90) | uncertain {unc:.1%}")

if __name__ == "__main__":
    main()
