"""Feature engineering. Every feature must be knowable BEFORE the shipment
outcome is observed; anything derived from actual delivery is leakage and is
blacklisted in data.LEAKAGE_COLUMNS."""
import pandas as pd

FEATURES_SYNTHETIC = ["scheduled_days", "port_congestion", "disruption_window",
                      "order_value", "units", "month", "dow"]


def add_time_features(df):
    df = df.copy()
    df["month"] = df["order_date"].dt.month
    df["dow"] = df["order_date"].dt.dayofweek
    return df


def feature_matrix(df, features):
    return df[features].astype(float)

# --- Real DataCo features -------------------------------------------------
# Admissibility rule (docs/02): a feature must be knowable at order /
# shipment-booking time, BEFORE the delivery outcome exists. Anything derived
# from the actual shipment (real shipping days, delivery status, profit and
# benefit figures computed after fulfilment) is excluded as leakage or
# post-outcome information.
FEATURES_DATACO_NUMERIC = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Order Item Total",
    "Product Price",
    "month",
    "dow",
]
FEATURES_DATACO_CATEGORICAL = [
    "Type", "Shipping Mode", "Customer Segment", "Market",
    "Order Region", "Category Name", "Department Name",
]
FEATURES_DATACO = FEATURES_DATACO_NUMERIC + FEATURES_DATACO_CATEGORICAL

# Post-outcome / derived columns deliberately excluded, with reasons:
EXCLUDED_DATACO = {
    "Days for shipping (real)": "the realized outcome itself - direct leakage",
    "Delivery Status": "derived from realized vs scheduled dates - direct leakage",
    "shipping date (DateOrders)": "known only once shipped - post-outcome",
    "Benefit per order": "profit computed after fulfilment - post-outcome",
    "Order Profit Per Order": "profit computed after fulfilment - post-outcome",
    "Order Item Profit Ratio": "derived from post-fulfilment profit - post-outcome",
    "Sales per customer": "aggregate computed over the full dataset window - temporal leakage risk",
}


def dataco_preprocessor():
    """One-hot encode DataCo categoricals, pass numerics through."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, FEATURES_DATACO_NUMERIC),
        ("cat", categorical, FEATURES_DATACO_CATEGORICAL),
    ])
