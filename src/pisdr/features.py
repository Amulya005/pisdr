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
