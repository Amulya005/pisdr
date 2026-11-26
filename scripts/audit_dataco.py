"""DataCo data-quality audit. Profiles the RAW file exactly as downloaded
and writes reports/dataco_quality_audit.md with measured values only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pisdr.data import PII_COLUMNS, LEAKAGE_COLUMNS
from pisdr.features import EXCLUDED_DATACO

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")


def main():
    df = pd.read_csv(RAW, encoding="latin-1", low_memory=False)
    lines = ["# DataCo Quality Audit (measured on the raw file as downloaded)",
             "", f"- Source file: {RAW} ({RAW.stat().st_size/1e6:.1f} MB)",
             f"- Rows: {len(df):,} | Columns: {df.shape[1]}",
             f"- Exact duplicate rows: {int(df.duplicated().sum()):,}", ""]
    dates = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    lines += [f"- Order dates parseable: {int(dates.notna().sum()):,} "
              f"({dates.notna().mean():.1%}); range "
              f"{dates.min()} to {dates.max()}", ""]
    if "Late_delivery_risk" in df:
        vc = df["Late_delivery_risk"].value_counts(normalize=True).sort_index()
        lines += ["- Target balance (Late_delivery_risk): "
                  + ", ".join(f"{k}: {v:.1%}" for k, v in vc.items()), ""]
    miss = (df.isna().mean() * 100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    lines += ["## Missing values (columns with any)", ""]
    lines += [f"- {c}: {v}%" for c, v in miss.items()] or ["- none"]
    lines += ["", "## PII / credential columns present in raw file (dropped at load, never features)", ""]
    lines += [f"- {c}" for c in PII_COLUMNS if c in df.columns]
    lines += ["", "## Leakage columns (banned from features)", ""]
    lines += [f"- {c}" for c in LEAKAGE_COLUMNS if c in df.columns]
    lines += ["", "## Other post-outcome columns excluded (docs/02)", ""]
    lines += [f"- {c}: {r}" for c, r in EXCLUDED_DATACO.items() if c in df.columns]
    lines += ["", "## Anomalies", ""]
    sched = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
    real = pd.to_numeric(df["Days for shipping (real)"], errors="coerce")
    lines += [f"- Scheduled days: min {sched.min()}, max {sched.max()}, "
              f"negative values: {int((sched < 0).sum())}",
              f"- Real shipping days: min {real.min()}, max {real.max()}, "
              f"negative values: {int((real < 0).sum())}",
              f"- Rows where real < 0 or scheduled <= 0: "
              f"{int(((real < 0) | (sched <= 0)).sum())}"]
    neg_sales = int((pd.to_numeric(df['Sales'], errors='coerce') < 0).sum()) if 'Sales' in df else 0
    lines += [f"- Negative Sales rows: {neg_sales}"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dataco_quality_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
