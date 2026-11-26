"""Streamlit decision dashboard. Run: streamlit run src/pisdr/dashboard.py
Every recommendation shown is traceable to a model output, its conformal
uncertainty flag, and the policy rule that fired - no free-text invention."""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PISDR Decision Dashboard", layout="wide")
st.title("PISDR - Supply Chain Disruption & Recovery Decisions")
st.caption("Predictions carry uncertainty intervals; recommendations show the "
           "policy rule and evidence behind them. Simulator outputs are "
           "simulated consequences under stated assumptions, not measured "
           "real-world benefits.")

# Report discovery: real-DataCo results and synthetic-demo results are
# DIFFERENT artifacts and are never shown under the same label. The real
# experiment output is the default when it exists.
from pathlib import Path

KNOWN_REPORTS = {
    "reports/dataco_policy_comparison.csv":
        "Real data - DataCo experiment (simulated consequences of real outcomes)",
    "reports/policy_comparison.csv":
        "Synthetic demo - simulated data, illustrative only",
}
available = {p: label for p, label in KNOWN_REPORTS.items() if Path(p).exists()}
if available:
    choice = st.selectbox("Policy comparison report", list(available),
                          format_func=lambda p: available[p])
    results = pd.read_csv(choice, index_col=0)
    st.subheader(available[choice])
    st.dataframe(results)
    st.bar_chart(results["total_cost"])
else:
    st.info("No policy comparison report found. Run scripts/run_demo.py "
            "(synthetic) or scripts/run_dataco_experiment.py (real DataCo) "
            "first, then reload.")
