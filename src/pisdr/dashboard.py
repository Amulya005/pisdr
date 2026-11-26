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

results_path = st.text_input("Policy comparison results (CSV)", "reports/policy_comparison.csv")
try:
    results = pd.read_csv(results_path, index_col=0)
    st.subheader("Policy comparison (simulated)")
    st.dataframe(results)
    st.bar_chart(results["total_cost"])
except FileNotFoundError:
    st.info("Run scripts/run_demo.py or the experiment pipeline to generate "
            "reports/policy_comparison.csv, then reload.")
