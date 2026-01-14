# application.py
# OmniFlow-D2D : Streamlit Application (MODULE-BASED)

import os
import sys
import streamlit as st

# --------------------------------------------------
# ENSURE PROJECT ROOT IS IN PYTHON PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# CORRECT MODULE IMPORTS
# --------------------------------------------------
from modules.demand_forecasting import demand_forecasting_page
from modules.inventory_optimization import inventory_analysis
from modules.logistics_prediction import logistics_dashboard
from modules.ai_decision_engine import ai_insights

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="OmniFlow D2D",
    page_icon="📦",
    layout="wide"
)

st.title("📦 OmniFlow D2D")
st.subheader("AI-Powered Demand-to-Delivery Optimization System")

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Demand Intelligence",
        "Inventory Optimization",
        "Predictive Logistics",
        "AI Insights"
    ]
)

# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------
if menu == "Demand Intelligence":
    st.header("📈 Demand Intelligence")

    with st.spinner("Running Demand Forecasting Model..."):
        result = run_demand_forecasting()

    st.success("Demand Forecast Completed")

    st.subheader("Model Comparison")
    st.dataframe(result["model_comparison"])

    st.subheader("Forecast Preview")
    st.dataframe(result["forecast"].head())

elif menu == "Inventory Optimization":
    st.header("📦 Inventory Optimization")
    inventory_analysis()

elif menu == "Predictive Logistics":
    st.header("🚚 Predictive Logistics")
    logistics_dashboard()

elif menu == "AI Insights":
    st.header("🤖 AI Decision Engine")
    ai_insights()
