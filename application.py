import streamlit as st
import demand_forecasting
from modules.inventory_optimization import inventory_analysis
from modules.logistics_prediction import logistics_dashboard
from modules.ai_decision_engine import ai_insights

st.set_page_config(
    page_title="OmniFlow D2D",
    page_icon="📦",
    layout="wide"
)

st.title("📦 OmniFlow D2D")
st.subheader("AI-Powered Demand-to-Delivery Optimization System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Demand Intelligence",
        "Inventory Optimization",
        "Predictive Logistics",
        "AI Insights"
    ]
)

if menu == "Demand Intelligence":
    demand_forecasting()

elif menu == "Inventory Optimization":
    inventory_analysis()

elif menu == "Predictive Logistics":
    logistics_dashboard()

elif menu == "AI Insights":
    ai_insights()

