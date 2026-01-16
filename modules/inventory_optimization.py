# ======================================================================================
# OmniFlow-D2D : Inventory Optimization & Risk Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# DESCRIPTION:
# - Consumes Inventory Master Data
# - Consumes Demand Forecasting Output
# - Calculates reorder quantity, safety stock
# - Identifies stock-out & overstock risks
# - Provides NLP-style Q&A (DATA-DRIVEN, NO API)
#
# NOTE:
# - This is a Streamlit PAGE MODULE
# - Called from application.py
# ======================================================================================

# ----------------------------------
# IMPORTS
# ----------------------------------
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------
# PATH CONFIGURATION (DO NOT CHANGE)
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INVENTORY_PATH = os.path.join(BASE_DIR, "data", "inventory.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "outputs", "forecast_demand.csv")

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_inventory():
    return pd.read_csv(INVENTORY_PATH)

@st.cache_data
def load_forecast():
    return pd.read_csv(FORECAST_PATH)

# ======================================================================================
# INVENTORY METRICS ENGINE
# ======================================================================================
def compute_inventory_metrics(inv, fc):

    df = inv.merge(fc, on="product_id", how="left")

    # Safety stock (95% service level)
    df["safety_stock"] = (
        1.65
        * ((df["upper_bound"] - df["lower_bound"]) / 2)
        * np.sqrt(df["lead_time_days"])
    )

    # Reorder quantity
    df["reorder_qty"] = (
        df["forecast_demand"] * df["lead_time_days"]
        + df["safety_stock"]
        - df["current_stock"]
    )

    df["reorder_qty"] = df["reorder_qty"].apply(lambda x: max(0, int(x)))

    # Risk classification
    def risk_level(row):
        if row["current_stock"] < row["lower_bound"]:
            return "🔴 High Stock-out Risk"
        elif row["current_stock"] < row["forecast_demand"]:
            return "🟠 Medium Risk"
        elif row["current_stock"] > row["upper_bound"]:
            return "🟢 Overstock"
        else:
            return "🟡 Balanced"

    df["risk_status"] = df.apply(risk_level, axis=1)

    return df

# ======================================================================================
# NLP-STYLE QUESTION ANSWERING (RULE + DATA BASED)
# ======================================================================================
def inventory_nlp_engine(question, df):

    q = question.lower()

    if re.search(r"(low stock|stock.?out|risk)", q):
        risky = df[df["risk_status"] == "🔴 High Stock-out Risk"]
        if risky.empty:
            return "No products are currently at high stock-out risk."
        return "High stock-out risk products: " + ", ".join(
            risky["product_id"].astype(str)
        )

    if re.search(r"(overstock|excess)", q):
        over = df[df["risk_status"] == "🟢 Overstock"]
        if over.empty:
            return "No products are currently overstocked."
        return "Overstocked products: " + ", ".join(
            over["product_id"].astype(str)
        )

    if re.search(r"(reorder|order quantity|how much)", q):
        top = df.sort_values("reorder_qty", ascending=False).head(5)
        return "\n".join(
            f"Product {r.product_id}: reorder {r.reorder_qty} units"
            for _, r in top.iterrows()
        )

    if re.search(r"(unstable|volatile)", q):
        df["volatility"] = df["upper_bound"] - df["lower_bound"]
        v = df.sort_values("volatility", ascending=False).head(3)
        return "\n".join(
            f"Product {r.product_id} has high demand volatility ({int(r.volatility)})"
            for _, r in v.iterrows()
        )

    if re.search(r"(warehouse)", q):
        wh = df.groupby("warehouse_id")["reorder_qty"].sum().reset_index()
        return "\n".join(
            f"Warehouse {r.warehouse_id}: reorder {int(r.reorder_qty)} units"
            for _, r in wh.iterrows()
        )

    return (
        "I can answer questions like:\n"
        "• Which product has low stock?\n"
        "• Which product is overstocked?\n"
        "• How much should we reorder?\n"
        "• Which product has unstable demand?\n"
        "• Warehouse-wise inventory risk"
    )

# ======================================================================================
# STREAMLIT PAGE FUNCTION
# ======================================================================================
def inventory_optimization_page():

    st.header("📦 Inventory Optimization & Risk Intelligence")

    # --------------------------------------------------
    # VALIDATION CHECKS (USER-FRIENDLY)
    # --------------------------------------------------
    if not os.path.exists(INVENTORY_PATH):
        st.error("❌ inventory.csv not found in data/ folder")
        st.stop()

    if not os.path.exists(FORECAST_PATH):
        st.warning("⚠ Demand Forecast output not found")
        st.info(
            "Please run **Demand Intelligence → Demand Forecasting** first.\n\n"
            "Inventory optimization depends on forecasted demand."
        )
        return

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    inv = load_inventory()
    fc = load_forecast()

    df = compute_inventory_metrics(inv, fc)

    # --------------------------------------------------
    # KPI DASHBOARD
    # --------------------------------------------------
    st.subheader("📊 Executive Inventory KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "High Risk Products",
        df[df["risk_status"] == "🔴 High Stock-out Risk"].shape[0]
    )
    c2.metric(
        "Overstocked Products",
        df[df["risk_status"] == "🟢 Overstock"].shape[0]
    )
    c3.metric(
        "Total Reorder Qty",
        int(df["reorder_qty"].sum())
    )
    c4.metric(
        "Avg Safety Stock",
        int(df["safety_stock"].mean())
    )

    # --------------------------------------------------
    # INVENTORY TABLE
    # --------------------------------------------------
    st.subheader("📋 Inventory Optimization Table")
    st.dataframe(df, width="stretch")

    # --------------------------------------------------
    # RISK VISUALIZATION
    # --------------------------------------------------
    fig = px.bar(
        df,
        x="product_id",
        y="reorder_qty",
        color="risk_status",
        title="Reorder Quantity by Risk Level",
        text="reorder_qty"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    # --------------------------------------------------
    # NLP QUESTION INTERFACE
    # --------------------------------------------------
    st.divider()
    st.subheader("🤖 Inventory Intelligence Assistant")

    st.caption(
        "Try asking:\n"
        "• Which product has low stock?\n"
        "• Which product is overstocked?\n"
        "• How much should we reorder?\n"
        "• Which product has unstable demand?\n"
        "• Warehouse-wise risk"
    )

    if "inventory_chat" not in st.session_state:
        st.session_state.inventory_chat = []

    user_q = st.chat_input("Ask a question about inventory...")

    if user_q:
        answer = inventory_nlp_engine(user_q, df)
        st.session_state.inventory_chat.append(
            {"q": user_q, "a": answer}
        )

    for chat in st.session_state.inventory_chat[-10:]:
        with st.chat_message("user"):
            st.write(chat["q"])
        with st.chat_message("assistant"):
            st.write(chat["a"])

    st.success("✅ Inventory Optimization Completed Successfully")
