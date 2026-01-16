# ======================================================================================
# OmniFlow-D2D : Inventory Optimization Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# PURPOSE:
# - Consume Inventory master data
# - Consume Demand Forecasting output
# - Optimize stock levels
# - Detect stock-out / overstock risk
# - Recommend reorder quantity
# - Answer natural language questions using DATA ONLY
#
# IMPORTANT:
# - This file does NOT run standalone
# - It exposes ONE function: inventory_optimization_page()
# ======================================================================================

# ----------------------------------
# LIBRARIES
# ----------------------------------
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------
# PATH CONFIG (DO NOT CHANGE)
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INVENTORY_PATH = os.path.join(BASE_DIR, "data", "inventory.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "outputs", "forecast_demand.csv")

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_inventory():
    df = pd.read_csv(INVENTORY_PATH)
    return df

@st.cache_data
def load_forecast():
    df = pd.read_csv(FORECAST_PATH)
    return df

# ======================================================================================
# INVENTORY CALCULATIONS
# ======================================================================================
def compute_inventory_metrics(inv, fc):

    df = inv.merge(fc, on="product_id", how="left")

    # Safety stock (service level Z = 1.65 ~ 95%)
    df["safety_stock"] = 1.65 * (
        (df["upper_bound"] - df["lower_bound"]) / 2
    ) * np.sqrt(df["lead_time_days"])

    # Reorder quantity
    df["reorder_qty"] = (
        df["forecast_demand"] * df["lead_time_days"]
        + df["safety_stock"]
        - df["current_stock"]
    )

    df["reorder_qty"] = df["reorder_qty"].apply(lambda x: max(0, int(x)))

    # Risk classification
    def risk_flag(row):
        if row["current_stock"] < row["lower_bound"]:
            return "🔴 High Stock-out Risk"
        elif row["current_stock"] < row["forecast_demand"]:
            return "🟠 Medium Risk"
        elif row["current_stock"] > row["upper_bound"]:
            return "🟢 Overstock"
        else:
            return "🟡 Balanced"

    df["risk_status"] = df.apply(risk_flag, axis=1)

    return df

# ======================================================================================
# NLP-LIKE QUESTION ANSWERING ENGINE (NO API)
# ======================================================================================
def inventory_nlp_engine(question, df):

    q = question.lower()

    # ----------------------------
    # High risk products
    # ----------------------------
    if re.search(r"(stock.?out|low stock|risk)", q):
        risky = df[df["risk_status"] == "🔴 High Stock-out Risk"]
        if risky.empty:
            return "No products are currently at high stock-out risk."
        return (
            "High stock-out risk products:\n" +
            ", ".join(risky["product_id"].astype(str))
        )

    # ----------------------------
    # Overstock products
    # ----------------------------
    if re.search(r"(over.?stock|excess)", q):
        over = df[df["risk_status"] == "🟢 Overstock"]
        if over.empty:
            return "No products are overstocked."
        return (
            "Overstocked products:\n" +
            ", ".join(over["product_id"].astype(str))
        )

    # ----------------------------
    # Reorder quantity
    # ----------------------------
    if re.search(r"(reorder|how much to order)", q):
        top = df.sort_values("reorder_qty", ascending=False).head(5)
        return (
            "Top products requiring reorder:\n" +
            "\n".join(
                f"Product {r.product_id}: {r.reorder_qty} units"
                for _, r in top.iterrows()
            )
        )

    # ----------------------------
    # Unstable / volatile demand
    # ----------------------------
    if re.search(r"(unstable|volatile)", q):
        df["volatility"] = df["upper_bound"] - df["lower_bound"]
        unstable = df.sort_values("volatility", ascending=False).head(3)
        return (
            "Products with most unstable demand:\n" +
            "\n".join(
                f"Product {r.product_id} (volatility {int(r.volatility)})"
                for _, r in unstable.iterrows()
            )
        )

    # ----------------------------
    # Warehouse level question
    # ----------------------------
    if re.search(r"(warehouse)", q):
        wh = df.groupby("warehouse_id")["reorder_qty"].sum().reset_index()
        return (
            "Total reorder quantity by warehouse:\n" +
            "\n".join(
                f"Warehouse {r.warehouse_id}: {int(r.reorder_qty)} units"
                for _, r in wh.iterrows()
            )
        )

    # ----------------------------
    # Fallback
    # ----------------------------
    return (
        "I can answer questions like:\n"
        "• Which products are at stock-out risk?\n"
        "• Which products are overstocked?\n"
        "• How much should we reorder?\n"
        "• Which product has unstable demand?\n"
        "• Warehouse-wise inventory risk"
    )

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def inventory_optimization_page():

    st.header("📦 Inventory Optimization & Risk Intelligence")

    # ----------------------------
    # DATA VALIDATION
    # ----------------------------
    if not os.path.exists(INVENTORY_PATH):
        st.error("❌ inventory.csv not found in data/")
        st.stop()

    if not os.path.exists(FORECAST_PATH):
        st.error("❌ Run Demand Forecasting module first")
        st.stop()

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    inv = load_inventory()
    fc = load_forecast()

    df = compute_inventory_metrics(inv, fc)

    # ----------------------------
    # KPI DASHBOARD
    # ----------------------------
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

    # ----------------------------
    # INVENTORY TABLE
    # ----------------------------
    st.subheader("📋 Inventory Optimization Table")
    st.dataframe(df, width="stretch")

    # ----------------------------
    # RISK DISTRIBUTION
    # ----------------------------
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

    # ----------------------------
    # NLP QUESTION ANSWERING
    # ----------------------------
    st.divider()
    st.subheader("🤖 Inventory Intelligence Assistant")

    st.caption(
        "Try asking:\n"
        "• Which product has low stock?\n"
        "• Which product is overstocked?\n"
        "• How much should we reorder?\n"
        "• Which product has unstable demand?\n"
        "• Warehouse risk summary"
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

    st.success("✅ Inventory Optimization Completed")
