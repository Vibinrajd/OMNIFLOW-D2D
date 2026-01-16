# ======================================================================================
# OmniFlow-D2D : Inventory Optimization & NLP Decision Engine
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# MODULE TYPE : Streamlit Page Module
# PURPOSE :
# - Inventory optimization using demand forecast
# - Safety stock, ROP, EOQ
# - Stock risk classification
# - NLP-style question answering using pure data logic
#
# NO EXTERNAL APIs
# NO LLMs
# FULLY DATA-DRIVEN
# ======================================================================================

# ----------------------------------
# IMPORTS
# ----------------------------------
import os
import re
import math
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ----------------------------------
# CONFIG
# ----------------------------------
DATA_PATH = "data/sales.csv"            # same dataset you already use
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SERVICE_LEVEL_Z = 1.65   # ~95% service level
DEFAULT_LEAD_TIME = 7    # days
DEFAULT_HOLDING_COST = 2
DEFAULT_ORDERING_COST = 500

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "product_id",
        "daily_sales",
        "avg_demand",
        "demand_std",
        "volatility",
        "lead_time",
        "safety_stock",
        "reorder_point",
        "economic_order_qty",
        "inventory_status"
    ],
    "Description": [
        "Unique product identifier",
        "Daily units sold",
        "Average daily demand",
        "Demand standard deviation",
        "Demand volatility (std / mean)",
        "Supplier lead time (days)",
        "Buffer stock to avoid stock-out",
        "Stock level to trigger reorder",
        "Optimal order quantity",
        "Inventory health classification"
    ]
})

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# ======================================================================================
# INVENTORY METRICS ENGINE
# ======================================================================================
def build_inventory_metrics(df):

    metrics = []

    for pid, g in df.groupby("product_id"):
        avg_demand = g["daily_sales"].mean()
        std_demand = g["daily_sales"].std()
        volatility = std_demand / avg_demand if avg_demand > 0 else 0

        lead_time = DEFAULT_LEAD_TIME

        safety_stock = SERVICE_LEVEL_Z * std_demand * math.sqrt(lead_time)
        reorder_point = (avg_demand * lead_time) + safety_stock

        annual_demand = avg_demand * 365
        eoq = math.sqrt(
            (2 * annual_demand * DEFAULT_ORDERING_COST)
            / DEFAULT_HOLDING_COST
        )

        metrics.append({
            "product_id": pid,
            "avg_demand": round(avg_demand, 2),
            "demand_std": round(std_demand, 2),
            "volatility": round(volatility, 3),
            "lead_time": lead_time,
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "economic_order_qty": round(eoq, 2)
        })

    inv_df = pd.DataFrame(metrics)

    # Risk classification
    conditions = []
    for _, r in inv_df.iterrows():
        if r["volatility"] > 0.5:
            conditions.append("🔴 Unstable Demand")
        elif r["volatility"] > 0.3:
            conditions.append("🟠 Moderate Risk")
        else:
            conditions.append("🟢 Stable")

    inv_df["inventory_status"] = conditions
    return inv_df

# ======================================================================================
# NLP INTENT CLASSIFIER
# ======================================================================================
def detect_intent(question):

    q = question.lower()

    if re.search(r"unstable|volatile|volatility", q):
        return "VOLATILITY"

    if re.search(r"less stock|low stock|stock out|risk", q):
        return "STOCK_RISK"

    if re.search(r"highest demand|top demand|max demand", q):
        return "MAX_DEMAND"

    if re.search(r"lowest demand|least demand|min demand", q):
        return "MIN_DEMAND"

    if re.search(r"reorder|rop", q):
        return "REORDER"

    if re.search(r"eoq|order quantity", q):
        return "EOQ"

    if re.search(r"average|mean", q):
        return "AVERAGE"

    return "GENERAL"

# ======================================================================================
# NLP ENTITY EXTRACTOR
# ======================================================================================
def extract_product_id(question, products):
    for pid in products:
        if str(pid) in question:
            return pid
    return None

# ======================================================================================
# NLP ANSWER ENGINE (CORE LOGIC)
# ======================================================================================
def answer_question(question, inv_df):

    intent = detect_intent(question)
    product_ids = inv_df["product_id"].tolist()
    pid = extract_product_id(question, product_ids)

    # -------------------------------
    if intent == "VOLATILITY":
        row = inv_df.sort_values("volatility", ascending=False).iloc[0]
        return (
            f"Product {row.product_id} has the most unstable demand.\n\n"
            f"• Average demand: {row.avg_demand} units\n"
            f"• Demand volatility: {row.volatility}\n"
            f"• Inventory status: {row.inventory_status}"
        )

    # -------------------------------
    if intent == "MAX_DEMAND":
        row = inv_df.sort_values("avg_demand", ascending=False).iloc[0]
        return (
            f"Product {row.product_id} has the highest demand.\n\n"
            f"• Average demand: {row.avg_demand} units/day"
        )

    # -------------------------------
    if intent == "MIN_DEMAND":
        row = inv_df.sort_values("avg_demand").iloc[0]
        return (
            f"Product {row.product_id} has the lowest demand.\n\n"
            f"• Average demand: {row.avg_demand} units/day"
        )

    # -------------------------------
    if intent == "REORDER":
        if pid:
            r = inv_df[inv_df.product_id == pid].iloc[0]
            return (
                f"Reorder point for Product {pid}:\n\n"
                f"• Reorder Point (ROP): {r.reorder_point} units\n"
                f"• Safety Stock: {r.safety_stock} units"
            )
        else:
            return "Please specify a product ID to calculate reorder point."

    # -------------------------------
    if intent == "EOQ":
        if pid:
            r = inv_df[inv_df.product_id == pid].iloc[0]
            return (
                f"Economic Order Quantity for Product {pid}:\n\n"
                f"• EOQ: {r.economic_order_qty} units"
            )
        else:
            return "Please specify a product ID for EOQ."

    # -------------------------------
    return (
        "I can answer questions about:\n"
        "• Unstable demand\n"
        "• Stock risk\n"
        "• Highest / lowest demand\n"
        "• Reorder point\n"
        "• EOQ\n\n"
        "Try asking:\n"
        "“Which product has unstable demand?”"
    )

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def inventory_optimization_page():

    st.header("📦 Inventory Optimization & Decision Intelligence")

    df = load_data()
    inv_df = build_inventory_metrics(df)

    # -------------------------------
    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    # -------------------------------
    st.subheader("📊 Inventory KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Products", inv_df.shape[0])
    c2.metric("High Risk SKUs", (inv_df.volatility > 0.5).sum())
    c3.metric("Avg Volatility", round(inv_df.volatility.mean(), 2))

    # -------------------------------
    st.subheader("📋 Inventory Table")
    st.dataframe(inv_df, width="stretch")

    # -------------------------------
    st.subheader("📈 Demand Volatility Chart")
    fig = px.bar(
        inv_df,
        x="product_id",
        y="volatility",
        color="inventory_status",
        title="Product-wise Demand Volatility"
    )
    st.plotly_chart(fig, width="stretch")

    # -------------------------------
    st.divider()
    st.subheader("💬 Inventory Q&A (Data-Driven NLP)")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    q = st.chat_input("Ask any inventory or demand related question...")

    if q:
        ans = answer_question(q, inv_df)
        st.session_state.chat.append((q, ans))

    for q, a in st.session_state.chat[-10:]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
