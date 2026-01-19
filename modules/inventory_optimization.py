# ======================================================================================
# OmniFlow-D2D : Inventory Optimization Module
# Uses FULL Demand Forecast Output via JOIN
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

import os
import pandas as pd
import numpy as np
import streamlit as st

# ======================================================================================
# PATH CONFIG
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

FORECAST_FILE = os.path.join(DATA_DIR, "forecast_demand.csv")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.csv")

# ======================================================================================
# LOAD DATA
# ======================================================================================
@st.cache_data
def load_data():
    forecast_df = pd.read_csv(FORECAST_FILE)
    inventory_df = pd.read_csv(INVENTORY_FILE)

    # standardize column names
    forecast_df.columns = forecast_df.columns.str.strip().str.lower()
    inventory_df.columns = inventory_df.columns.str.strip().str.lower()

    if "date" in forecast_df.columns:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    return forecast_df, inventory_df

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def inventory_optimization_page():

    st.header("📦 Inventory Optimization Module")

    # ------------------------------------------------------------------------------
    # FILE CHECK
    # ------------------------------------------------------------------------------
    if not os.path.exists(FORECAST_FILE):
        st.error("❌ forecast_demand.csv not found. Run Demand Forecasting module first.")
        return

    if not os.path.exists(INVENTORY_FILE):
        st.error("❌ inventory.csv not found.")
        return

    # ------------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------------
    forecast_df, inventory_df = load_data()

    # ------------------------------------------------------------------------------
    # INVENTORY COLUMN MAPPING (NO ASSUMPTIONS)
    # ------------------------------------------------------------------------------
    st.subheader("⚙ Inventory Column Mapping")

    st.write("Inventory dataset columns:")
    st.code(inventory_df.columns.tolist())

    stock_col = st.selectbox(
        "Select the column that represents CURRENT STOCK",
        inventory_df.columns
    )

    # ------------------------------------------------------------------------------
    # JOIN DEMAND FORECAST WITH INVENTORY
    # ------------------------------------------------------------------------------
    merged_df = pd.merge(
        inventory_df,
        forecast_df,
        on=["store_id", "product_id"],
        how="left"
    )

    st.subheader("🔗 Joined Demand–Inventory Preview")
    st.write("Joined dataset shape:", merged_df.shape)
    st.dataframe(merged_df.head(10), use_container_width=True)

    # ------------------------------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------------------------------
    store = st.selectbox(
        "Select Store",
        sorted(merged_df["store_id"].unique())
    )

    product = st.selectbox(
        "Select Product",
        sorted(
            merged_df[merged_df["store_id"] == store]["product_id"].unique()
        )
    )

    data = merged_df[
        (merged_df["store_id"] == store) &
        (merged_df["product_id"] == product)
    ]

    # ------------------------------------------------------------------------------
    # CHECK FORECAST AVAILABILITY
    # ------------------------------------------------------------------------------
    if "forecast" not in data.columns or data["forecast"].isna().all():
        st.warning("⚠ No forecast available for this Store–Product combination.")
        return

    # ------------------------------------------------------------------------------
    # INVENTORY CALCULATIONS
    # ------------------------------------------------------------------------------
    avg_demand = data["forecast"].mean()
    safety_stock = (data["upper_ci"] - data["forecast"]).mean()

    lead_time = st.slider("Lead Time (days)", 1, 15, 5)

    current_stock = data[stock_col].iloc[0]
    reorder_point = (avg_demand * lead_time) + safety_stock
    order_quantity = max(0, reorder_point - current_stock)

    # ------------------------------------------------------------------------------
    # EXECUTIVE KPIs
    # ------------------------------------------------------------------------------
    st.subheader("📊 Inventory KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Current Stock", round(current_stock, 2))
    c2.metric("Avg Daily Demand", round(avg_demand, 2))
    c3.metric("Safety Stock", round(safety_stock, 2))
    c4.metric("Reorder Point", round(reorder_point, 2))

    # ------------------------------------------------------------------------------
    # INVENTORY DECISION OUTPUT
    # ------------------------------------------------------------------------------
    st.subheader("📄 Inventory Decision Output")

    output = pd.DataFrame({
        "store_id": [store],
        "product_id": [product],
        "current_stock": [round(current_stock, 2)],
        "avg_daily_demand": [round(avg_demand, 2)],
        "safety_stock": [round(safety_stock, 2)],
        "lead_time_days": [lead_time],
        "reorder_point": [round(reorder_point, 2)],
        "order_quantity": [round(order_quantity, 2)]
    })

    st.dataframe(output, use_container_width=True)

    # ------------------------------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------------------------------
    st.download_button(
        "⬇ Download Inventory Plan",
        output.to_csv(index=False),
        "inventory_plan.csv"
    )

    st.success("✅ Inventory Optimization Completed Successfully")
