# ======================================================================================
# OmniFlow-D2D : Inventory Optimization & Risk Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

import os
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================
# PATH CONFIGURATION
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INVENTORY_PATH = os.path.join(BASE_DIR, "data", "inventory.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "outputs", "forecast_demand.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "inventory_optimization.csv")

# ======================================================================================
# LOAD DATA
# ======================================================================================
@st.cache_data
def load_inventory():
    return pd.read_csv(INVENTORY_PATH)

@st.cache_data
def load_forecast():
    return pd.read_csv(FORECAST_PATH)

# ======================================================================================
# INVENTORY CALCULATIONS
# ======================================================================================
def calculate_inventory_metrics(inv, fc):

    demand_stats = fc.groupby("product_id")["forecast_demand"].agg(
        avg_demand="mean",
        demand_std="std"
    ).reset_index()

    df = inv.merge(demand_stats, on="product_id", how="inner")

    # Safety Stock (Z=1.65 for 95%)
    df["safety_stock"] = 1.65 * df["demand_std"] * np.sqrt(df["lead_time_days"])

    # Reorder Point
    df["reorder_point"] = (
        df["avg_demand"] * df["lead_time_days"]
        + df["safety_stock"]
    )

    # EOQ
    df["EOQ"] = np.sqrt(
        (2 * df["avg_demand"] * df["ordering_cost"]) / df["holding_cost"]
    )

    # Risk Scores
    df["stock_gap"] = df["current_stock"] - df["reorder_point"]

    df["risk_level"] = np.where(
        df["stock_gap"] < 0, "STOCKOUT RISK",
        np.where(df["stock_gap"] > df["EOQ"], "OVERSTOCK", "OPTIMAL")
    )

    df["risk_score"] = np.abs(df["stock_gap"]) / df["reorder_point"]

    df.to_csv(OUTPUT_PATH, index=False)
    return df

# ======================================================================================
# INVENTORY NLP ENGINE (NO API)
# ======================================================================================
class InventoryNLP:

    def __init__(self, df):
        self.df = df
        self.kb = self._build_knowledge()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.kb.keys())

    def _build_knowledge(self):
        kb = {}

        for _, r in self.df.iterrows():
            p = r["product_id"]

            kb[f"stock level product {p}"] = (
                f"Product {p} current stock is {r['current_stock']:.0f} units."
            )
            kb[f"reorder point product {p}"] = (
                f"Product {p} reorder point is {r['reorder_point']:.0f} units."
            )
            kb[f"safety stock product {p}"] = (
                f"Product {p} safety stock is {r['safety_stock']:.0f} units."
            )
            kb[f"risk product {p}"] = (
                f"Product {p} is at {r['risk_level']}."
            )
            kb[f"eoq product {p}"] = (
                f"Product {p} EOQ is {r['EOQ']:.0f} units."
            )

        kb["highest stockout risk"] = (
            f"Highest stockout risk is product "
            f"{self.df.sort_values('stock_gap').iloc[0]['product_id']}."
        )

        kb["overstock products"] = (
            "Overstocked products are: "
            + ", ".join(
                self.df[self.df["risk_level"] == "OVERSTOCK"]["product_id"].astype(str)
            )
        )

        return kb

    def answer(self, question):
        q_vec = self.vectorizer.transform([question.lower()])
        sim = cosine_similarity(q_vec, self.matrix)
        idx = sim.argmax()

        if sim[0][idx] < 0.25:
            return (
                "I could not infer that.\n\n"
                "Try:\n"
                "- Which product has stockout risk?\n"
                "- Reorder point for product 1002\n"
                "- EOQ for product 1005\n"
            )

        return list(self.kb.values())[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def inventory_optimization_page():

    st.header("📦 Inventory Optimization & Risk Intelligence")

    if not os.path.exists(FORECAST_PATH):
        st.error("❌ Run Demand Forecasting module first.")
        return

    inv = load_inventory()
    fc = load_forecast()

    df = calculate_inventory_metrics(inv, fc)

    # ---------------- KPIs ----------------
    st.subheader("📊 Inventory KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stockout Risks", (df["risk_level"] == "STOCKOUT RISK").sum())
    c2.metric("Overstock Items", (df["risk_level"] == "OVERSTOCK").sum())
    c3.metric("Avg EOQ", int(df["EOQ"].mean()))

    # ---------------- TABLE ----------------
    st.subheader("📋 Inventory Optimization Output")
    st.dataframe(df, width="stretch")

    # ---------------- CHART ----------------
    fig = px.bar(
        df,
        x="product_id",
        y="stock_gap",
        color="risk_level",
        title="Stock Gap vs Risk"
    )
    st.plotly_chart(fig, width="stretch")

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "⬇ Download Inventory Optimization CSV",
        data=df.to_csv(index=False),
        file_name="inventory_optimization.csv"
    )

    # ---------------- NLP Q&A ----------------
    nlp = InventoryNLP(df)

    st.divider()
    st.subheader("🤖 Inventory Intelligence Assistant")

    q = st.chat_input("Ask inventory-related questions...")
    if q:
        ans = nlp.answer(q)
        with st.chat_message("assistant"):
            st.write(ans)

    st.success("Inventory optimization completed and saved.")
