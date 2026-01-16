# ======================================================================================
# OmniFlow-D2D : Inventory Optimization & Risk Intelligence Module
# Project Type : MSc Data Science – MAJOR PROJECT
# Author Style : Enterprise / Production-grade Analytics
# ======================================================================================

# ---------------------------------
# STANDARD LIBRARIES
# ---------------------------------
import os
import math
import warnings
from typing import Dict, List

# ---------------------------------
# THIRD-PARTY LIBRARIES
# ---------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ======================================================================================
# PATH CONFIGURATION
# ======================================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.csv")
FORECAST_FILE = os.path.join(OUTPUT_DIR, "forecast_demand.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "inventory_optimization.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# COLUMN RESOLUTION LAYER (REAL-WORLD SAFE)
# ======================================================================================
COLUMN_ALIASES = {
    "product_id": ["product_id", "sku", "item_id"],
    "current_stock": ["current_stock", "stock", "on_hand_qty"],
    "lead_time": ["lead_time_days", "lead_time", "supplier_lead_time"],
    "ordering_cost": ["ordering_cost", "order_cost"],
    "holding_cost": ["holding_cost", "holding_cost_per_unit"]
}

def resolve_column(df: pd.DataFrame, logical_name: str) -> str:
    for col in COLUMN_ALIASES[logical_name]:
        if col in df.columns:
            return col
    raise KeyError(f"Required column missing: {logical_name}")

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_inventory_data() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_FILE)

@st.cache_data
def load_forecast_data() -> pd.DataFrame:
    return pd.read_csv(FORECAST_FILE)

# ======================================================================================
# INVENTORY DATA PROFILING
# ======================================================================================
def inventory_profile(df: pd.DataFrame) -> Dict:
    return {
        "Total Products": df["product_id"].nunique(),
        "Avg Stock Level": round(df["current_stock"].mean(), 2),
        "Min Stock": int(df["current_stock"].min()),
        "Max Stock": int(df["current_stock"].max())
    }

# ======================================================================================
# CORE INVENTORY OPTIMIZATION ENGINE
# ======================================================================================
def compute_inventory_metrics(inv: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """
    Core business logic: EOQ, Safety Stock, Reorder Point, Risk
    """

    pid = resolve_column(inv, "product_id")
    stock = resolve_column(inv, "current_stock")
    lead = resolve_column(inv, "lead_time")
    order = resolve_column(inv, "ordering_cost")
    hold = resolve_column(inv, "holding_cost")

    # Demand statistics
    demand = (
        fc.groupby("product_id")["forecast_demand"]
        .agg(avg_demand="mean", demand_std="std")
        .reset_index()
    )

    df = inv.merge(demand, left_on=pid, right_on="product_id", how="inner")

    # ------------------------------
    # SAFETY STOCK (Service level 95%)
    # ------------------------------
    df["safety_stock"] = 1.65 * df["demand_std"] * np.sqrt(df[lead])

    # ------------------------------
    # REORDER POINT
    # ------------------------------
    df["reorder_point"] = (df["avg_demand"] * df[lead]) + df["safety_stock"]

    # ------------------------------
    # EOQ
    # ------------------------------
    df["EOQ"] = np.sqrt(
        (2 * df["avg_demand"] * df[order]) / df[hold]
    )

    # ------------------------------
    # STOCK GAP
    # ------------------------------
    df["stock_gap"] = df[stock] - df["reorder_point"]

    # ------------------------------
    # RISK CLASSIFICATION
    # ------------------------------
    conditions = [
        df["stock_gap"] < 0,
        df["stock_gap"] > df["EOQ"]
    ]
    choices = ["STOCKOUT RISK", "OVERSTOCK"]

    df["risk_level"] = np.select(
        conditions, choices, default="OPTIMAL"
    )

    df.to_csv(OUTPUT_FILE, index=False)
    return df

# ======================================================================================
# NLP QUESTION-ANSWER ENGINE (NO API)
# ======================================================================================
class InventoryNLP:
    """
    TF-IDF + Cosine Similarity
    Answers ONLY from computed inventory results
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.knowledge_base = self._build_knowledge_base()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.knowledge_base.keys())

    def _build_knowledge_base(self) -> Dict[str, str]:
        kb = {}

        for _, r in self.df.iterrows():
            pid = r["product_id"]

            kb[f"stock risk product {pid}"] = (
                f"Product {pid} is at {r['risk_level']} with stock gap "
                f"{r['stock_gap']:.0f} units."
            )

            kb[f"reorder point product {pid}"] = (
                f"Reorder point for product {pid} is "
                f"{r['reorder_point']:.0f} units."
            )

            kb[f"eoq product {pid}"] = (
                f"Economic Order Quantity for product {pid} is "
                f"{r['EOQ']:.0f} units."
            )

        # Global insights
        high_risk = self.df.sort_values("stock_gap").iloc[0]
        kb["which product has stockout risk"] = (
            f"Product {high_risk['product_id']} has the highest stockout risk."
        )

        unstable = self.df.sort_values("demand_std", ascending=False).iloc[0]
        kb["which product has unstable demand"] = (
            f"Product {unstable['product_id']} has the highest demand volatility "
            f"({unstable['demand_std']:.2f} units)."
        )

        return kb

    def answer(self, question: str) -> str:
        vec = self.vectorizer.transform([question.lower()])
        similarity = cosine_similarity(vec, self.matrix)
        idx = similarity.argmax()

        if similarity[0][idx] < 0.35:
            return (
                "Ask about:\n"
                "• stockout risk\n"
                "• reorder point\n"
                "• EOQ\n"
                "• unstable demand\n"
                "• inventory risk"
            )

        return list(self.knowledge_base.values())[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def inventory_optimization_page():

    st.header("📦 Inventory Optimization & Risk Intelligence")

    if not os.path.exists(FORECAST_FILE):
        st.error("❌ Run Demand Forecasting module first")
        return

    inv = load_inventory_data()
    fc = load_forecast_data()

    df = compute_inventory_metrics(inv, fc)

    # ------------------------------
    # KPI CARDS
    # ------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Stockout Risk Items", (df["risk_level"] == "STOCKOUT RISK").sum())
    c2.metric("Overstock Items", (df["risk_level"] == "OVERSTOCK").sum())
    c3.metric("Avg EOQ", int(df["EOQ"].mean()))

    # ------------------------------
    # TABLE
    # ------------------------------
    st.subheader("📋 Inventory Optimization Output")
    st.dataframe(df, width="stretch")

    # ------------------------------
    # VISUAL ANALYTICS
    # ------------------------------
    fig1 = px.bar(
        df, x="product_id", y="stock_gap",
        color="risk_level",
        title="Stock Gap by Product"
    )

    fig2 = px.scatter(
        df, x="avg_demand", y="current_stock",
        color="risk_level",
        size="EOQ",
        title="Demand vs Stock Position"
    )

    st.plotly_chart(fig1, width="stretch")
    st.plotly_chart(fig2, width="stretch")

    # ------------------------------
    # DOWNLOAD
    # ------------------------------
    st.download_button(
        "⬇ Download Inventory Optimization CSV",
        df.to_csv(index=False),
        file_name="inventory_optimization.csv"
    )

    # ------------------------------
    # NLP CHAT
    # ------------------------------
    st.divider()
    st.subheader("🤖 Inventory Intelligence Assistant")

    nlp = InventoryNLP(df)

    question = st.chat_input(
        "Ask: which product has unstable demand? | reorder point for product 1002?"
    )

    if question:
        with st.chat_message("assistant"):
            st.write(nlp.answer(question))
