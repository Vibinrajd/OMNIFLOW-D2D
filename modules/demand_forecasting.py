# ======================================================================================
# OmniFlow-D2D : Demand Forecasting + Data Intelligence Engine
# MSc Data Science – MAJOR PROJECT
#
# File  : demand_forecasting.py
# Type  : Streamlit Page Module
# Mode  : Offline (No API, No Limits)
# ======================================================================================

# ======================================================================================
# IMPORTS
# ======================================================================================

import os
import re
import math
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ======================================================================================
# CONFIG
# ======================================================================================

DATA_PATH = "data/sales.csv"

# ======================================================================================
# DATA LOADING
# ======================================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error("❌ data/sales.csv not found")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================

def profile_data(df: pd.DataFrame) -> Dict:
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Number of Products": df["product_id"].nunique(),
        "Average Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2),
        "Zero Sales %": round((df["daily_sales"] == 0).mean() * 100, 2),
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"])

    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_mean_7"] = (
        df.groupby("product_id")["daily_sales"]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ======================================================================================
# MODEL TRAINING
# ======================================================================================

def train_models(X_train, y_train, X_test, y_test):

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=18, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        )
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": math.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        predictions[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    return results_df, best_model, predictions[best_model]

# ======================================================================================
# KNOWLEDGE BASE (DATA → TEXT)
# ======================================================================================

def build_knowledge_base(df_raw: pd.DataFrame, forecast_df: pd.DataFrame) -> List[str]:

    kb = []

    # ---------------- GLOBAL METRICS ----------------
    avg_demand = forecast_df["forecast"].mean()
    peak_demand = forecast_df["forecast"].max()
    volatility = forecast_df["forecast"].std()
    cv = volatility / avg_demand

    kb.append(f"Average forecast demand is {avg_demand:.2f} units.")
    kb.append(f"Peak forecast demand reaches {peak_demand:.2f} units.")
    kb.append(f"Demand volatility is {volatility:.2f}.")
    kb.append(f"Coefficient of variation is {cv:.2f}.")

    if cv > 0.3:
        kb.append("Demand is highly volatile and requires higher safety stock.")
        kb.append("High volatility increases stock-out risk.")
    else:
        kb.append("Demand volatility is moderate and manageable.")

    # ---------------- PRODUCT LEVEL ----------------
    product_avg = df_raw.groupby("product_id")["daily_sales"].mean()
    product_std = df_raw.groupby("product_id")["daily_sales"].std()

    high_product = product_avg.idxmax()
    low_product = product_avg.idxmin()

    kb.append(f"Highest demand product is {high_product}.")
    kb.append(f"Lowest demand product is {low_product}.")

    for pid in product_avg.index:
        kb.append(
            f"Product {pid} has average demand {product_avg[pid]:.2f} "
            f"and volatility {product_std[pid]:.2f}."
        )

        if product_std[pid] / product_avg[pid] > 0.35:
            kb.append(f"Product {pid} has high demand risk.")
        else:
            kb.append(f"Product {pid} has stable demand.")

    # ---------------- INVENTORY ----------------
    kb.append("Products with high volatility require safety stock.")
    kb.append("Stock-out risk increases near upper confidence bound.")
    kb.append("Stable demand products need less frequent replenishment.")
    kb.append("High demand products require close monitoring.")

    # ---------------- ML MODELS ----------------
    kb.append("Random Forest captures non-linear demand patterns.")
    kb.append("Gradient Boosting improves predictions iteratively.")
    kb.append("Linear Regression assumes linear relationships.")
    kb.append("Lower RMSE indicates better model performance.")

    # ---------------- BUSINESS ----------------
    kb.append("High demand and volatile products need priority management.")
    kb.append("Inventory buffers should align with demand uncertainty.")
    kb.append("Confidence intervals help in worst-case planning.")

    return kb

# ======================================================================================
# NLP QUESTION ANSWERING ENGINE
# ======================================================================================

class DataQnAEngine:

    def __init__(self, knowledge: List[str]):
        self.knowledge = knowledge
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(
            [self._clean(k) for k in knowledge]
        )

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return text

    def answer(self, question: str) -> str:
        q_vec = self.vectorizer.transform([self._clean(question)])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idx = sims.argmax()

        if sims[idx] < 0.12:
            return (
                "This question cannot be answered directly from the data. "
                "Please ask about demand, products, volatility, risk, models, or inventory."
            )

        return self.knowledge[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================

def demand_forecasting_page():

    st.title("📦 OmniFlow-D2D : Demand Forecasting & Intelligence")

    # -------- LOAD DATA --------
    df_raw = load_data()
    profile = profile_data(df_raw)

    with st.expander("📘 Data Profile"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # -------- FEATURE ENGINEERING --------
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_mean_7"]
    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # -------- MODEL TRAINING --------
    results_df, best_model, preds = train_models(
        X_train, y_train, X_test, y_test
    )

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast"] = preds
    std = preds.std()
    forecast_df["lower"] = preds - 1.96 * std
    forecast_df["upper"] = preds + 1.96 * std

    # -------- KPIs --------
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(forecast_df["forecast"].std(), 2))

    # -------- MODEL COMPARISON --------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df)

    st.plotly_chart(
        px.bar(results_df, x="Model", y="RMSE", text="RMSE"),
        width="stretch"
    )

    # -------- PRODUCT FILTER --------
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast"], name="Forecast"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper"], name="Upper CI", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower"], name="Lower CI", fill="tonexty"))
    fig.update_layout(title="Forecast with Confidence Interval")

    st.plotly_chart(fig, width="stretch")

    # -------- NLP ENGINE --------
    knowledge = build_knowledge_base(df_raw, forecast_df)
    engine = DataQnAEngine(knowledge)

    st.divider()
    st.subheader("🧠 Ask Anything About Your Data")

    st.caption(
        "Examples: high demand product | demand risk | volatility | "
        "inventory recommendation | model choice | stock-out risk"
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_q = st.chat_input("Ask your question...")

    if user_q:
        answer = engine.answer(user_q)
        st.session_state.chat.append((user_q, answer))

    for q, a in st.session_state.chat[-10:]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

# ======================================================================================
# ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
    demand_forecasting_page()
