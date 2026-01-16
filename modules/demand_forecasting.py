# ======================================================================================
# OmniFlow-D2D : Demand Forecasting & Data Intelligence Engine
# MSc Data Science – MAJOR PROJECT
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
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ data/sales.csv not found")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================

def profile_data(df):
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Products": df["product_id"].nunique(),
        "Avg Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2),
        "Zero Sales (%)": round((df["daily_sales"] == 0).mean() * 100, 2),
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================

def feature_engineering(df):
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
        ),
    }

    results, predictions = [], {}

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
# ANALYTICS HELPERS
# ======================================================================================

def classify_demand_trend(fdf):
    slope = np.polyfit(range(len(fdf)), fdf["forecast"], 1)[0]
    if slope > 0.05:
        return "Increasing Demand"
    elif slope < -0.05:
        return "Decreasing Demand"
    else:
        return "Stable Demand"

def calculate_risk_score(fdf):
    cv = fdf["forecast"].std() / fdf["forecast"].mean()
    return min(100, int(cv * 100))

def explain_forecast_chart(fdf):
    trend = classify_demand_trend(fdf)
    avg = fdf["forecast"].mean()
    vol = fdf["forecast"].std()

    return (
        f"Demand trend is classified as {trend}. "
        f"Average forecast demand is {avg:.2f} units with volatility {vol:.2f}. "
        f"Confidence intervals indicate uncertainty boundaries."
    )

# ======================================================================================
# KNOWLEDGE BASE
# ======================================================================================

def build_knowledge_base(df_raw, forecast_df):

    kb = []

    avg = forecast_df["forecast"].mean()
    peak = forecast_df["forecast"].max()
    vol = forecast_df["forecast"].std()
    cv = vol / avg

    kb.extend([
        f"Average demand is {avg:.2f} units.",
        f"Peak demand is {peak:.2f} units.",
        f"Demand volatility is {vol:.2f}.",
        f"Coefficient of variation is {cv:.2f}.",
    ])

    if cv > 0.3:
        kb.append("Demand is highly volatile and risky.")
    else:
        kb.append("Demand is relatively stable.")

    prod_avg = df_raw.groupby("product_id")["daily_sales"].mean()
    prod_std = df_raw.groupby("product_id")["daily_sales"].std()

    for pid in prod_avg.index:
        kb.append(
            f"Product {pid} has average demand {prod_avg[pid]:.2f} "
            f"and volatility {prod_std[pid]:.2f}."
        )

    kb.extend([
        "Random Forest handles non-linear demand patterns well.",
        "Gradient Boosting improves forecasts iteratively.",
        "Linear Regression assumes linear relationships.",
        "Lower RMSE indicates better model accuracy.",
        "High volatility products need safety stock.",
        "Inventory buffers should match demand uncertainty."
    ])

    return kb

# ======================================================================================
# NLP ENGINE
# ======================================================================================

def route_question(q):
    q = q.lower()
    if any(w in q for w in ["trend", "increase", "decrease"]):
        return "trend"
    if any(w in q for w in ["risk", "volatile"]):
        return "risk"
    if any(w in q for w in ["model", "accuracy", "rmse"]):
        return "model"
    if any(w in q for w in ["inventory", "stock"]):
        return "inventory"
    return "general"

class DataQnAEngine:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform(knowledge)

    def answer(self, question, fdf):
        intent = route_question(question)

        if intent == "trend":
            return f"Demand trend is {classify_demand_trend(fdf)}."

        if intent == "risk":
            return f"Demand risk score is {calculate_risk_score(fdf)}."

        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idx = sims.argmax()

        if sims[idx] < 0.12:
            return "Question not directly answerable from data. Please ask about demand, risk, models, or inventory."

        return self.knowledge[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================

def demand_forecasting_page():

    st.title("📦 OmniFlow-D2D – Demand Forecasting Intelligence")

    df_raw = load_data()
    profile = profile_data(df_raw)

    with st.expander("📘 Data Profile"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_mean_7"]
    X, y = df[FEATURES], df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast"] = preds
    std = preds.std()
    forecast_df["lower"] = preds - 1.96 * std
    forecast_df["upper"] = preds + 1.96 * std

    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Risk Score", calculate_risk_score(forecast_df))

    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df)

    st.plotly_chart(
        px.bar(results_df, x="Model", y="RMSE", text="RMSE"),
        width="stretch"
    )

    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast"], name="Forecast"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper"], name="Upper CI", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower"], name="Lower CI", fill="tonexty"))
    fig.update_layout(title="Forecast with Confidence Interval")

    st.plotly_chart(fig, width="stretch")
    st.info(explain_forecast_chart(fdf))

    knowledge = build_knowledge_base(df_raw, forecast_df)
    engine = DataQnAEngine(knowledge)

    st.divider()
    st.subheader("🧠 Ask Anything About Your Data")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    q = st.chat_input("Ask about demand, risk, trend, model, inventory...")

    if q:
        a = engine.answer(q, fdf)
        st.session_state.chat.append((q, a))

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
