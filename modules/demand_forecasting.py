# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

import os
import warnings
from typing import Dict, List

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
# PATH CONFIGURATION
# ======================================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

SALES_FILE = os.path.join(DATA_DIR, "sales.csv")
FORECAST_FILE = os.path.join(OUTPUT_DIR, "forecast_demand.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================

DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date", "product_id", "daily_sales", "price", "promotion",
        "lag_1", "lag_7", "rolling_7",
        "forecast_demand", "lower_bound", "upper_bound"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Selling price",
        "Promotion flag (0/1)",
        "Lagged demand (t-1)",
        "Lagged demand (t-7)",
        "7-day rolling mean",
        "Forecasted demand",
        "Lower confidence bound",
        "Upper confidence bound"
    ]
})

# ======================================================================================
# DATA LOADING
# ======================================================================================

@st.cache_data
def load_sales_data() -> pd.DataFrame:
    df = pd.read_csv(SALES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================

def data_profile(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Unique Products": df["product_id"].nunique(),
        "Missing Values (%)": round(df.isnull().mean().mean() * 100, 2),
        "Avg Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Std Dev": round(df["daily_sales"].std(), 2)
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"])

    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)

    df["rolling_7"] = (
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
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    }

    metrics = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        predictions[name] = preds

    metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
    best_model = metrics_df.iloc[0]["Model"]

    return metrics_df, best_model, predictions[best_model]

# ======================================================================================
# NLP ANALYTICS ENGINE (NO API)
# ======================================================================================

class DemandAnalyticsNLP:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.kb = self._build_kb()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.kb.keys())

    def _build_kb(self):
        kb = {}
        g = self.df.groupby("product_id")

        avg = g["forecast_demand"].mean()
        std = g["forecast_demand"].std()

        kb["highest demand product"] = f"Product {avg.idxmax()} has the highest demand."
        kb["lowest demand product"] = f"Product {avg.idxmin()} has the lowest demand."
        kb["unstable demand"] = f"Product {std.idxmax()} shows highest volatility."
        kb["high risk product"] = f"Product {std.idxmax()} has high demand risk."
        kb["confidence interval"] = "Confidence interval shows uncertainty around forecasts."

        return kb

    def answer(self, question: str) -> str:
        vec = self.vectorizer.transform([question.lower()])
        sim = cosine_similarity(vec, self.matrix)
        idx = sim.argmax()

        if sim[0][idx] < 0.3:
            return "Try asking about demand, risk, volatility, or confidence interval."

        return list(self.kb.values())[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================

def demand_forecasting_page():

    st.header("📈 OmniFlow-D2D | Demand Forecasting Intelligence")

    if not os.path.exists(SALES_FILE):
        st.error("sales.csv not found in data folder")
        return

    df_raw = load_sales_data()
    profile = data_profile(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, use_container_width=True)

    with st.expander("🔍 Data Profiling"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]
    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = preds

    sigma = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * sigma
    forecast_df["upper_bound"] = preds + 1.96 * sigma

    forecast_df.to_csv(FORECAST_FILE, index=False)

    # KPIs
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(sigma, 2))

    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("📈 Forecast Visualization")
    pid = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == pid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], name="Lower CI", fill="tonexty"))

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇ Download Forecast Output",
        forecast_df.to_csv(index=False),
        file_name="forecast_demand.csv"
    )

    st.divider()
    st.subheader("📊 Demand Analytics Assistant")

    nlp = DemandAnalyticsNLP(forecast_df)
    q = st.chat_input("Ask about demand, risk, volatility...")

    if q:
        st.write(nlp.answer(q))

    st.success("✅ Demand Forecasting Module Completed")
