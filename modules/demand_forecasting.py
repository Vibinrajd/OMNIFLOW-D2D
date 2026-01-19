# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# PURPOSE:
# - Demand analysis & forecasting
# - Feature engineering
# - Multiple ML models
# - Model comparison
# - Confidence intervals
# - Executive KPIs
# - NLP-based analytics Q&A (NO API)
# - Downloadable outputs for downstream modules
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

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

SALES_FILE = os.path.join(DATA_DIR, "sales.csv")
FORECAST_FILE = os.path.join(OUTPUT_DIR, "forecast_demand.csv")

os.makedirs(DATA_DIR, exist_ok=True)
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
        "Unit selling price",
        "Promotion flag (0/1)",
        "Previous day demand",
        "Demand 7 days ago",
        "7-day rolling average",
        "Predicted demand",
        "Lower confidence interval",
        "Upper confidence interval"
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
        "Average Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2)
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
# MODEL TRAINING & COMPARISON
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

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        predictions[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    return results_df, best_model, predictions[best_model]

# ======================================================================================
# NLP ANALYTICS ENGINE (NO API)
# ======================================================================================

class DemandAnalyticsNLP:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.kb = self._build_kb()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.kb.keys())

    def _build_kb(self) -> Dict[str, str]:
        kb = {}
        g = self.df.groupby("product_id")

        avg = g["forecast_demand"].mean()
        std = g["forecast_demand"].std()

        kb["which product has highest demand"] = (
            f"Product {avg.idxmax()} has the highest average demand "
            f"({avg.max():.2f} units)."
        )

        kb["which product has lowest demand"] = (
            f"Product {avg.idxmin()} has the lowest average demand "
            f"({avg.min():.2f} units)."
        )

        kb["which product has unstable demand"] = (
            f"Product {std.idxmax()} shows the highest demand volatility."
        )

        kb["which product has highest risk"] = (
            f"Product {std.idxmax()} has the highest forecast risk due to volatility."
        )

        kb["what is confidence interval"] = (
            "A confidence interval represents the uncertainty range around the "
            "forecasted demand based on historical variation."
        )

        return kb

    def answer(self, question: str) -> str:
        q_vec = self.vectorizer.transform([question.lower()])
        sim = cosine_similarity(q_vec, self.matrix)
        idx = sim.argmax()

        if sim[0][idx] < 0.3:
            return (
                "Try asking:\n"
                "- which product has highest demand\n"
                "- which product has unstable demand\n"
                "- which product has highest risk\n"
                "- what is confidence interval"
            )

        return list(self.kb.values())[idx]

# ======================================================================================
# STREAMLIT APPLICATION
# ======================================================================================

def main():

    st.set_page_config(page_title="OmniFlow-D2D | Demand Forecasting", layout="wide")
    st.title("📈 OmniFlow-D2D – Demand Forecasting Intelligence")

    if not os.path.exists(SALES_FILE):
        st.error("❌ sales.csv not found. Place it inside the /data folder.")
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

    # Model comparison
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    fig = px.bar(results_df, x="Model", y="RMSE", text="RMSE")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast visualization
    st.subheader("📈 Forecast with Confidence Interval")
    pid = st.selectbox("Select Product ID", sorted(forecast_df["product_id"].unique()))
    fdf = forecast_df[forecast_df["product_id"] == pid]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    fig2.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI", line=dict(dash="dot")))
    fig2.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], name="Lower CI", fill="tonexty"))
    st.plotly_chart(fig2, use_container_width=True)

    # Download
    st.download_button(
        "⬇ Download Forecast Output",
        forecast_df.to_csv(index=False),
        file_name="forecast_demand.csv"
    )

    # NLP assistant
    st.divider()
    st.subheader("🧠 Demand Analytics Assistant (No API)")

    nlp = DemandAnalyticsNLP(forecast_df)
    question = st.chat_input("Ask about demand, risk, volatility...")

    if question:
        st.write(nlp.answer(question))

    st.success("✅ Demand Forecasting Module Executed Successfully")

# ======================================================================================
# RUN
# ======================================================================================

if __name__ == "__main__":
    main()
