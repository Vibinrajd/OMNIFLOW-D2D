# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# FEATURES INCLUDED (AS REQUESTED):
# ✔ Data Dictionary
# ✔ Data Profiling & Quality Checks
# ✔ Feature Engineering (Lag + Rolling)
# ✔ 3 ML Models (LR, RF, GB)
# ✔ Model Comparison
# ✔ Confidence Intervals
# ✔ Executive KPIs
# ✔ Interactive Filters
# ✔ Forecast Visualizations
# ✔ NLP-based Analytics Q&A (NO API, NO LLM)
# ✔ Downloadable Output for Inventory Module
# ======================================================================================

# ---------------------------------------
# IMPORTS
# ---------------------------------------
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
# PATH CONFIG
# ======================================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        "Selling price per unit",
        "Promotion indicator (0/1)",
        "Sales one day before",
        "Sales seven days before",
        "7-day rolling mean demand",
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
        "Sales Volatility (Std)": round(df["daily_sales"].std(), 2)
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

    results, predictions = [], {}

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
# NLP ANALYTICS ENGINE (RULE + DATA BASED)
# ======================================================================================
class DemandAnalyticsNLP:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.kb = self._build_knowledge()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.kb.keys())

    def _build_knowledge(self) -> Dict[str, str]:
        kb = {}
        g = self.df.groupby("product_id")

        avg = g["forecast_demand"].mean()
        std = g["forecast_demand"].std()
        ci = g["upper_bound"].mean() - g["lower_bound"].mean()

        kb["highest demand product"] = (
            f"Product {avg.idxmax()} has the highest average demand "
            f"({avg.max():.2f} units)."
        )

        kb["lowest demand product"] = (
            f"Product {avg.idxmin()} has the lowest average demand "
            f"({avg.min():.2f} units)."
        )

        kb["unstable demand product"] = (
            f"Product {std.idxmax()} shows the highest demand volatility "
            f"({std.max():.2f} units)."
        )

        kb["demand risk"] = (
            f"Product {std.idxmax()} has the highest demand risk due to "
            f"high variability and uncertainty."
        )

        kb["confidence interval"] = (
            "Confidence interval represents uncertainty around forecasted demand "
            "based on historical variability."
        )

        kb["monitoring product"] = (
            f"Product {std.idxmax()} requires close monitoring due to unstable demand."
        )

        return kb

    def answer(self, question: str) -> str:
        q = question.lower()
        vec = self.vectorizer.transform([q])
        sim = cosine_similarity(vec, self.matrix)
        idx = sim.argmax()

        if sim[0][idx] < 0.3:
            return (
                "I can answer:\n"
                "- highest demand product\n"
                "- lowest demand product\n"
                "- unstable demand product\n"
                "- demand risk\n"
                "- confidence interval\n"
                "- monitoring product"
            )

        return list(self.kb.values())[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – Intelligence Module")

    if not os.path.exists(SALES_FILE):
        st.error("❌ sales.csv not found in data folder")
        return

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    df_raw = load_sales_data()
    profile = data_profile(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]
    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # -------------------------------
    # TRAIN MODELS
    # -------------------------------
    results_df, best_model, preds = train_models(
        X_train, y_train, X_test, y_test
    )

    # ✅ CORRECT ALIGNMENT (FIXED)
    forecast_df = df.iloc[split:].copy()
    forecast_df["forecast_demand"] = preds

    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std

    forecast_df.to_csv(FORECAST_FILE, index=False)

    # -------------------------------
    # KPIs
    # -------------------------------
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(forecast_df["forecast_demand"].std(), 2))

    # -------------------------------
    # MODEL COMPARISON
    # -------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig_rmse = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="RMSE Comparison"
    )
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, width="stretch")

    # -------------------------------
    # PRODUCT FILTER
    # -------------------------------
    product = st.selectbox(
        "Select Product",
        sorted(forecast_df["product_id"].unique())
    )
    fdf = forecast_df[forecast_df["product_id"] == product]

    # -------------------------------
    # FORECAST PLOT
    # -------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fdf["date"],
        y=fdf["forecast_demand"],
        mode="lines+markers",
        name="Forecast"
    ))
    fig.add_trace(go.Scatter(
        x=fdf["date"],
        y=fdf["upper_bound"],
        name="Upper CI",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fdf["date"],
        y=fdf["lower_bound"],
        name="Lower CI",
        fill="tonexty"
    ))
    fig.update_layout(title="Forecast with Confidence Interval")
    st.plotly_chart(fig, width="stretch")

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    st.download_button(
        "⬇ Download Forecast Output",
        forecast_df.to_csv(index=False),
        file_name="forecast_demand.csv"
    )

    # -------------------------------
    # NLP ANALYTICS CHAT
    # -------------------------------
    st.divider()
    st.subheader("📊 Demand Analytics Assistant")

    nlp = DemandAnalyticsNLP(forecast_df)
    q = st.chat_input("Ask questions about demand, risk, volatility...")

    if q:
        with st.chat_message("assistant"):
            st.write(nlp.answer(q))

    st.success("✅ Demand Forecasting Completed Successfully")
