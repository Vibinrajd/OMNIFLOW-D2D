# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# PURPOSE:
# - Demand forecasting with ML
# - Cold-start product handling
# - Risk & volatility analytics
# - Rule-based NLP Q&A over forecasted data
# - Streamlit-compatible, GitHub deployable
# ======================================================================================

import os
import re
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

warnings.filterwarnings("ignore")

# ======================================================================================
# CONFIG
# ======================================================================================

DATA_PATH = "data/sales.csv"
OUTPUT_DIR = "outputs"
FORECAST_FILE = f"{OUTPUT_DIR}/forecast_demand.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_HISTORY_DAYS = 8  # minimum for lag + rolling

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================

DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date", "product_id", "daily_sales", "price", "promotion",
        "lag_1", "lag_7", "rolling_7",
        "forecast_demand", "lower_ci", "upper_ci",
        "volatility", "forecast_status"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Unit selling price",
        "Promotion flag (0/1)",
        "Sales previous day",
        "Sales 7 days ago",
        "7-day rolling mean demand",
        "Predicted demand",
        "Lower confidence interval",
        "Upper confidence interval",
        "Demand standard deviation",
        "sufficient_history / cold_start"
    ]
})

# ======================================================================================
# DATA LOADING
# ======================================================================================

@st.cache_data
def load_sales_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("sales.csv not found in data/ folder")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["product_id", "date"])


# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)

    df["rolling_7"] = (
        df.groupby("product_id")["daily_sales"]
          .rolling(7)
          .mean()
          .reset_index(level=0, drop=True)
    )

    return df


# ======================================================================================
# PRODUCT SEGMENTATION
# ======================================================================================

def split_products(df: pd.DataFrame):
    counts = df.groupby("product_id").size()

    sufficient = counts[counts >= MIN_HISTORY_DAYS].index.tolist()
    cold_start = counts[counts < MIN_HISTORY_DAYS].index.tolist()

    return sufficient, cold_start


# ======================================================================================
# MODEL TRAINING
# ======================================================================================

def train_models(X_train, y_train, X_test, y_test):

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
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
# COLD START FORECAST
# ======================================================================================

def cold_start_forecast(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for pid, g in df.groupby("product_id"):
        avg = g["daily_sales"].mean()
        std = g["daily_sales"].std()

        out.append({
            "product_id": pid,
            "forecast_demand": avg,
            "lower_ci": avg - 1.65 * std,
            "upper_ci": avg + 1.65 * std,
            "volatility": std,
            "forecast_status": "cold_start"
        })

    return pd.DataFrame(out)


# ======================================================================================
# NLP ENGINE (RULE-BASED, DATA-DRIVEN)
# ======================================================================================

class DemandNLP:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def answer(self, query: str) -> str:
        q = query.lower()

        if "highest demand" in q or "high demand" in q:
            pid = self.df.groupby("product_id")["forecast_demand"].mean().idxmax()
            val = self.df.groupby("product_id")["forecast_demand"].mean().max()
            return f"Product {pid} has the highest average forecast demand ({val:.2f} units)."

        if "lowest demand" in q or "low demand" in q:
            pid = self.df.groupby("product_id")["forecast_demand"].mean().idxmin()
            val = self.df.groupby("product_id")["forecast_demand"].mean().min()
            return f"Product {pid} has the lowest average forecast demand ({val:.2f} units)."

        if "unstable" in q or "volatile" in q:
            pid = self.df.groupby("product_id")["volatility"].mean().idxmax()
            val = self.df.groupby("product_id")["volatility"].mean().max()
            return f"Product {pid} shows the most unstable demand with volatility {val:.2f}."

        if "risk" in q or "stockout" in q:
            risky = self.df[
                self.df["forecast_demand"] > self.df["upper_ci"] * 0.9
            ]["product_id"].unique()
            return f"Products at stock-out risk: {', '.join(map(str, risky))}"

        if re.search(r"product\s+\d+", q):
            pid = int(re.search(r"product\s+(\d+)", q).group(1))
            if pid not in self.df["product_id"].values:
                return f"No forecast data available for product {pid}."
            g = self.df[self.df["product_id"] == pid]
            return (
                f"Product {pid}: avg demand {g['forecast_demand'].mean():.2f}, "
                f"volatility {g['volatility'].mean():.2f}, "
                f"status {g['forecast_status'].iloc[0]}."
            )

        return (
            "I can answer questions like:\n"
            "- highest demand product\n"
            "- lowest demand product\n"
            "- unstable demand\n"
            "- stock-out risk\n"
            "- product 1002 demand analysis"
        )


# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================

def demand_forecasting_page():

    st.title("📈 Demand Forecasting & Risk Intelligence")

    df_raw = load_sales_data()

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    sufficient, cold_start = split_products(df_raw)

    # ---------------- Forecast – sufficient history ----------------
    df_feat = engineer_features(df_raw)
    df_sufficient = df_feat[df_feat["product_id"].isin(sufficient)].dropna()

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]

    X = df_sufficient[FEATURES]
    y = df_sufficient["daily_sales"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results_df, best_model, preds = train_models(
        X_train, y_train, X_test, y_test
    )

    forecast = df_sufficient.iloc[X_test.index].copy()
    forecast["forecast_demand"] = preds
    forecast["volatility"] = forecast.groupby("product_id")["forecast_demand"].transform("std")
    forecast["lower_ci"] = forecast["forecast_demand"] - 1.65 * forecast["volatility"]
    forecast["upper_ci"] = forecast["forecast_demand"] + 1.65 * forecast["volatility"]
    forecast["forecast_status"] = "sufficient_history"

    # ---------------- Cold start ----------------
    df_cold = df_raw[df_raw["product_id"].isin(cold_start)]
    cold_forecast = cold_start_forecast(df_cold)

    final_df = pd.concat(
        [
            forecast[[
                "product_id", "forecast_demand", "lower_ci",
                "upper_ci", "volatility", "forecast_status"
            ]],
            cold_forecast
        ],
        ignore_index=True
    )

    final_df.to_csv(FORECAST_FILE, index=False)

    # ---------------- UI ----------------
    st.subheader("📊 Model Comparison")
    st.dataframe(results_df, width="stretch")

    st.subheader("📈 Forecast Output")
    st.dataframe(final_df, width="stretch")

    st.download_button(
        "⬇ Download Forecast Output",
        final_df.to_csv(index=False),
        file_name="forecast_demand.csv"
    )

    # ---------------- NLP CHAT ----------------
    st.divider()
    st.subheader("🤖 Demand Intelligence Q&A")

    nlp = DemandNLP(final_df)

    question = st.text_input(
        "Ask any question related to demand data"
    )

    if question:
        answer = nlp.answer(question)
        st.success(answer)
