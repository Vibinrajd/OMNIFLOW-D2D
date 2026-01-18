# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

import os
import re
import warnings
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
MIN_HISTORY_DAYS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        "7-day rolling average",
        "Predicted demand",
        "Lower confidence interval",
        "Upper confidence interval",
        "Demand volatility",
        "Forecast reliability flag"
    ]
})

# ======================================================================================
# LOAD DATA
# ======================================================================================

@st.cache_data
def load_sales_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["product_id", "date"])

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================

def engineer_features(df):
    df = df.copy()
    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_7"] = (
        df.groupby("product_id")["daily_sales"]
        .rolling(7).mean()
        .reset_index(level=0, drop=True)
    )
    return df

# ======================================================================================
# PRODUCT SEGMENTATION
# ======================================================================================

def split_products(df):
    counts = df.groupby("product_id").size()
    sufficient = counts[counts >= MIN_HISTORY_DAYS].index.tolist()
    cold = counts[counts < MIN_HISTORY_DAYS].index.tolist()
    return sufficient, cold

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
    preds_map = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        preds_map[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    return results_df, best_model, preds_map[best_model]

# ======================================================================================
# COLD START FORECAST
# ======================================================================================

def cold_start_forecast(df):
    rows = []
    for pid, g in df.groupby("product_id"):
        avg = g["daily_sales"].mean()
        std = g["daily_sales"].std()
        rows.append({
            "product_id": pid,
            "forecast_demand": avg,
            "lower_ci": avg - 1.65 * std,
            "upper_ci": avg + 1.65 * std,
            "volatility": std,
            "forecast_status": "cold_start"
        })
    return pd.DataFrame(rows)

# ======================================================================================
# NLP ENGINE (RULE BASED)
# ======================================================================================

class DemandNLP:
    def __init__(self, df):
        self.df = df

    def answer(self, q):
        q = q.lower()

        if "unstable" in q or "volatile" in q:
            pid = self.df.groupby("product_id")["volatility"].mean().idxmax()
            return f"Product {pid} has the most unstable demand."

        if "highest demand" in q:
            pid = self.df.groupby("product_id")["forecast_demand"].mean().idxmax()
            return f"Product {pid} has the highest forecast demand."

        if re.search(r"product\s+\d+", q):
            pid = int(re.search(r"product\s+(\d+)", q).group(1))
            g = self.df[self.df["product_id"] == pid]
            if g.empty:
                return "Product not found."
            return (
                f"Product {pid}: "
                f"avg demand {g['forecast_demand'].mean():.2f}, "
                f"volatility {g['volatility'].mean():.2f}, "
                f"status {g['forecast_status'].iloc[0]}"
            )

        return "Ask about highest demand, unstable demand, or product 1001."

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================

def demand_forecasting_page():

    st.title("📈 Demand Forecasting")

    df_raw = load_sales_data()

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    sufficient, cold = split_products(df_raw)

    df_feat = engineer_features(df_raw)
    df_suff = df_feat[df_feat["product_id"].isin(sufficient)].dropna().reset_index(drop=True)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]
    X = df_suff[FEATURES]
    y = df_suff["daily_sales"]

    split = int(len(df_suff) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast = df_suff.iloc[split:].copy()
    forecast["forecast_demand"] = preds
    forecast["volatility"] = forecast.groupby("product_id")["forecast_demand"].transform("std")
    forecast["lower_ci"] = forecast["forecast_demand"] - 1.65 * forecast["volatility"]
    forecast["upper_ci"] = forecast["forecast_demand"] + 1.65 * forecast["volatility"]
    forecast["forecast_status"] = "sufficient_history"

    cold_df = cold_start_forecast(df_raw[df_raw["product_id"].isin(cold)])

    final_df = pd.concat([
        forecast[[
            "product_id", "forecast_demand", "lower_ci",
            "upper_ci", "volatility", "forecast_status"
        ]],
        cold_df
    ], ignore_index=True)

    final_df.to_csv(FORECAST_FILE, index=False)

    st.subheader("📊 Model Comparison")
    st.dataframe(results_df, width="stretch")

    st.subheader("📈 Forecast Output")
    st.dataframe(final_df, width="stretch")

    st.download_button(
        "⬇ Download Forecast CSV",
        final_df.to_csv(index=False),
        "forecast_demand.csv"
    )

    st.divider()
    st.subheader("🤖 Demand Q&A")

    nlp = DemandNLP(final_df)
    q = st.text_input("Ask a question about demand")

    if q:
        st.success(nlp.answer(q))
