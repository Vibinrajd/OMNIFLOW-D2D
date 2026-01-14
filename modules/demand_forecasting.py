# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Dashboard
# ======================================================================================
# Project Type : MAJOR MSc Data Science Project
# File Type    : Single-file Streamlit Application
# Module       : Demand Forecasting (Standalone)
#
# Features:
# - Data Dictionary
# - Data Profiling & Quality Checks
# - Feature Engineering
# - 3 ML Models (LR, RF, GB)
# - Model Comparison
# - Confidence Intervals
# - Executive KPI Cards
# - Interactive Filters
# - Advanced Charts (Plotly)
# - AI Insights Engine
# - Downloadable PDF Report
#
# ======================================================================================

# ---------------------------------------
# STANDARD LIBRARIES
# ---------------------------------------
import os
import sys
import json
import warnings
from datetime import datetime, timedelta

# ---------------------------------------
# THIRD PARTY LIBRARIES
# ---------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# ======================================================================================
# STREAMLIT CONFIG
# ======================================================================================

st.set_page_config(
    page_title="OmniFlow D2D – Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand Intelligence – OmniFlow D2D")
st.caption("AI-Powered Demand Forecasting & Decision Support System")

# ======================================================================================
# CONFIG
# ======================================================================================

DATA_PATH = "data/sales.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================

DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date", "product_id", "daily_sales", "price", "promotion",
        "lag_sales_1", "lag_sales_7", "rolling_mean_7",
        "forecast_demand", "lower_bound", "upper_bound"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Unit selling price",
        "Promotion applied flag (0/1)",
        "Previous day sales",
        "Previous week sales",
        "7-day rolling average demand",
        "Predicted demand",
        "Lower confidence interval",
        "Upper confidence interval"
    ]
})

# ======================================================================================
# LOAD DATA
# ======================================================================================

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

if not os.path.exists(DATA_PATH):
    st.error("❌ data/sales.csv not found")
    st.stop()

raw_df = load_data(DATA_PATH)

# ======================================================================================
# DATA PROFILING
# ======================================================================================

st.subheader("📘 Data Dictionary")
st.dataframe(DATA_DICTIONARY, use_container_width=True)

st.subheader("🔍 Data Quality Summary")

dq_col1, dq_col2, dq_col3 = st.columns(3)
dq_col1.metric("Total Records", len(raw_df))
dq_col2.metric("Products", raw_df["product_id"].nunique())
dq_col3.metric("Date Range", f"{raw_df['date'].min().date()} → {raw_df['date'].max().date()}")

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================

df = raw_df.copy()
df = df.sort_values(["product_id", "date"])

df["lag_sales_1"] = df.groupby("product_id")["daily_sales"].shift(1)
df["lag_sales_7"] = df.groupby("product_id")["daily_sales"].shift(7)
df["rolling_mean_7"] = (
    df.groupby("product_id")["daily_sales"]
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = [
    "price",
    "promotion",
    "lag_sales_1",
    "lag_sales_7",
    "rolling_mean_7"
]

X = df[FEATURES]
y = df["daily_sales"]

# ======================================================================================
# TRAIN / TEST SPLIT (TIME-AWARE)
# ======================================================================================

split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ======================================================================================
# ML MODELS
# ======================================================================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
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
best_model_name = results_df.iloc[0]["Model"]
best_preds = predictions[best_model_name]

# ======================================================================================
# FORECAST OUTPUT
# ======================================================================================

forecast_df = df.iloc[X_test.index].copy()
forecast_df["forecast_demand"] = best_preds

std_dev = np.std(best_preds)
forecast_df["lower_bound"] = best_preds - 1.96 * std_dev
forecast_df["upper_bound"] = best_preds + 1.96 * std_dev
forecast_df["model_used"] = best_model_name

# ======================================================================================
# KPI CARDS
# ======================================================================================

st.subheader("📊 Executive KPIs")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Avg Forecast", f"{forecast_df['forecast_demand'].mean():.0f}")
k2.metric("Best Model", best_model_name)
k3.metric("RMSE", f"{results_df.iloc[0]['RMSE']:.2f}")
k4.metric("Demand Volatility", f"{forecast_df['forecast_demand'].std():.2f}")

# ======================================================================================
# MODEL COMPARISON
# ======================================================================================

st.subheader("🤖 Model Comparison")
st.dataframe(results_df, use_container_width=True)

# ======================================================================================
# FILTERS
# ======================================================================================

st.subheader("🎯 Filters")

product = st.selectbox(
    "Select Product",
    sorted(forecast_df["product_id"].unique())
)

filtered_df = forecast_df[forecast_df["product_id"] == product]

# ======================================================================================
# FORECAST CHART
# ======================================================================================

st.subheader("📈 Demand Forecast with Confidence Interval")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df["date"],
    y=filtered_df["forecast_demand"],
    mode="lines+markers",
    name="Forecast"
))

fig.add_trace(go.Scatter(
    x=filtered_df["date"],
    y=filtered_df["upper_bound"],
    name="Upper Bound",
    line=dict(dash="dot")
))

fig.add_trace(go.Scatter(
    x=filtered_df["date"],
    y=filtered_df["lower_bound"],
    name="Lower Bound",
    fill="tonexty",
    line=dict(dash="dot")
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Units",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# AI INSIGHTS
# ======================================================================================

st.subheader("🤖 AI-Generated Insights")

insights = []

if filtered_df["forecast_demand"].std() > 0.25 * filtered_df["forecast_demand"].mean():
    insights.append("⚠ High demand volatility detected – consider safety stock.")

if filtered_df["forecast_demand"].mean() > filtered_df["upper_bound"].mean() * 0.9:
    insights.append("📦 Demand approaching upper limit – risk of stock-out.")

if filtered_df["forecast_demand"].mean() < filtered_df["lower_bound"].mean() * 1.1:
    insights.append("📉 Demand weakening – inventory reduction recommended.")

for i in insights:
    st.write(i)

# ======================================================================================
# PDF REPORT
# ======================================================================================

def generate_pdf(forecast, insights):
    file_path = f"{OUTPUT_DIR}/demand_forecast_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>OmniFlow-D2D Demand Forecast Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(f"Best Model: {best_model_name}", styles["Normal"]))
    story.append(Paragraph(f"Avg Forecast: {forecast['forecast_demand'].mean():.2f}", styles["Normal"]))

    story.append(Paragraph("<br/><b>AI Insights</b>", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(i, styles["Normal"]))

    doc.build(story)
    return file_path

if st.button("📥 Download AI Forecast Report (PDF)"):
    pdf_path = generate_pdf(filtered_df, insights)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download Report",
            f,
            file_name="demand_forecast_report.pdf"
        )

# ======================================================================================
# SAVE CSV
# ======================================================================================

forecast_df.to_csv(f"{OUTPUT_DIR}/forecast_demand.csv", index=False)

st.success("✅ Demand Forecasting Pipeline Completed")
