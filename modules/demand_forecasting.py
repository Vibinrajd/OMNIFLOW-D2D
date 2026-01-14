# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# ======================================================================================
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

# ------------------------------
# LIBRARIES
# ------------------------------
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# ------------------------------
# CONFIG
# ------------------------------
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
        "Promotion indicator (0/1)",
        "Previous day demand",
        "Demand one week ago",
        "7-day rolling average demand",
        "Predicted future demand",
        "Lower confidence bound",
        "Upper confidence bound"
    ]
})

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================
def data_profiling(df):
    profile = {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Missing Values (%)": round(df.isnull().mean().mean() * 100, 2),
        "Zero Sales Ratio (%)": round((df["daily_sales"] == 0).mean() * 100, 2),
        "Avg Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Std Dev": round(df["daily_sales"].std(), 2)
    }
    return profile

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def feature_engineering(df):
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
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        predictions[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    return results_df, best_model, predictions[best_model]

# ======================================================================================
# PDF REPORT
# ======================================================================================
def generate_pdf(metrics, insights):
    path = f"{OUTPUT_DIR}/Demand_Forecast_Report.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>OmniFlow-D2D Demand Forecast Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>AI-Generated Insights</b>", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(ins, styles["Normal"]))

    doc.build(story)
    return path

# ======================================================================================
# MAIN STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – Intelligence Module")

    # ------------------------------
    # LOAD & PROFILE DATA
    # ------------------------------
    df_raw = load_data()
    profile = data_profiling(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, use_container_width=True)

    with st.expander("🔍 Data Profiling & Quality Checks"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # ------------------------------
    # FEATURE ENGINEERING
    # ------------------------------
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_sales_1", "lag_sales_7", "rolling_mean_7"]
    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ------------------------------
    # TRAIN MODELS
    # ------------------------------
    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = preds
    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std

    # ------------------------------
    # KPIs
    # ------------------------------
    st.subheader("📊 Executive KPI Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Demand Volatility", round(forecast_df["forecast_demand"].std(), 2))

    # ------------------------------
    # MODEL COMPARISON
    # ------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    fig_rmse = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        title="Model RMSE Comparison",
        text="RMSE"
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    # ------------------------------
    # FILTERS
    # ------------------------------
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # ------------------------------
    # ADVANCED CHARTS (5+)
    # ------------------------------
    st.subheader("📈 Forecast Analysis")

    fig1 = px.line(fdf, x="date", y="forecast_demand", title="Forecast Trend")
    fig2 = px.line(fdf, x="date", y="rolling_mean_7", title="Rolling Demand")
    fig3 = px.histogram(fdf, x="forecast_demand", title="Forecast Distribution")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI", line=dict(dash="dot")))
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], name="Lower CI", fill="tonexty"))

    fig5 = px.box(fdf, y="forecast_demand", title="Demand Volatility")

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # AI INSIGHTS
    # ------------------------------
    st.subheader("🤖 AI-Driven Insights")

    insights = []
    cv = fdf["forecast_demand"].std() / fdf["forecast_demand"].mean()

    insights.append(
        f"Demand coefficient of variation is {cv:.2f}, indicating "
        f"{'high' if cv > 0.3 else 'moderate'} demand volatility."
    )

    if fdf["forecast_demand"].mean() > fdf["upper_bound"].mean() * 0.9:
        insights.append(
            "Forecast demand is approaching the upper confidence bound, "
            "indicating potential stock-out risk."
        )

    for ins in insights:
        st.write("•", ins)

    # ------------------------------
    # PDF DOWNLOAD
    # ------------------------------
    if st.button("📥 Download Detailed Demand Forecast Report (PDF)"):
        pdf = generate_pdf(
            {
                "Best Model": best_model,
                "Average Forecast": round(forecast_df["forecast_demand"].mean(), 2),
                "RMSE": round(results_df.iloc[0]["RMSE"], 2)
            },
            insights
        )
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name="Demand_Forecast_Report.pdf")

    st.success("✅ Demand Forecasting Analysis Completed")
