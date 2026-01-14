# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# ======================================================================================
# Project Type : MAJOR MSc Data Science Project
# Module Type  : Streamlit Page Module (Imported into app.py)
#
# IMPORTANT:
# - This file DOES NOT run standalone
# - It exposes ONE function: demand_forecasting_page()
# - app.py controls routing and layout
#
# ======================================================================================

# ----------------------------------
# STANDARD LIBRARIES
# ----------------------------------
import os
import json
import warnings
from datetime import timedelta

# ----------------------------------
# THIRD PARTY LIBRARIES
# ----------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

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
        "forecast_demand", "lower_bound", "upper_bound", "model_used"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Unit selling price",
        "Promotion flag (0/1)",
        "Previous day sales",
        "Previous week sales",
        "7-day rolling average demand",
        "Predicted demand",
        "Lower confidence interval",
        "Upper confidence interval",
        "Best ML model selected"
    ]
})

# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================

@st.cache_data
def load_sales_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


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


def generate_pdf_report(metrics, insights):
    file_path = f"{OUTPUT_DIR}/demand_forecast_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>OmniFlow-D2D Demand Forecast Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Paragraph("<br/><b>AI Insights</b>", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"- {i}", styles["Normal"]))

    doc.build(story)
    return file_path


# ======================================================================================
# MAIN STREAMLIT PAGE FUNCTION
# ======================================================================================

def demand_forecasting_page():
    """
    Streamlit Demand Forecasting Page
    Called from app.py
    """

    st.header("📈 Demand Forecasting Intelligence")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    if not os.path.exists(DATA_PATH):
        st.error("❌ sales.csv not found in data/ folder")
        return

    raw_df = load_sales_data()

    # --------------------------------------------------
    # DATA DICTIONARY
    # --------------------------------------------------
    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, use_container_width=True)

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    df = feature_engineering(raw_df)

    FEATURES = [
        "price",
        "promotion",
        "lag_sales_1",
        "lag_sales_7",
        "rolling_mean_7"
    ]

    X = df[FEATURES]
    y = df["daily_sales"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------
    results_df, best_model, best_preds = train_models(
        X_train, y_train, X_test, y_test
    )

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = best_preds

    std = np.std(best_preds)
    forecast_df["lower_bound"] = best_preds - 1.96 * std
    forecast_df["upper_bound"] = best_preds + 1.96 * std
    forecast_df["model_used"] = best_model

    # --------------------------------------------------
    # KPIs
    # --------------------------------------------------
    st.subheader("📊 Executive KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Forecast", f"{forecast_df['forecast_demand'].mean():.0f}")
    c2.metric("Best Model", best_model)
    c3.metric("RMSE", f"{results_df.iloc[0]['RMSE']:.2f}")
    c4.metric("Volatility", f"{forecast_df['forecast_demand'].std():.2f}")

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    # --------------------------------------------------
    # FILTERS
    # --------------------------------------------------
    product = st.selectbox(
        "Select Product",
        sorted(forecast_df["product_id"].unique())
    )

    filtered_df = forecast_df[forecast_df["product_id"] == product]

    # --------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------
    st.subheader("📈 Forecast with Confidence Interval")

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
        line=dict(dash="dot"),
        name="Upper Bound"
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["lower_bound"],
        fill="tonexty",
        line=dict(dash="dot"),
        name="Lower Bound"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------
    st.subheader("🤖 AI Insights")

    insights = []
    if filtered_df["forecast_demand"].std() > 0.25 * filtered_df["forecast_demand"].mean():
        insights.append("⚠ High demand volatility detected")
    if filtered_df["forecast_demand"].mean() > filtered_df["upper_bound"].mean() * 0.9:
        insights.append("📦 Risk of stock-out")

    for i in insights:
        st.write(i)

    # --------------------------------------------------
    # PDF EXPORT
    # --------------------------------------------------
    if st.button("📥 Download Demand Forecast Report (PDF)"):
        pdf_path = generate_pdf_report(
            {
                "Best Model": best_model,
                "Avg Forecast": f"{forecast_df['forecast_demand'].mean():.2f}"
            },
            insights
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="demand_forecast_report.pdf"
            )

    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------
    forecast_df.to_csv(f"{OUTPUT_DIR}/forecast_demand.csv", index=False)

    st.success("✅ Demand Forecasting Completed")
