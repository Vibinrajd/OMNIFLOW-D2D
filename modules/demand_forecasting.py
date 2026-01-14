# ============================================================
# OmniFlow-D2D : Unified Streamlit Application (MAJOR PROJECT)
# ============================================================

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------------------------------------
# FIX PROJECT PATH (COLAB / GITHUB / STREAMLIT CLOUD SAFE)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ------------------------------------------------------------
# IMPORT MODULE 1
# ------------------------------------------------------------
from modules.demand_forecasting import run_demand_forecasting

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="OmniFlow D2D",
    page_icon="📦",
    layout="wide"
)

st.title("📦 OmniFlow D2D")
st.subheader("AI-Powered Demand-to-Delivery Optimization System")

# ============================================================
# DATA DICTIONARY
# ============================================================

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
        "Selling price per unit",
        "Promotion flag (0/1)",
        "Previous day sales",
        "Sales 7 days ago",
        "7-day rolling average demand",
        "Predicted future demand",
        "Lower confidence interval",
        "Upper confidence interval",
        "Best ML model selected"
    ]
})

# ============================================================
# SIDEBAR
# ============================================================
menu = st.sidebar.radio(
    "Navigation",
    [
        "Demand Intelligence",
        "Data Dictionary"
    ]
)

# ============================================================
# DEMAND INTELLIGENCE DASHBOARD
# ============================================================
if menu == "Demand Intelligence":

    with st.spinner("Running Demand Forecasting Engine..."):
        result = run_demand_forecasting()

    forecast_df = result["forecast"]
    model_df = result["model_comparison"]

    st.success("✅ Demand Forecast Completed")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📦 Avg Forecast Demand",
        f"{int(forecast_df['forecast_demand'].mean()):,}"
    )

    col2.metric(
        "📉 Best RMSE",
        f"{model_df['RMSE'].min():.2f}"
    )

    col3.metric(
        "🏆 Best Model",
        model_df.sort_values("RMSE").iloc[0]["Model"]
    )

    col4.metric(
        "⚠ Demand Volatility",
        f"{forecast_df['forecast_demand'].std():.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------
    product_id = st.selectbox(
        "Select Product",
        sorted(forecast_df["product_id"].unique())
    )

    filtered_df = forecast_df[
        forecast_df["product_id"] == product_id
    ].sort_values("date")

    # --------------------------------------------------------
    # MODEL COMPARISON TABLE
    # --------------------------------------------------------
    st.subheader("📊 Model Comparison")
    st.dataframe(model_df, use_container_width=True)

    # --------------------------------------------------------
    # FORECAST CHART WITH CONFIDENCE BANDS
    # --------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["forecast_demand"],
        mode="lines+markers",
        name="Forecast Demand",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["upper_bound"],
        mode="lines",
        name="Upper Bound",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["lower_bound"],
        mode="lines",
        name="Lower Bound",
        fill="tonexty",
        line=dict(dash="dot")
    ))

    fig.update_layout(
        title="📈 Demand Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------------
    st.subheader("🤖 Automated AI Insights")

    insights = []

    if filtered_df["forecast_demand"].mean() > filtered_df["upper_bound"].mean() * 0.9:
        insights.append("⚠ Demand nearing upper confidence limit → Stock-out risk.")

    if filtered_df["forecast_demand"].std() > 0.3 * filtered_df["forecast_demand"].mean():
        insights.append("📈 High demand volatility detected.")

    if filtered_df["forecast_demand"].mean() < filtered_df["lower_bound"].mean() * 1.1:
        insights.append("📉 Demand weakening → Inventory reduction recommended.")

    if not insights:
        insights.append("✅ Demand stable. No immediate action required.")

    for ins in insights:
        st.write(ins)

    # --------------------------------------------------------
    # PDF REPORT GENERATION
    # --------------------------------------------------------
    def generate_pdf(insights, df):
        os.makedirs("outputs", exist_ok=True)
        file_path = "outputs/demand_forecast_report.pdf"

        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("OmniFlow-D2D Demand Forecast Report", styles["Title"]))
        story.append(Paragraph("<br/>", styles["Normal"]))

        story.append(Paragraph(
            f"Average Forecast Demand: {df['forecast_demand'].mean():.2f}",
            styles["Normal"]
        ))

        story.append(Paragraph(
            f"Best Model: {model_df.sort_values('RMSE').iloc[0]['Model']}",
            styles["Normal"]
        ))

        story.append(Paragraph("<br/><b>AI Insights:</b>", styles["Heading2"]))
        for i in insights:
            story.append(Paragraph(f"- {i}", styles["Normal"]))

        doc.build(story)
        return file_path

    if st.button("📥 Download Executive Forecast Report (PDF)"):
        pdf_path = generate_pdf(insights, filtered_df)
        st.download_button(
            "Download PDF",
            open(pdf_path, "rb"),
            file_name="OmniFlow_Demand_Forecast_Report.pdf"
        )

# ============================================================
# DATA DICTIONARY PAGE
# ============================================================
elif menu == "Data Dictionary":
    st.header("📘 Data Dictionary")
    st.dataframe(DATA_DICTIONARY, use_container_width=True)

