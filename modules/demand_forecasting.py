# ============================================================
# OmniFlow-D2D | Demand Forecasting Dashboard
# Major MSc Data Science Project
# ============================================================

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------------------------------------
# PATH FIX (Colab / Streamlit Cloud / Local)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.demand_forecasting import run_demand_forecasting

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="OmniFlow D2D | Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand Intelligence Dashboard")
st.caption("AI-Powered Demand Forecasting | OmniFlow-D2D")

# ------------------------------------------------------------
# RUN MODEL
# ------------------------------------------------------------
with st.spinner("Running demand forecasting models..."):
    result = run_demand_forecasting()

forecast_df = result["forecast"]
model_comparison = result["model_comparison"]

st.success("Demand Forecast Completed")

# ============================================================
# DATA DICTIONARY
# ============================================================
with st.expander("📘 Data Dictionary"):
    st.markdown("""
**sales.csv & derived features**

| Column | Description |
|------|------------|
| date | Sales transaction date |
| product_id | Unique product identifier |
| daily_sales | Units sold per day |
| price | Unit selling price |
| promotion | Promotion flag (0/1) |
| lag_sales_1 | Previous day demand |
| lag_sales_7 | Last week demand |
| rolling_mean_7 | 7-day avg demand |
| forecast_demand | Predicted demand |
| lower_bound | Lower confidence bound |
| upper_bound | Upper confidence bound |
| model_used | Selected ML model |
""")

# ============================================================
# KPI CARDS
# ============================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "📦 Avg Forecast Demand",
    f"{int(forecast_df['forecast_demand'].mean()):,}"
)

c2.metric(
    "📉 Best RMSE",
    f"{model_comparison['RMSE'].min():.2f}"
)

c3.metric(
    "🏆 Best Model",
    model_comparison.sort_values("RMSE").iloc[0]["Model"]
)

c4.metric(
    "⚠ Demand Volatility",
    f"{forecast_df['forecast_demand'].std():.2f}"
)

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("🔍 Filters")

product_id = st.sidebar.selectbox(
    "Select Product",
    sorted(forecast_df["product_id"].unique())
)

filtered_df = forecast_df[
    forecast_df["product_id"] == product_id
].sort_values("date")

# ============================================================
# MODEL COMPARISON TABLE
# ============================================================
st.subheader("🤖 ML Model Comparison")
st.dataframe(model_comparison, use_container_width=True)

st.info("""
**Models Used**
• Linear Regression – baseline  
• Random Forest – non-linear demand  
• Gradient Boosting – seasonality & promotions  
""")

# ============================================================
# FORECAST CHART (ADVANCED)
# ============================================================
st.subheader("📊 Demand Forecast with Confidence Interval")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df["date"],
    y=filtered_df["forecast_demand"],
    mode="lines+markers",
    name="Forecast Demand",
    line=dict(width=3),
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
    xaxis_title="Date",
    yaxis_title="Units",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FORECAST PREVIEW
# ============================================================
st.subheader("📋 Forecast Preview")
st.dataframe(filtered_df.head(10), use_container_width=True)

# ============================================================
# AI INSIGHTS
# ============================================================
def generate_ai_insights(df):
    insights = []

    mean_demand = df["forecast_demand"].mean()

    if mean_demand > df["upper_bound"].mean() * 0.9:
        insights.append("⚠ High risk of stock-out. Increase safety stock.")

    if df["forecast_demand"].std() > 0.3 * mean_demand:
        insights.append("📈 High demand volatility detected.")

    if mean_demand < df["lower_bound"].mean() * 1.1:
        insights.append("📉 Demand softening. Reduce over-stocking.")

    insights.append("📦 Align inventory replenishment with forecast trend.")

    return insights

st.subheader("🤖 Automated AI Insights")
insights = generate_ai_insights(filtered_df)

for i in insights:
    st.write(i)

# ============================================================
# PDF REPORT GENERATION
# ============================================================
def generate_pdf_report(df, insights):
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/demand_forecast_report.pdf"

    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("OmniFlow-D2D Demand Forecast Report", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(f"Product ID: {df['product_id'].iloc[0]}", styles["Normal"]))
    story.append(Paragraph(f"Avg Forecast Demand: {df['forecast_demand'].mean():.2f}", styles["Normal"]))
    story.append(Paragraph("<br/><b>AI Insights</b>", styles["Heading2"]))

    for ins in insights:
        story.append(Paragraph(f"- {ins}", styles["Normal"]))

    doc.build(story)
    return path

# ============================================================
# DOWNLOAD PDF
# ============================================================
if st.button("📥 Download Executive Forecast Report (PDF)"):
    pdf_path = generate_pdf_report(filtered_df, insights)
    st.success("PDF Generated")

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download PDF",
            f,
            file_name="demand_forecast_report.pdf"
        )
