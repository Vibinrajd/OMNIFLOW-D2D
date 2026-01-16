# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

# -----------------------------------
# IMPORTS
# -----------------------------------
import os
import warnings
import requests

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

# -----------------------------------
# CONFIG
# -----------------------------------
DATA_PATH = "data/sales.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# HUGGING FACE GEN-AI (FREE TIER – WORKING)
# ======================================================================================
def hf_genai_response(user_query, context):
    api_key = st.secrets.get("HF_API_KEY", None)

    if api_key is None:
        return "⚠️ Hugging Face API key not configured."

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a supply chain analytics expert.

Context:
{context}

Question:
{user_query}

Answer clearly with business reasoning:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # flan-t5 returns list[dict]
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]

        return "⚠️ Unexpected response format from Hugging Face."

    except Exception as e:
        return f"⚠️ Hugging Face error: {str(e)}"

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
        "Promotion flag (0/1)",
        "Previous day demand",
        "Demand one week ago",
        "7-day rolling average demand",
        "Predicted demand",
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
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Missing Values (%)": round(df.isnull().mean().mean() * 100, 2),
        "Zero Sales (%)": round((df["daily_sales"] == 0).mean() * 100, 2),
        "Average Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2)
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def feature_engineering(df):
    df = df.sort_values(["product_id", "date"])

    df["lag_sales_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_sales_7"] = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_mean_7"] = (
        df.groupby("product_id")["daily_sales"]
        .rolling(7).mean()
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
# PDF REPORT
# ======================================================================================
def generate_pdf(metrics, insights):
    path = f"{OUTPUT_DIR}/Demand_Forecast_Report.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("OmniFlow-D2D Demand Forecast Report", styles["Title"]))
    story.append(Spacer(1, 12))

    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("AI Insights", styles["Heading2"]))

    for ins in insights:
        story.append(Paragraph(ins, styles["Normal"]))

    doc.build(story)
    return path

# ======================================================================================
# MAIN STREAMLIT PAGE FUNCTION
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – AI Intelligence Module")

    # LOAD DATA
    df_raw = load_data()
    profile = data_profiling(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # FEATURE ENGINEERING
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_sales_1", "lag_sales_7", "rolling_mean_7"]
    X, y = df[FEATURES], df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # TRAIN MODELS
    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = preds

    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std

    # KPI CARDS
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(forecast_df["forecast_demand"].std(), 2))

    # MODEL COMPARISON
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig_rmse = px.bar(results_df, x="Model", y="RMSE", text="RMSE")
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, width="stretch")

    # FILTER
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # CHARTS
    fig1 = px.line(fdf, x="date", y="forecast_demand", markers=True)
    fig2 = px.histogram(fdf, x="forecast_demand")
    fig3 = px.box(fdf, y="forecast_demand")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI"))
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], name="Lower CI", fill="tonexty"))

    for fig in [fig1, fig2, fig3, fig4]:
        st.plotly_chart(fig, width="stretch")

    # INSIGHTS
    avg_demand = fdf["forecast_demand"].mean()
    peak_demand = fdf["forecast_demand"].max()
    volatility = (fdf["forecast_demand"].std() / avg_demand) * 100

    insights = [
        f"Average demand is {avg_demand:.0f} units.",
        f"Peak demand reaches {peak_demand:.0f} units.",
        f"Demand volatility is {volatility:.2f}%.",
        f"{best_model} achieved best accuracy."
    ]

    st.subheader("🤖 AI Insights")
    for ins in insights:
        st.write("•", ins)

    # PDF
    if st.button("📥 Download PDF Report"):
        pdf = generate_pdf(
            {"Best Model": best_model, "Avg Demand": avg_demand},
            insights
        )
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f)

    # GENAI CHAT
    st.divider()
    st.subheader("🤖 GenAI Demand Assistant")

    context = f"""
    Best Model: {best_model}
    Average Demand: {avg_demand}
    Peak Demand: {peak_demand}
    Volatility: {volatility}
    """

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask about demand, risk, or models")

    if user_input:
        reply = hf_genai_response(user_input, context)
        st.session_state.chat_history.append((user_input, reply))

    for q, a in st.session_state.chat_history[-10:]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
