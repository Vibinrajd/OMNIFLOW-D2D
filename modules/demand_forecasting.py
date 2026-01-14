# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

# ----------------------------------
# IMPORTS
# ----------------------------------
import os
import warnings
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

from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore")

# ----------------------------------
# CONFIG
# ----------------------------------
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
        "Promotion flag (0/1)",
        "Previous day sales",
        "Sales 7 days ago",
        "7-day rolling average demand",
        "Predicted demand",
        "Lower confidence interval",
        "Upper confidence interval"
    ]
})

# ======================================================================================
# HUGGING FACE GEN-AI FUNCTION
# ======================================================================================
# ======================================================================================
# HUGGING FACE GEN-AI (CONVERSATIONAL – FIXED)
# ======================================================================================
# ======================================================================================
# HUGGING FACE GEN-AI (NO AUTO-ROUTER – FIXED)
# ======================================================================================
def hf_genai_response(user_query, context):

    hf_token = st.secrets.get("HF_API_TOKEN", None)

    if hf_token is None:
        return "⚠️ Hugging Face API token not configured."

    try:
        client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=hf_token,
            provider="hf-inference"   # ✅ DISABLE AUTO-ROUTER
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior supply chain analytics expert.\n\n"
                    "Use ONLY the given context.\n"
                    "Explain ML results clearly.\n"
                    "Give business recommendations.\n\n"
                    f"Context:\n{context}"
                )
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Hugging Face error: {str(e)}"


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

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("AI-Driven Insights", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(ins, styles["Normal"]))

    doc.build(story)
    return path

# ======================================================================================
# MAIN STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – AI Intelligence Module")

    # ----------------------------------
    # LOAD DATA
    # ----------------------------------
    df_raw = load_data()
    profile = data_profiling(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling & Quality Checks"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # ----------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_sales_1", "lag_sales_7", "rolling_mean_7"]
    X, y = df[FEATURES], df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ----------------------------------
    # TRAIN MODELS
    # ----------------------------------
    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = preds

    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std

    # ----------------------------------
    # KPI CARDS
    # ----------------------------------
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(forecast_df["forecast_demand"].std(), 2))

    # ----------------------------------
    # MODEL COMPARISON
    # ----------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig_rmse = px.bar(
        results_df, x="Model", y="RMSE",
        text="RMSE", title="RMSE Comparison Across Models"
    )
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, width="stretch")

    # ----------------------------------
    # FILTER
    # ----------------------------------
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # ----------------------------------
    # CHARTS
    # ----------------------------------
    st.subheader("📈 Forecast Analysis")

    fig1 = px.line(
        fdf, x="date", y="forecast_demand",
        markers=True, text=fdf["forecast_demand"].round(0),
        title="Forecast Trend"
    )
    fig1.update_traces(textposition="top center")

    fig2 = px.line(fdf, x="date", y="rolling_mean_7", title="Rolling Mean Demand")
    fig3 = px.histogram(fdf, x="forecast_demand", title="Demand Distribution")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI", line=dict(dash="dot")))
    fig4.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], name="Lower CI", fill="tonexty"))
    fig4.update_layout(title="Forecast with Confidence Interval")

    fig5 = px.box(fdf, y="forecast_demand", title="Demand Volatility")

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        st.plotly_chart(fig, width="stretch")

    # ----------------------------------
    # RULE-BASED INSIGHTS
    # ----------------------------------
    avg_demand = fdf["forecast_demand"].mean()
    peak_demand = fdf["forecast_demand"].max()
    volatility_pct = (fdf["forecast_demand"].std() / avg_demand) * 100
    best_rmse = results_df.iloc[0]["RMSE"]

    insights = [
        f"Average forecast demand is {avg_demand:.0f} units.",
        f"Peak demand reaches {peak_demand:.0f} units.",
        f"Demand volatility is {volatility_pct:.2f}%.",
        f"{best_model} achieved the lowest RMSE ({best_rmse:.2f})."
    ]

    st.subheader("🤖 AI-Driven Insights")
    for ins in insights:
        st.write("•", ins)

    # ----------------------------------
    # PDF EXPORT
    # ----------------------------------
    if st.button("📥 Download Demand Forecast Report (PDF)"):
        pdf = generate_pdf(
            {
                "Best Model": best_model,
                "Average Demand": round(avg_demand, 2),
                "RMSE": round(best_rmse, 2)
            },
            insights
        )
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f)

    # ----------------------------------
    # HUGGING FACE CHATBOT
    # ----------------------------------
    st.divider()
    st.subheader("🤖 AI Demand Assistant (Open-Source LLM)")

    hf_context = f"""
Best Model: {best_model}
Average Demand: {avg_demand:.2f}
Peak Demand: {peak_demand:.2f}
Volatility (%): {volatility_pct:.2f}
RMSE: {best_rmse:.2f}
Insights: {"; ".join(insights)}
"""

    if "hf_chat" not in st.session_state:
        st.session_state.hf_chat = []

    user_input = st.chat_input("Ask about demand, models, risks, insights...")

    if user_input:
        reply = hf_genai_response(user_input, genai_context)
        st.session_state.hf_chat.append(
            {"user": user_input, "assistant": reply}
        )

    for chat in st.session_state.hf_chat[-10:]:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])

    st.success("✅ Demand Forecasting Module Completed Successfully")
