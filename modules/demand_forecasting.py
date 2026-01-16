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
        "Predicted future demand",
        "Lower confidence interval",
        "Upper confidence interval"
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
        "Average Daily Sales": round(df["daily_sales"].mean(), 2),
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
# LOCAL AI ENGINE (NO API)
# ======================================================================================
def local_ai_response(question, metrics):
    q = question.lower()

    avg = metrics["avg"]
    peak = metrics["peak"]
    vol = metrics["vol"]
    rmse = metrics["rmse"]
    model = metrics["model"]

    if "model" in q:
        return f"{model} was selected because it achieved the lowest RMSE of {rmse:.2f}, indicating superior predictive accuracy."

    if "accuracy" in q or "reliable" in q:
        return f"The model is reliable with an RMSE of {rmse:.2f}. Lower RMSE indicates reduced forecast error."

    if "risk" in q or "stock" in q:
        return (
            "High stock-out risk detected due to high demand volatility."
            if vol > 30 else
            "Stock-out risk is moderate and manageable."
        )

    if "volatility" in q or "stable" in q:
        return (
            f"Demand volatility is {vol:.2f}%. "
            + ("This indicates unstable demand." if vol > 30 else "Demand is relatively stable.")
        )

    if "confidence" in q:
        return "Confidence intervals represent forecast uncertainty. Wider intervals indicate higher planning risk."

    if "recommend" in q or "action" in q:
        return "It is recommended to maintain safety stock during peak demand and monitor volatility trends."

    return (
        "I can explain model choice, demand risk, volatility, confidence intervals, "
        "and business recommendations."
    )

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
    story.append(Paragraph("AI Insights", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(i, styles["Normal"]))

    doc.build(story)
    return path

# ======================================================================================
# MAIN STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – AI Intelligence Module")

    df_raw = load_data()
    profile = data_profiling(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_sales_1", "lag_sales_7", "rolling_mean_7"]
    X, y = df[FEATURES], df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = preds
    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std

    # KPIs
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Demand", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility (%)", round((forecast_df["forecast_demand"].std() /
                                       forecast_df["forecast_demand"].mean()) * 100, 2))

    # Model Comparison
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig_rmse = px.bar(results_df, x="Model", y="RMSE", text="RMSE",
                      title="Model RMSE Comparison")
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, width="stretch")

    # Filter
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # Charts
    st.subheader("📈 Forecast Analysis")

    charts = [
        px.line(fdf, x="date", y="forecast_demand", markers=True,
                title="Forecast Trend"),
        px.line(fdf, x="date", y="rolling_mean_7", title="Rolling Mean"),
        px.histogram(fdf, x="forecast_demand", title="Demand Distribution"),
        px.box(fdf, y="forecast_demand", title="Demand Volatility")
    ]

    ci_fig = go.Figure()
    ci_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    ci_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI"))
    ci_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], name="Lower CI",
                                fill="tonexty"))
    ci_fig.update_layout(title="Forecast with Confidence Interval")
    charts.append(ci_fig)

    for fig in charts:
        st.plotly_chart(fig, width="stretch")

    # Insights
    avg = fdf["forecast_demand"].mean()
    peak = fdf["forecast_demand"].max()
    vol = (fdf["forecast_demand"].std() / avg) * 100
    rmse = results_df.iloc[0]["RMSE"]

    insights = [
        f"Average forecast demand is {avg:.0f} units.",
        f"Peak demand reaches {peak:.0f} units.",
        f"Demand volatility is {vol:.2f}%.",
        f"{best_model} provides the most accurate forecasts (RMSE {rmse:.2f})."
    ]

    st.subheader("🤖 AI Insights")
    for i in insights:
        st.write("•", i)

    # PDF
    if st.button("📥 Download Forecast Report (PDF)"):
        pdf = generate_pdf(
            {
                "Best Model": best_model,
                "Average Demand": round(avg, 2),
                "RMSE": round(rmse, 2),
                "Volatility (%)": round(vol, 2)
            },
            insights
        )
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f)

    # Offline Chat
    st.divider()
    st.subheader("🤖 AI Demand Assistant (Offline)")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_q = st.chat_input("Ask about model, risk, volatility, confidence interval")

    if user_q:
        reply = local_ai_response(
            user_q,
            {"avg": avg, "peak": peak, "vol": vol, "rmse": rmse, "model": best_model}
        )
        st.session_state.chat.append((user_q, reply))

    for q, a in st.session_state.chat[-10:]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
