# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

# ------------------------------
# LIBRARIES
# ------------------------------
import os
import warnings
from openai import OpenAI

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
        "Lower confidence bound (risk lower limit)",
        "Upper confidence bound (risk upper limit)"
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
        "Zero Sales Ratio (%)": round((df["daily_sales"] == 0).mean() * 100, 2),
        "Average Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Standard Deviation": round(df["daily_sales"].std(), 2)
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

    story.append(Paragraph("<b>OmniFlow-D2D Demand Forecast Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>AI-Driven Analytical Insights</b>", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(ins, styles["Normal"]))

    doc.build(story)
    return path




    # ======================================================================================
    # GEN-AI HELPER FUNCTION
    # ======================================================================================
    
    def genai_response(user_query, context):
        """
        Context-aware GenAI response generator
        """
    
        # Read API key safely
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    
        if api_key is None:
            return "⚠️ OpenAI API key not configured. Please add it to Streamlit secrets."
    
        client = OpenAI(api_key=api_key)
    
        system_prompt = f"""
        You are an AI supply-chain analyst.
    
        Context (do NOT hallucinate beyond this):
        {context}
    
        Rules:
        - Answer only using the context
        - Be analytical and business-focused
        - Explain ML results and risks clearly
        - If data is insufficient, say so
        """
    
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2
        )
    
        return response.choices[0].message.content

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
    # MODEL TRAINING
    # ------------------------------
    results_df, best_model, preds = train_models(X_train, y_train, X_test, y_test)

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast_demand"] = preds

    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std


    
    # ------------------------------
    # KPI CARDS
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
        text="RMSE",
        title="RMSE Comparison Across Models"
    )
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, use_container_width=True)

    # ------------------------------
    # FILTERS
    # ------------------------------
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # ------------------------------
    # ADVANCED CHARTS WITH DATA LABELS
    # ------------------------------
    st.subheader("📈 Forecast Analysis")

    fig1 = px.line(
        fdf,
        x="date",
        y="forecast_demand",
        markers=True,
        text=fdf["forecast_demand"].round(0),
        title="Forecast Trend with Data Labels"
    )
    fig1.update_traces(textposition="top center")

    fig2 = px.line(
        fdf,
        x="date",
        y="rolling_mean_7",
        title="Rolling Mean Demand Trend"
    )

    fig3 = px.histogram(
        fdf,
        x="forecast_demand",
        nbins=20,
        title="Forecast Demand Distribution"
    )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=fdf["date"],
        y=fdf["forecast_demand"],
        mode="lines+markers",
        name="Forecast",
        text=fdf["forecast_demand"].round(0)
    ))
    fig4.add_trace(go.Scatter(
        x=fdf["date"],
        y=fdf["upper_bound"],
        name="Upper CI",
        line=dict(dash="dot")
    ))
    fig4.add_trace(go.Scatter(
        x=fdf["date"],
        y=fdf["lower_bound"],
        name="Lower CI",
        fill="tonexty"
    ))
    fig4.update_layout(title="Forecast with Confidence Interval")

    fig5 = px.box(
        fdf,
        y="forecast_demand",
        title="Demand Volatility Distribution"
    )

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # CHART-WISE INSIGHTS
    # ------------------------------
    st.subheader("📊 Chart-wise Analytical Insights")

    trend_change = fdf["forecast_demand"].iloc[-1] - fdf["forecast_demand"].iloc[0]
    volatility_pct = (fdf["forecast_demand"].std() / fdf["forecast_demand"].mean()) * 100
    ci_width = (fdf["upper_bound"] - fdf["lower_bound"]).mean()

    st.write(f"• Forecast shows a net change of **{trend_change:.0f} units**, indicating demand trend.")
    st.write(f"• Demand volatility is **{volatility_pct:.2f}%**, impacting inventory safety stock.")
    st.write(f"• Average confidence interval width is **{ci_width:.2f} units**, representing uncertainty.")

    # ------------------------------
    # MODEL-WISE INSIGHTS
    # ------------------------------
    st.subheader("🤖 Model-wise Insights")

    best_rmse = results_df.iloc[0]["RMSE"]
    worst_rmse = results_df.iloc[-1]["RMSE"]

    st.write(
        f"• **{best_model}** achieved the lowest RMSE (**{best_rmse:.2f}**), "
        f"outperforming the weakest model by **{(worst_rmse - best_rmse):.2f} units**."
    )

    if best_model == "Random Forest":
        st.write("• Random Forest captured non-linear demand patterns and interactions.")
    elif best_model == "Gradient Boosting":
        st.write("• Gradient Boosting refined predictions through sequential error correction.")
    else:
        st.write("• Linear Regression performed well due to strong linear demand relationships.")

    # ------------------------------
    # AI-DRIVEN BUSINESS INSIGHTS
    # ------------------------------
    st.subheader("🤖 AI-Driven Decision Insights")

    insights = []
    avg_demand = fdf["forecast_demand"].mean()
    peak_demand = fdf["forecast_demand"].max()

    insights.append(
        f"Average forecast demand is **{avg_demand:.0f} units**, "
        f"with peak demand reaching **{peak_demand:.0f} units**."
    )

    if peak_demand > fdf["upper_bound"].mean():
        insights.append(
            "Peak demand exceeds expected confidence limits, indicating potential stock-out risk."
        )

    if volatility_pct > 30:
        insights.append(
            "High demand volatility suggests need for safety stock and frequent replenishment."
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
                "Average Forecast": round(avg_demand, 2),
                "RMSE": round(best_rmse, 2),
                "Volatility (%)": round(volatility_pct, 2)
            },
            insights
        )
        with open(pdf, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="Demand_Forecast_Report.pdf"
            )

    st.success("✅ Demand Forecasting Analysis Completed Successfully")

    genai_context = f"""
    Best Model: {best_model}
    Average Forecast: {avg_demand:.2f}
    Peak Demand: {peak_demand:.2f}
    Volatility (%): {volatility_pct:.2f}
    RMSE: {best_rmse:.2f}
    
    Key Insights:
    {"; ".join(insights)}
    """

    
    st.divider()
    st.subheader("🤖 GenAI Demand Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_input = st.chat_input("Ask about demand, risk, models, or insights...")
    
    if user_input:
        with st.spinner("Thinking..."):
            reply = genai_response(user_input, genai_context)
    
        st.session_state.chat_history.append(
            {"user": user_input, "assistant": reply}
        )
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])
