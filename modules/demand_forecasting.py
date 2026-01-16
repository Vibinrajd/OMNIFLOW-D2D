# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# FEATURES (UNCHANGED):
# - Data Dictionary
# - Data Profiling & Quality Checks
# - Feature Engineering
# - 3 ML Models
# - Model Comparison
# - Confidence Intervals
# - KPI Cards
# - Filters
# - Advanced Charts
# - NLP-based Q&A (NO API, NO LIMITS)
# ======================================================================================

# ------------------------------
# IMPORTS
# ------------------------------
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ------------------------------
# CONFIG
# ------------------------------
DATA_PATH = "data/sales.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA DICTIONARY (UNCHANGED)
# ======================================================================================
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date",
        "product_id",
        "daily_sales",
        "price",
        "promotion",
        "lag_sales_1",
        "lag_sales_7",
        "rolling_mean_7",
        "forecast_demand",
        "lower_bound",
        "upper_bound"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Unit selling price",
        "Promotion flag (0 = No, 1 = Yes)",
        "Previous day demand",
        "Demand 7 days ago",
        "7-day rolling average demand",
        "Predicted demand from ML model",
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
        "Average Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2)
    }

# ======================================================================================
# FEATURE ENGINEERING (UNCHANGED)
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
# MODEL TRAINING (UNCHANGED)
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
# NLP KNOWLEDGE BASE (RAW + FORECAST DATA)
# ======================================================================================
def build_knowledge_base(raw_df, forecast_df, model_name, metrics_df):

    knowledge = []

    # ---- RAW SALES KNOWLEDGE
    for pid, g in raw_df.groupby("product_id"):
        knowledge.append(
            f"Product {pid} has average daily sales of {g['daily_sales'].mean():.2f} units "
            f"with volatility {g['daily_sales'].std():.2f} units."
        )

    # ---- FORECAST KNOWLEDGE
    for pid, g in forecast_df.groupby("product_id"):
        knowledge.append(
            f"Product {pid} forecast average demand is {g['forecast_demand'].mean():.2f} units "
            f"with confidence interval width {(g['upper_bound'] - g['lower_bound']).mean():.2f} units."
        )

    # ---- MODEL KNOWLEDGE
    for _, r in metrics_df.iterrows():
        knowledge.append(
            f"Model {r['Model']} achieved RMSE {r['RMSE']:.2f} and R2 score {r['R2']:.2f}."
        )

    knowledge.append(f"The best performing model selected was {model_name}.")

    return knowledge

# ======================================================================================
# NLP QUESTION ANSWER ENGINE (NO RULES, NO API)
# ======================================================================================
class SalesNLP:

    def __init__(self, knowledge_sentences):
        self.sentences = knowledge_sentences
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.sentences)

    def answer(self, question, top_k=4):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        idx = scores.argsort()[-top_k:][::-1]
        responses = [self.sentences[i] for i in idx]

        reply = "Based on sales data analysis:\n\n"
        for r in responses:
            reply += f"• {r}\n"

        return reply
        
# ======================================================================================
# ANALYTICAL QUESTION HANDLERS (DATA-DRIVEN LOGIC)
# ======================================================================================

def find_unstable_product(raw_df):
    """
    Identifies product with highest demand instability using CV.
    """
    stats = (
        raw_df.groupby("product_id")["daily_sales"]
        .agg(["mean", "std"])
        .reset_index()
    )

    stats["cv"] = stats["std"] / stats["mean"]

    worst = stats.sort_values("cv", ascending=False).iloc[0]

    return (
        f"Product {int(worst['product_id'])} has the most unstable demand. "
        f"It shows high volatility of {worst['std']:.2f} units and a "
        f"coefficient of variation of {worst['cv']:.2f}, "
        f"indicating large fluctuations in daily sales."
    )


def find_stable_product(raw_df):
    stats = (
        raw_df.groupby("product_id")["daily_sales"]
        .agg(["mean", "std"])
        .reset_index()
    )

    stats["cv"] = stats["std"] / stats["mean"]

    best = stats.sort_values("cv", ascending=True).iloc[0]

    return (
        f"Product {int(best['product_id'])} has the most stable demand. "
        f"It shows low volatility ({best['std']:.2f} units) and consistent sales."
    )


def highest_demand_product(raw_df):
    avg_sales = (
        raw_df.groupby("product_id")["daily_sales"]
        .mean()
        .reset_index()
        .sort_values("daily_sales", ascending=False)
        .iloc[0]
    )

    return (
        f"Product {int(avg_sales['product_id'])} has the highest average demand "
        f"at {avg_sales['daily_sales']:.2f} units per day."
    )

# ======================================================================================
# MAIN STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – Data Intelligence Module")

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    raw_df = load_data()
    profile = data_profiling(raw_df)

    # ------------------------------
    # DATA DICTIONARY
    # ------------------------------
    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    # ------------------------------
    # DATA PROFILING
    # ------------------------------
    with st.expander("🔍 Data Profiling & Quality Checks"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # ------------------------------
    # FEATURE ENGINEERING
    # ------------------------------
    df = feature_engineering(raw_df)

    FEATURES = ["price", "promotion", "lag_sales_1", "lag_sales_7", "rolling_mean_7"]
    X, y = df[FEATURES], df["daily_sales"]

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
    # KPIs
    # ------------------------------
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(forecast_df["forecast_demand"].std(), 2))

    # ------------------------------
    # MODEL COMPARISON
    # ------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig_rmse = px.bar(
        results_df, x="Model", y="RMSE",
        text="RMSE", title="RMSE Comparison"
    )
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, width="stretch")

    # ------------------------------
    # FILTER
    # ------------------------------
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # ------------------------------
    # CHARTS
    # ------------------------------
    st.subheader("📈 Forecast Analysis")

    fig1 = px.line(
        fdf, x="date", y="forecast_demand",
        markers=True, title="Forecast Trend"
    )

    fig2 = px.line(
        fdf, x="date", y="rolling_mean_7",
        title="Rolling Mean Demand"
    )

    fig3 = px.histogram(
        fdf, x="forecast_demand",
        title="Forecast Distribution"
    )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=fdf["date"], y=fdf["forecast_demand"],
        mode="lines+markers", name="Forecast"
    ))
    fig4.add_trace(go.Scatter(
        x=fdf["date"], y=fdf["upper_bound"],
        name="Upper CI", line=dict(dash="dot")
    ))
    fig4.add_trace(go.Scatter(
        x=fdf["date"], y=fdf["lower_bound"],
        name="Lower CI", fill="tonexty"
    ))
    fig4.update_layout(title="Forecast with Confidence Interval")

    fig5 = px.box(fdf, y="forecast_demand", title="Demand Volatility")

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        st.plotly_chart(fig, width="stretch")

    # ------------------------------
    # NLP-BASED Q&A (CORE REQUIREMENT)
    # ------------------------------
    st.divider()
    st.subheader("🧠 Data-Driven Question Answering (NLP)")

    knowledge = build_knowledge_base(raw_df, forecast_df, best_model, results_df)
    nlp_engine = SalesNLP(knowledge)

    question = st.text_input(
        "Ask ANY question related to sales or demand:",
        placeholder="Which product has unstable demand?"
    )

    if question:
        answer = nlp_engine.answer(question)
        st.text_area("Answer", answer, height=220)

    st.success("✅ Demand Forecasting & NLP Intelligence Ready")
