# ======================================================================================
# OmniFlow-D2D : Demand Forecasting + NLP Intelligence Module
# MSc Data Science – MAJOR PROJECT
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
        "Products": df["product_id"].nunique(),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Average Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2),
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def feature_engineering(df):
    df = df.sort_values(["product_id", "date"])

    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_7"] = (
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
            n_estimators=200, max_depth=15, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, random_state=42
        )
    }

    results = []
    preds_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        preds_dict[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    return results_df, best_model, preds_dict[best_model]

# ======================================================================================
# NLP KNOWLEDGE BASE (CORE)
# ======================================================================================
def build_knowledge_base(df):
    """
    Converts numeric sales data into natural language facts
    """
    knowledge = []

    for pid, g in df.groupby("product_id"):
        avg = g["daily_sales"].mean()
        std = g["daily_sales"].std()
        min_sale = g["daily_sales"].min()
        max_sale = g["daily_sales"].max()

        trend = (
            "increasing"
            if g["daily_sales"].iloc[-1] > g["daily_sales"].iloc[0]
            else "decreasing"
        )

        knowledge.append(
            f"Product {pid} has average daily sales of {avg:.2f} units "
            f"with volatility {std:.2f}. Sales range between {min_sale:.0f} "
            f"and {max_sale:.0f} units and show a {trend} trend."
        )

    overall_avg = df["daily_sales"].mean()
    overall_std = df["daily_sales"].std()

    knowledge.append(
        f"Overall average sales across all products is {overall_avg:.2f} units "
        f"with overall volatility {overall_std:.2f}."
    )

    return knowledge

# ======================================================================================
# NLP QUESTION ANSWERING ENGINE
# ======================================================================================
class SalesNLPQnA:
    def __init__(self, knowledge_sentences):
        self.sentences = knowledge_sentences
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.sentences)

    def answer(self, question, top_k=3):
        q_vec = self.vectorizer.transform([question])
        similarity = cosine_similarity(q_vec, self.matrix).flatten()

        top_idx = similarity.argsort()[-top_k:][::-1]
        matched = [self.sentences[i] for i in top_idx]

        response = "Based on sales data analysis:\n\n"
        for m in matched:
            response += f"• {m}\n"

        return response

# ======================================================================================
# MAIN STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.title("📈 Demand Forecasting & NLP Intelligence")
    st.caption("Pure NLP-based question answering from sales data (No APIs)")

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    df_raw = load_data()
    profile = data_profiling(df_raw)

    with st.expander("📊 Dataset Profile"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # ------------------------------
    # FEATURE ENGINEERING
    # ------------------------------
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]
    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ------------------------------
    # MODEL TRAINING
    # ------------------------------
    results_df, best_model, preds = train_models(
        X_train, y_train, X_test, y_test
    )

    forecast_df = df.iloc[X_test.index].copy()
    forecast_df["forecast"] = preds

    # ------------------------------
    # KPI CARDS
    # ------------------------------
    st.subheader("📊 Executive KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))

    # ------------------------------
    # MODEL COMPARISON
    # ------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig_rmse = px.bar(
        results_df, x="Model", y="RMSE",
        title="RMSE Comparison", text="RMSE"
    )
    fig_rmse.update_traces(textposition="outside")
    st.plotly_chart(fig_rmse, width="stretch")

    # ------------------------------
    # FORECAST CHART
    # ------------------------------
    product = st.selectbox(
        "Select Product", forecast_df["product_id"].unique()
    )
    fdf = forecast_df[forecast_df["product_id"] == product]

    fig_forecast = px.line(
        fdf, x="date", y="forecast",
        markers=True, title="Forecast Trend"
    )
    st.plotly_chart(fig_forecast, width="stretch")

    # ==================================================================================
    # NLP QUESTION ANSWERING
    # ==================================================================================
    st.divider()
    st.subheader("🧠 Ask Anything About Sales Data")

    knowledge = build_knowledge_base(df_raw)
    qna_engine = SalesNLPQnA(knowledge)

    st.caption(
        "Examples:\n"
        "- Which product has low sales?\n"
        "- Highest demand product\n"
        "- Which products are volatile?\n"
        "- Products with decreasing trend\n"
        "- Compare products"
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    question = st.chat_input("Ask a question related to sales data...")

    if question:
        answer = qna_engine.answer(question)
        st.session_state.chat.append((question, answer))

    for q, a in st.session_state.chat[-10:]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
