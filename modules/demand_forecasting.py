# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module
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
        "Selling price per unit",
        "Promotion flag (0 = No, 1 = Yes)",
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
        "Average Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2),
        "Zero Sales %": round((df["daily_sales"] == 0).mean() * 100, 2)
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
# ANALYTICAL DECISION FUNCTIONS (CRITICAL)
# ======================================================================================
def most_unstable_product(df):
    stats = df.groupby("product_id")["daily_sales"].agg(["mean", "std"]).reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    worst = stats.sort_values("cv", ascending=False).iloc[0]

    return (
        f"Product {int(worst['product_id'])} has the most unstable demand. "
        f"It has average daily sales of {worst['mean']:.2f} units and "
        f"high volatility of {worst['std']:.2f} units, indicating "
        f"significant demand fluctuations and higher risk."
    )

def most_stable_product(df):
    stats = df.groupby("product_id")["daily_sales"].agg(["mean", "std"]).reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    best = stats.sort_values("cv").iloc[0]

    return (
        f"Product {int(best['product_id'])} has the most stable demand with "
        f"low volatility of {best['std']:.2f} units relative to "
        f"average sales of {best['mean']:.2f} units."
    )

# ======================================================================================
# NLP KNOWLEDGE BASE (TF-IDF)
# ======================================================================================
def build_knowledge_base(raw_df, forecast_df):
    sentences = []

    for pid in raw_df["product_id"].unique():
        rdf = raw_df[raw_df["product_id"] == pid]
        fdf = forecast_df[forecast_df["product_id"] == pid]

        sentences.append(
            f"Product {pid} has average daily sales of {rdf['daily_sales'].mean():.2f} "
            f"units with volatility {rdf['daily_sales'].std():.2f} units."
        )

        sentences.append(
            f"Product {pid} forecast average demand is {fdf['forecast_demand'].mean():.2f} "
            f"units with confidence interval width "
            f"{(fdf['upper_bound'] - fdf['lower_bound']).mean():.2f} units."
        )

    return sentences

def nlp_answer(question, knowledge):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(knowledge + [question])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    idx = sims.argmax()
    return knowledge[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – Intelligence Module")

    # LOAD DATA
    raw_df = load_data()
    profile = data_profiling(raw_df)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # FEATURE ENGINEERING
    df = feature_engineering(raw_df)

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

    # KPI
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(forecast_df["forecast_demand"].std(), 2))

    # MODEL COMPARISON
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig = px.bar(results_df, x="Model", y="RMSE", text="RMSE")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    # FILTER
    product = st.selectbox("Select Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    # FORECAST CHART
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_demand"], name="Forecast"))
    fig2.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper_bound"], name="Upper CI"))
    fig2.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower_bound"], fill="tonexty", name="Lower CI"))
    st.plotly_chart(fig2, width="stretch")

    # =========================
    # NLP + ANALYTICS Q&A
    # =========================
    st.divider()
    st.subheader("🧠 Ask Questions from Your Data")

    knowledge = build_knowledge_base(raw_df, forecast_df)
    question = st.text_input("Ask any sales-related question:")

    if question:
        q = question.lower()

        if "unstable" in q or "volatile" in q or "risk" in q:
            answer = most_unstable_product(raw_df)
        elif "stable" in q:
            answer = most_stable_product(raw_df)
        else:
            answer = nlp_answer(question, knowledge)

        st.text_area("Answer", answer, height=180)
