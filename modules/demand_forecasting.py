# ======================================================================================
# OmniFlow-D2D : Demand Forecasting & Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================
# PATH CONFIGURATION
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FORECAST_PATH = os.path.join(OUTPUT_DIR, "forecast_demand.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================
DATA_DICTIONARY = {
    "date": "Transaction date",
    "product_id": "Unique product identifier",
    "daily_sales": "Units sold per day",
    "price": "Selling price per unit",
    "promotion": "Promotion flag (0/1)",
    "lag_1": "Previous day sales",
    "lag_7": "Sales 7 days ago",
    "rolling_7": "7-day rolling average demand",
    "forecast_demand": "Predicted demand",
    "lower_bound": "Lower confidence interval",
    "upper_bound": "Upper confidence interval"
}

# ======================================================================================
# LOAD DATA
# ======================================================================================
@st.cache_data
def load_sales():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def engineer_features(df):
    df = df.sort_values(["product_id", "date"])
    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_7"] = (
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
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        )
    }

    results, predictions = [], {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": math.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        predictions[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]
    return results_df, best_model, predictions[best_model]

# ======================================================================================
# DEMAND ANALYTICS LIBRARY (CORE INTELLIGENCE)
# ======================================================================================
class DemandAnalytics:

    def __init__(self, forecast_df, raw_df):
        self.f = forecast_df
        self.r = raw_df

    def avg_demand(self):
        return self.f.groupby("product_id")["forecast_demand"].mean()

    def volatility(self):
        return self.f.groupby("product_id")["forecast_demand"].std()

    def confidence_width(self):
        return (self.f["upper_bound"] - self.f["lower_bound"]).groupby(
            self.f["product_id"]
        ).mean()

    def stability_score(self):
        return self.avg_demand() / self.volatility()

    def trend_direction(self):
        t = self.r.groupby("date")["daily_sales"].sum()
        return "increasing" if t.iloc[-1] > t.iloc[0] else "decreasing"

# ======================================================================================
# NLP QUESTION ENGINE (NO API, REAL NLP)
# ======================================================================================
class DemandNLP:

    def __init__(self, analytics: DemandAnalytics):
        self.a = analytics
        self.knowledge = self._build_knowledge_base()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.knowledge.keys())

    def _build_knowledge_base(self):
        avg = self.a.avg_demand()
        vol = self.a.volatility()
        ci = self.a.confidence_width()
        stab = self.a.stability_score()

        kb = {}

        for p in avg.index:
            kb[f"average demand product {p}"] = (
                f"Product {p} average demand is {avg[p]:.2f} units."
            )
            kb[f"volatile demand product {p}"] = (
                f"Product {p} has volatility {vol[p]:.2f} units."
            )
            kb[f"confidence interval product {p}"] = (
                f"Product {p} confidence interval width is {ci[p]:.2f} units."
            )
            kb[f"stable demand product {p}"] = (
                f"Product {p} stability score is {stab[p]:.2f}."
            )

        # global insights
        kb["highest demand product"] = f"Highest demand product is {avg.idxmax()}."
        kb["lowest demand product"] = f"Lowest demand product is {avg.idxmin()}."
        kb["most unstable demand"] = f"Most unstable demand is product {vol.idxmax()}."
        kb["sales trend"] = f"Overall sales trend is {self.a.trend_direction()}."

        return kb

    def answer(self, question):
        q_vec = self.vectorizer.transform([question.lower()])
        sim = cosine_similarity(q_vec, self.matrix)
        idx = sim.argmax()

        if sim[0][idx] < 0.25:
            return (
                "I could not infer a precise answer.\n\n"
                "Try questions like:\n"
                "- Which product has unstable demand?\n"
                "- Highest demand product?\n"
                "- Confidence interval for product 1002\n"
                "- Sales trend\n"
            )

        key = list(self.knowledge.keys())[idx]
        return self.knowledge[key]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting & Demand Intelligence")

    # ---------------- DATA ----------------
    raw_df = load_sales()
    df = engineer_features(raw_df)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]
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

    forecast_df.to_csv(FORECAST_PATH, index=False)

    # ---------------- KPIs ----------------
    st.subheader("📊 KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(forecast_df["forecast_demand"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))

    # ---------------- CHART ----------------
    product = st.selectbox("Product", forecast_df["product_id"].unique())
    fdf = forecast_df[forecast_df["product_id"] == product]

    fig = px.line(
        fdf,
        x="date",
        y="forecast_demand",
        markers=True,
        title=f"Forecast Demand – Product {product}"
    )
    st.plotly_chart(fig, width="stretch")

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "⬇ Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name="forecast_demand.csv"
    )

    # ---------------- NLP Q&A ----------------
    analytics = DemandAnalytics(forecast_df, raw_df)
    nlp = DemandNLP(analytics)

    st.divider()
    st.subheader("🤖 Demand Intelligence Assistant (Data-Driven NLP)")

    q = st.chat_input("Ask any demand-related question...")
    if q:
        ans = nlp.answer(q)
        with st.chat_message("assistant"):
            st.write(ans)

    st.success("Forecast generated, saved, and ready for Inventory Optimization.")
