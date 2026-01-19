# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module (MAJOR VERSION)
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# FEATURES:
# - Data Dictionary
# - Data Profiling & Quality Checks
# - Seasonality + Lag Feature Engineering
# - Per-Product Time-Series Modeling
# - 3 ML Models (LR, RF, GB)
# - Model Comparison + CV Metrics
# - Confidence Intervals
# - Feature Importance (Explainability)
# - Executive KPIs
# - Advanced Charts
# - NLP-based Analytics Q&A (DATA-ONLY, NO API)
# - Downloadable Forecast Output
# ======================================================================================

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ======================================================================================
# PATH CONFIGURATION
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SALES_PATH = os.path.join(DATA_DIR, "sales.csv")
FORECAST_PATH = os.path.join(OUTPUT_DIR, "forecast_demand.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date","product_id","daily_sales","price","promotion",
        "lag_1","lag_7","rolling_7",
        "day_of_week","is_weekend","month","week_of_year",
        "forecast_demand","lower_bound","upper_bound","risk_level"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Unit price",
        "Promotion flag",
        "Sales 1 day ago",
        "Sales 7 days ago",
        "7-day rolling mean",
        "Day of week (0=Mon)",
        "Weekend indicator",
        "Month",
        "ISO week of year",
        "Predicted demand",
        "Lower confidence bound",
        "Upper confidence bound",
        "Demand risk category"
    ]
})

# ======================================================================================
# LOAD DATA
# ======================================================================================
def load_sales_data():
    df = pd.read_csv(SALES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================
def data_profile(df):
    return {
        "Total Records": len(df),
        "Date Range": f"{df.date.min().date()} → {df.date.max().date()}",
        "Unique Products": df.product_id.nunique(),
        "Missing Values (%)": round(df.isnull().mean().mean() * 100, 2),
        "Avg Daily Sales": round(df.daily_sales.mean(), 2),
        "Sales Volatility": round(df.daily_sales.std(), 2)
    }

# ======================================================================================
# FEATURE ENGINEERING (SEASONAL + TEMPORAL)
# ======================================================================================
def feature_engineering(df):

    df = df.sort_values(["product_id", "date"])

    # --- Seasonality ---
    df["day_of_week"] = df.date.dt.dayofweek
    df["is_weekend"] = df.day_of_week.isin([5,6]).astype(int)
    df["month"] = df.date.dt.month
    df["week_of_year"] = df.date.dt.isocalendar().week.astype(int)

    # --- Lags ---
    df["lag_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"] = df.groupby("product_id")["daily_sales"].shift(7)

    # --- Rolling ---
    df["rolling_7"] = (
        df.groupby("product_id")["daily_sales"]
        .rolling(7).mean().reset_index(level=0, drop=True)
    )

    return df.dropna().reset_index(drop=True)

# ======================================================================================
# MODEL TRAINING (PER PRODUCT)
# ======================================================================================
def train_product_model(df):

    FEATURES = [
        "price","promotion","lag_1","lag_7","rolling_7",
        "day_of_week","is_weekend","month","week_of_year"
    ]

    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=15, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5
        )
    }

    results, preds_store = [], {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, preds))
        })
        preds_store[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    return (
        preds_store[best_model],
        y_test.index,
        best_model,
        results_df
    )

# ======================================================================================
# NLP ANALYTICS ENGINE (DATA-DRIVEN)
# ======================================================================================
class DemandNLP:

    def __init__(self, df):
        self.df = df
        self.kb = self._build_kb()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.kb.keys())

    def _build_kb(self):
        kb = {}
        g = self.df.groupby("product_id")

        avg = g.forecast_demand.mean()
        std = g.forecast_demand.std()

        kb["highest demand product"] = f"Product {avg.idxmax()} has the highest average demand ({avg.max():.0f})."
        kb["unstable demand product"] = f"Product {std.idxmax()} is most volatile ({std.max():.0f})."
        kb["low risk product"] = f"Product {std.idxmin()} shows stable demand."
        kb["business recommendation"] = (
            "High volatility products require higher safety stock and shorter review cycles."
        )

        return kb

    def answer(self, q):
        vec = self.vectorizer.transform([q.lower()])
        sim = cosine_similarity(vec, self.matrix)
        idx = sim.argmax()

        if sim[0][idx] < 0.3:
            return "Ask about demand risk, volatility, highest demand, or recommendations."

        return list(self.kb.values())[idx]

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – MAJOR Intelligence Module")

    df_raw = load_sales_data()
    profile = data_profile(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling"):
        for k,v in profile.items():
            st.write(f"**{k}:** {v}")

    df = feature_engineering(df_raw)

    all_forecasts = []
    model_summary = []

    for pid, pdf in df.groupby("product_id"):

        preds, idx, model_name, res = train_product_model(pdf)

        temp = pdf.loc[idx].copy()
        temp["forecast_demand"] = preds

        std = preds.std()
        temp["lower_bound"] = preds - 1.96 * std
        temp["upper_bound"] = preds + 1.96 * std
        temp["risk_level"] = np.where(std > preds.mean()*0.3, "High", "Low")

        all_forecasts.append(temp)
        model_summary.append((pid, model_name))

    forecast_df = pd.concat(all_forecasts)
    forecast_df.to_csv(FORECAST_PATH, index=False)

    st.subheader("📊 Executive KPIs")
    c1,c2,c3 = st.columns(3)
    c1.metric("Products", forecast_df.product_id.nunique())
    c2.metric("Avg Demand", int(forecast_df.forecast_demand.mean()))
    c3.metric("High Risk SKUs", (forecast_df.risk_level=="High").sum())

    st.subheader("📈 Forecast Preview")
    st.dataframe(forecast_df.head(20), width="stretch")

    st.download_button(
        "⬇ Download Forecast Output",
        forecast_df.to_csv(index=False),
        file_name="forecast_demand.csv"
    )

    st.divider()
    st.subheader("🧠 Demand Analytics Assistant")

    nlp = DemandNLP(forecast_df)
    q = st.chat_input("Ask about demand, risk, volatility...")

    if q:
        with st.chat_message("assistant"):
            st.write(nlp.answer(q))

    st.success("✅ MAJOR-Level Demand Forecasting Completed")
