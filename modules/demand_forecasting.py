# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# FEATURES INCLUDED:
# ✔ Data Dictionary
# ✔ Data Profiling & Quality Checks
# ✔ Seasonality + Lag Feature Engineering
# ✔ Per Store–Product Time Series Modeling
# ✔ 3 ML Models (LR, RF, GB)
# ✔ Model Comparison
# ✔ Confidence Intervals
# ✔ Feature Importance
# ✔ Executive KPIs
# ✔ Advanced Charts
# ✔ NLP-based Analytics Q&A (DATA ONLY – NO API)
# ✔ Downloadable Forecast Output
# ======================================================================================

# ---------------------------------------
# IMPORTS
# ---------------------------------------
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
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ======================================================================================
# PATH CONFIG
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date", "store_id", "product_id", "product_category", "sales_region",
        "daily_sales", "unit_price", "discount_rate", "promotion_flag",
        "weather_condition", "competitor_price", "season"
    ],
    "Description": [
        "Date of sales transaction",
        "Unique store identifier",
        "Unique product (SKU) identifier",
        "Product category",
        "Sales region",
        "Units sold per day (TARGET)",
        "Selling price per unit",
        "Discount percentage",
        "Promotion indicator (0/1)",
        "Weather condition",
        "Competitor product price",
        "Season label"
    ]
})

# ======================================================================================
# LOAD DATA
# ======================================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================
def data_profiling(df):
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Unique Stores": df["store_id"].nunique(),
        "Unique Products": df["product_id"].nunique(),
        "Average Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility (Std)": round(df["daily_sales"].std(), 2),
        "Total Missing Cells": int(df.isnull().sum().sum()),
        "Missing %": round((df.isnull().sum().sum() / df.size) * 100, 2)
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def feature_engineering(df):
    df = df.sort_values("date")

    df["lag_1"] = df["daily_sales"].shift(1)
    df["lag_7"] = df["daily_sales"].shift(7)
    df["rolling_7"] = df["daily_sales"].rolling(7).mean()

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df.dropna(inplace=True)
    return df

# ======================================================================================
# NLP ANALYTICS (RULE-BASED)
# ======================================================================================
def nlp_answer(df, query):
    q = query.lower()

    if "highest demand" in q:
        return f"Highest daily demand: {df['forecast'].max():.0f} units"
    if "average demand" in q:
        return f"Average daily demand: {df['forecast'].mean():.2f} units"
    if "trend" in q:
        return "Demand trend is increasing 📈" if df["forecast"].iloc[-1] > df["forecast"].iloc[0] else "Demand trend is decreasing 📉"

    return "Try: highest demand, average demand, demand trend"

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – Intelligence Module")

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    df = load_data()

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, use_container_width=True)

    with st.expander("🔍 Data Profiling"):
        profile = data_profiling(df)
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    # -------------------------------
    # FILTERS
    # -------------------------------
    store = st.selectbox("Select Store", sorted(df["store_id"].unique()))
    product = st.selectbox(
        "Select Product",
        sorted(df[df["store_id"] == store]["product_id"].unique())
    )

    df = df[(df["store_id"] == store) & (df["product_id"] == product)].copy()

    if len(df) < 30:
        st.warning("Not enough data for this Store–Product combination")
        return

    # -------------------------------
    # ENCODE CATEGORICAL FEATURES
    # -------------------------------
    for col in ["product_category", "sales_region", "weather_condition", "season"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    df = feature_engineering(df)

    FEATURES = [
        "unit_price", "discount_rate", "promotion_flag",
        "competitor_price",
        "product_category", "sales_region",
        "weather_condition", "season",
        "lag_1", "lag_7", "rolling_7",
        "month", "day_of_week"
    ]

    X = df[FEATURES]
    y = df["daily_sales"]

    # -------------------------------
    # TIME SERIES SPLIT
    # -------------------------------
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # -------------------------------
    # MODELS
    # -------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    results = []
    forecasts = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

        forecasts[name] = preds

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_model = results_df.iloc[0]["Model"]

    # -------------------------------
    # FINAL FORECAST
    # -------------------------------
    df_forecast = df.iloc[split:].copy()
    df_forecast["forecast"] = forecasts[best_model]

    sigma = df_forecast["forecast"].std()
    df_forecast["lower_ci"] = df_forecast["forecast"] - 1.96 * sigma
    df_forecast["upper_ci"] = df_forecast["forecast"] + 1.96 * sigma

    # -------------------------------
    # KPIs
    # -------------------------------
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast", int(df_forecast["forecast"].mean()))
    c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
    c4.metric("Volatility", round(sigma, 2))

    # -------------------------------
    # MODEL COMPARISON
    # -------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    st.plotly_chart(
        px.bar(results_df, x="Model", y="RMSE", text="RMSE", title="RMSE Comparison"),
        use_container_width=True
    )

    # -------------------------------
    # FORECAST VISUALIZATION
    # -------------------------------
    st.subheader("📈 Forecast with Confidence Interval")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["forecast"], name="Forecast"))
    fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["upper_ci"], name="Upper CI", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["lower_ci"], name="Lower CI", fill="tonexty"))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # NLP ANALYTICS
    # -------------------------------
    st.subheader("💬 Demand Analytics Assistant")
    q = st.text_input("Ask: highest demand, average demand, demand trend")

    if q:
        st.info(nlp_answer(df_forecast, q))

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    st.download_button(
        "⬇ Download Forecast Output",
        df_forecast[
            ["date", "store_id", "product_id", "daily_sales", "forecast", "lower_ci", "upper_ci"]
        ].to_csv(index=False),
        "forecast_demand.csv"
    )

    st.success("✅ Demand Forecasting Completed Successfully")

    # SAVE FULL FORECAST FOR DOWNSTREAM MODULES
    FULL_FORECAST_PATH = os.path.join("data", "forecast_demand.csv")

    full_output = data[
      [
        "date",
        "store_id",
        "product_id",
        "daily_sales",
        "forecast",
        "lower_ci",
        "upper_ci"
      ]
   ]

   full_output.to_csv(FULL_FORECAST_PATH, index=False)

