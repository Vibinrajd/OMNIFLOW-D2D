# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module
# MSc Data Science – MAJOR PROJECT
# ======================================================================================
# OUTPUT:
# - outputs/forecast_demand.csv   (USED BY INVENTORY MODULE)
# ======================================================================================

# ----------------------------------
# IMPORTS
# ----------------------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------
# PATH CONFIG (CRITICAL)
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FORECAST_PATH = os.path.join(OUTPUT_DIR, "forecast_demand.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_sales_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ sales.csv not found at {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

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
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    }

    results = []
    predictions = {}

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
# STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting – Intelligence Module")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    df_raw = load_sales_data()

    with st.expander("🔍 Data Overview"):
        st.write("Total Records:", len(df_raw))
        st.write("Date Range:",
                 df_raw["date"].min().date(),
                 "→",
                 df_raw["date"].max().date())

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_1", "lag_7", "rolling_7"]
    X = df[FEATURES]
    y = df["daily_sales"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # --------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------
    results_df, best_model, preds = train_models(
        X_train, y_train, X_test, y_test
    )

    # --------------------------------------------------
    # FORECAST OUTPUT
    # --------------------------------------------------
    forecast_df = df.iloc[X_test.index].copy()

    forecast_df["forecast_demand"] = preds

    std = preds.std()
    forecast_df["lower_bound"] = preds - 1.96 * std
    forecast_df["upper_bound"] = preds + 1.96 * std

    final_forecast = forecast_df[
        [
            "date",
            "product_id",
            "forecast_demand",
            "lower_bound",
            "upper_bound"
        ]
    ]

    # --------------------------------------------------
    # SAVE OUTPUT (CRITICAL)
    # --------------------------------------------------
    final_forecast.to_csv(FORECAST_PATH, index=False)

    # --------------------------------------------------
    # KPIs
    # --------------------------------------------------
    st.subheader("📊 Forecast KPIs")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Forecast",
              int(final_forecast["forecast_demand"].mean()))
    c3.metric("RMSE",
              round(results_df.iloc[0]["RMSE"], 2))

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------
    st.subheader("🤖 Model Comparison")
    st.dataframe(results_df, width="stretch")

    fig = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="RMSE Comparison Across Models"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    # --------------------------------------------------
    # FORECAST VISUAL
    # --------------------------------------------------
    product = st.selectbox(
        "Select Product",
        final_forecast["product_id"].unique()
    )

    fdf = final_forecast[final_forecast["product_id"] == product]

    fig2 = px.line(
        fdf,
        x="date",
        y="forecast_demand",
        title=f"Demand Forecast for Product {product}",
        markers=True
    )
    st.plotly_chart(fig2, width="stretch")

    # --------------------------------------------------
    # SUCCESS MESSAGE (IMPORTANT)
    # --------------------------------------------------
    st.success(
        "✅ Demand Forecasting Completed.\n\n"
        "📁 File generated: outputs/forecast_demand.csv\n"
        "➡ You can now open Inventory Optimization."
    )
