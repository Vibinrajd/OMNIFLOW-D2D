# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module
# SKU–Location Level (Store × Product)
# MSc Data Science – Major Project
# ======================================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ======================================================================================
# STREAMLIT PAGE
# ======================================================================================

def demand_forecasting_page():

    st.title("📊 Demand Forecasting Intelligence")
    st.caption("OmniFlow-D2D | SKU–Location Level Demand Forecasting")

    # ==============================================================================
    # LOAD DATA
    # ==============================================================================
    @st.cache_data
    def load_data():
        return pd.read_csv("data/sales.csv")

    df = load_data()

    # ==============================================================================
    # COLUMN STANDARDIZATION
    # ==============================================================================
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df['date'] = pd.to_datetime(df['date'])

    # ==============================================================================
    # REQUIRED COLUMN VALIDATION
    # ==============================================================================
    required_cols = ['date', 'store_id', 'product_id', 'daily_sales']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.error(f"Dataset missing required columns: {missing_cols}")
        st.stop()

    # ==============================================================================
    # DATA DICTIONARY  ✅ (NEW – IMPORTANT)
    # ==============================================================================
    st.subheader("📘 Data Dictionary")

    data_dictionary = pd.DataFrame({
        "Column Name": [
            "date",
            "store_id",
            "product_id",
            "product_category",
            "sales_region",
            "daily_sales",
            "unit_price",
            "discount_rate",
            "promotion_flag",
            "weather_condition",
            "competitor_price",
            "season"
        ],
        "Description": [
            "Date of sales transaction",
            "Unique store / outlet identifier",
            "Unique product (SKU) identifier",
            "Product category classification",
            "Geographical sales region",
            "Units sold per day (target variable)",
            "Selling price per unit",
            "Discount percentage applied",
            "Promotion indicator (0 = No, 1 = Yes)",
            "Weather condition on the day",
            "Competitor product price",
            "Seasonal label (Summer, Winter, etc.)"
        ],
        "Usage in Model": [
            "Time index",
            "SKU–Location granularity",
            "SKU-level forecasting",
            "Demand pattern grouping",
            "Regional seasonality effect",
            "Prediction target",
            "Price sensitivity",
            "Promotion impact",
            "Demand uplift factor",
            "External demand driver",
            "Competitive pressure",
            "Explicit seasonality signal"
        ]
    })

    st.dataframe(data_dictionary, use_container_width=True)

    # ==============================================================================
    # DATA PROFILING
    # ==============================================================================
    st.subheader("🔍 Data Profiling & Quality Checks")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.isna().sum())

    # ==============================================================================
    # STORE & PRODUCT SELECTION
    # ==============================================================================
    st.subheader("🏬 SKU–Location Selection")

    store_id = st.selectbox(
        "Select Store ID",
        sorted(df['store_id'].unique())
    )

    product_id = st.selectbox(
        "Select Product ID",
        sorted(df[df['store_id'] == store_id]['product_id'].unique())
    )

    # ==============================================================================
    # LOCK BASE DATA (NEVER MODIFY)
    # ==============================================================================
    base_data = df[
        (df['store_id'] == store_id) &
        (df['product_id'] == product_id)
    ].sort_values("date").copy()

    if len(base_data) < 30:
        st.error("Not enough historical data for this Store–Product combination.")
        return

    # ==============================================================================
    # ENCODE CATEGORICAL VARIABLES
    # ==============================================================================
    categorical_cols = [
        'sales_region',
        'weather_condition',
        'season',
        'product_category'
    ]

    for col in categorical_cols:
        if col in base_data.columns:
            le = LabelEncoder()
            base_data[col] = le.fit_transform(base_data[col])

    # ==============================================================================
    # FEATURE ENGINEERING
    # ==============================================================================
    st.subheader("⚙️ Feature Engineering")

    data = base_data.copy()

    data['lag_1'] = data['daily_sales'].shift(1)
    data['lag_7'] = data['daily_sales'].shift(7)
    data['rolling_7'] = data['daily_sales'].rolling(7).mean()

    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek

    data.dropna(inplace=True)

    st.success("Lag, rolling mean & seasonal features created")

    # ==============================================================================
    # FEATURE SET
    # ==============================================================================
    features = [
        'unit_price',
        'discount_rate',
        'promotion_flag',
        'competitor_price',
        'sales_region',
        'weather_condition',
        'season',
        'lag_1',
        'lag_7',
        'rolling_7',
        'month',
        'day_of_week'
    ]

    features = [f for f in features if f in data.columns]

    X = data[features]
    y = data['daily_sales']

    # ==============================================================================
    # MODEL DEFINITIONS
    # ==============================================================================
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    tscv = TimeSeriesSplit(n_splits=5)
    model_scores = {}

    st.subheader("🤖 Model Comparison (Time-Series CV)")

    for name, model in models.items():
        mae = -cross_val_score(
            model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
        ).mean()
        model_scores[name] = mae

    results_df = pd.DataFrame.from_dict(
        model_scores, orient="index", columns=["MAE"]
    ).sort_values("MAE")

    st.dataframe(results_df)

    best_model_name = results_df.index[0]
    best_model = models[best_model_name]

    st.success(f"Best Model Selected → **{best_model_name}**")

    # ==============================================================================
    # FINAL TRAINING & FORECAST
    # ==============================================================================
    best_model.fit(X, y)
    data['forecast'] = best_model.predict(X)

    residuals = y - data['forecast']
    sigma = residuals.std()

    data['lower_ci'] = data['forecast'] - 1.96 * sigma
    data['upper_ci'] = data['forecast'] + 1.96 * sigma

    # ==============================================================================
    # VISUALIZATION
    # ==============================================================================
    st.subheader("📈 Actual vs Forecast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['daily_sales'], name="Actual"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['forecast'], name="Forecast"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['upper_ci'], name="Upper CI", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=data['date'], y=data['lower_ci'], name="Lower CI", line=dict(dash="dot")))

    st.plotly_chart(fig, use_container_width=True)

    # ==============================================================================
    # EXECUTIVE KPIs
    # ==============================================================================
    st.subheader("📌 Executive KPIs")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(mean_absolute_error(y, data['forecast']), 2))
    col2.metric("RMSE", round(np.sqrt(mean_squared_error(y, data['forecast'])), 2))
    col3.metric("R² Score", round(r2_score(y, data['forecast']), 2))

    # ==============================================================================
    # DOWNLOAD OUTPUT
    # ==============================================================================
    st.subheader("⬇️ Download Demand Forecast")

    output = data[
        ['date', 'store_id', 'product_id',
         'daily_sales', 'forecast',
         'lower_ci', 'upper_ci']
    ]

    st.download_button(
        "Download Forecast CSV",
        output.to_csv(index=False),
        "demand_forecast.csv",
        "text/csv"
    )
