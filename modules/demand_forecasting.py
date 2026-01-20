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
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ======================================================================================
# GLOBAL CSS (UI POLISH)
# ======================================================================================
st.markdown("""
<style>

/* ---------------- GLOBAL ---------------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

section.main > div {
    padding-top: 1rem;
}

/* ---------------- HEADERS ---------------- */
h1, h2, h3 {
    letter-spacing: 0.4px;
}

/* ---------------- KPI METRICS ---------------- */
[data-testid="metric-container"] {
    background: #0e1117;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 30px rgba(0,0,0,0.55);
}

[data-testid="metric-label"] {
    font-size: 14px;
    opacity: 0.75;
}

[data-testid="metric-value"] {
    font-size: 28px;
    font-weight: 600;
}

/* ---------------- BUTTONS ---------------- */
.stDownloadButton button,
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4);
}

.stDownloadButton button:hover,
.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 24px rgba(79,70,229,0.65);
}

/* ---------------- SELECTBOX ---------------- */
.stSelectbox > div > div {
    border-radius: 10px;
}

/* ---------------- EXPANDERS ---------------- */
details {
    background: #0e1117;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

details[open] {
    box-shadow: 0 10px 26px rgba(0,0,0,0.45);
}

/* ---------------- DATAFRAMES ---------------- */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 26px rgba(0,0,0,0.4);
}

/* ---------------- TABS ---------------- */
button[data-baseweb="tab"] {
    font-size: 15px;
    padding: 10px 18px;
}

button[data-baseweb="tab"]:hover {
    background-color: rgba(99,102,241,0.18);
    border-radius: 10px;
}

/* ---------------- TOOLTIP ---------------- */
.tooltip {
    position: relative;
    cursor: pointer;
}

.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    background: #111827;
    color: #fff;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
}

.tooltip:hover::after {
    opacity: 1;
}

</style>
""", unsafe_allow_html=True)

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
# PATH CONFIG
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

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
# NLP ANALYTICS
# ======================================================================================
def nlp_answer(df, query):
    q = query.lower()

    if "highest" in q:
        return f"Highest daily demand forecast: {df['forecast'].max():.0f} units"
    if "average" in q:
        return f"Average daily demand forecast: {df['forecast'].mean():.2f} units"
    if "trend" in q:
        return (
            "Demand trend is increasing 📈"
            if df["forecast"].iloc[-1] > df["forecast"].iloc[0]
            else "Demand trend is decreasing 📉"
        )
    return "Try: highest demand, average demand, demand trend"

# ======================================================================================
# MAIN PAGE
# ======================================================================================
def demand_forecasting_page():

    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ---------------- OVERVIEW ----------------
    with tab1:
        st.subheader("OMNIFLOW D2D : Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System")

        st.markdown("""
        OmniFlow D2D is an AI-driven enterprise platform integrating demand forecasting,
        supply chain planning, logistics optimization, and Gen-AI intelligence into a
        single closed-loop decision system.
        """)

    # ---------------- APPLICATION ----------------
    with tab2:
        st.header("Demand Forecasting – Intelligence Module")

        df = load_data()

        with st.expander("Data Dictionary"):
            st.dataframe(DATA_DICTIONARY, use_container_width=True)

        with st.expander("Data Profiling"):
            profile = data_profiling(df)
            for k, v in profile.items():
                st.write(f"**{k}:** {v}")

        store = st.selectbox("Select Store", sorted(df["store_id"].unique()))
        product = st.selectbox(
            "Select Product",
            sorted(df[df["store_id"] == store]["product_id"].unique())
        )

        df = df[(df["store_id"] == store) & (df["product_id"] == product)].copy()

        if len(df) < 30:
            st.warning("Not enough data for this Store–Product combination")
            return

        for col in ["product_category", "sales_region", "weather_condition", "season"]:
            df[col] = LabelEncoder().fit_transform(df[col])

        df = feature_engineering(df)

        FEATURES = [
            "unit_price", "discount_rate", "promotion_flag", "competitor_price",
            "product_category", "sales_region", "weather_condition", "season",
            "lag_1", "lag_7", "rolling_7", "month", "day_of_week"
        ]

        X, y = df[FEATURES], df["daily_sales"]

        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
        }

        results, forecasts = [], {}

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

        df_forecast = df.iloc[split:].copy()
        df_forecast["forecast"] = forecasts[best_model]

        sigma = df_forecast["forecast"].std()
        df_forecast["lower_ci"] = df_forecast["forecast"] - 1.96 * sigma
        df_forecast["upper_ci"] = df_forecast["forecast"] + 1.96 * sigma

        st.subheader("Executive KPIs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", best_model)
        c2.metric("Avg Forecast", int(df_forecast["forecast"].mean()))
        c3.metric("RMSE", round(results_df.iloc[0]["RMSE"], 2))
        c4.metric("Volatility", round(sigma, 2))

        st.subheader("Forecast with Confidence Interval")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["forecast"], name="Forecast"))
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["upper_ci"], name="Upper CI", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["lower_ci"], fill="tonexty", name="Lower CI"))

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💬 Demand Analytics Assistant")
        q = st.text_input("Ask: highest demand, average demand, demand trend")
        if q:
            st.info(nlp_answer(df_forecast, q))
