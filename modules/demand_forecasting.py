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
st.set_page_config(layout="wide", page_title="OmniFlow D2D")

# ======================================================================================
# GLOBAL CSS
# ======================================================================================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
}

.block-container {
    padding-top: 2rem;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: #020617;
    border-radius: 10px;
    padding: 10px 18px;
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.25s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    box-shadow: 0 8px 25px rgba(37,99,235,0.4);
}

/* ---------- KPI CARDS ---------- */
.kpi-card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 15px 35px rgba(0,0,0,0.45);
    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 25px 45px rgba(37,99,235,0.35);
}

.kpi-title {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #e5e7eb;
}

/* ---------- BUTTONS ---------- */
.stDownloadButton button, .stButton button {
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
    box-shadow: 0 10px 25px rgba(37,99,235,0.4);
    transition: all 0.25s ease;
}

.stDownloadButton button:hover, .stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(37,99,235,0.6);
}

/* ---------- DATAFRAME ---------- */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* ---------- POPUP TOOLTIP ---------- */
.tooltip {
    position: relative;
    cursor: pointer;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 220px;
    background-color: #020617;
    color: #e5e7eb;
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    position: absolute;
    z-index: 1;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}

.tooltip:hover .tooltiptext {
    visibility: visible;
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
        return f"Highest forecasted demand: {df['forecast'].max():.0f} units"
    if "average" in q:
        return f"Average forecasted demand: {df['forecast'].mean():.2f} units"
    if "trend" in q:
        return "Demand trend is increasing 📈" if df["forecast"].iloc[-1] > df["forecast"].iloc[0] else "Demand trend is decreasing 📉"
    return "Try: highest demand, average demand, demand trend"

# ======================================================================================
# MAIN PAGE
def demand_forecasting_page():

    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ==================================================================================
    # TAB 1 : OVERVIEW (FULLY RESTORED)
    # ==================================================================================
    with tab1:

        st.markdown("""
        <div style="background:#020617;padding:28px;border-radius:18px;
                    box-shadow:0 20px 40px rgba(0,0,0,0.45)">
        """, unsafe_allow_html=True)

        st.subheader(
            "OMNIFLOW D2D : Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System"
        )

        st.markdown("""
        **ABSTRACT**

        OmniFlow D2D is an AI-driven end-to-end enterprise platform that integrates marketing demand forecasting,
        supply chain planning, manufacturing optimization, transportation logistics, and Gen-AI decision intelligence
        into a single system.

        It removes operational silos by ensuring that demand signals directly drive procurement, production schedules,
        and delivery planning. The platform functions as a closed-loop intelligence engine that predicts demand,
        optimizes execution, and continuously improves decisions using real-time data and AI.
        """)

        st.markdown("### 🛠 Tools")
        st.markdown("""
        - Python, SQL, Pandas, NumPy  
        - Scikit-Learn  
        - Time-Series Models (Prophet / SARIMA)  
        - Optimization Models (Linear Programming)  
        - Power BI / Tableau  
        - Streamlit  
        - Gen AI (LLMs, RAG)  
        - Matplotlib / Plotly  
        """)

        st.markdown("### What Can Be Done")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Demand Intelligence**
            - Forecast product demand accurately  
            - Reduce overstocking and understocking  

            **Predictive Logistics**
            - Forecast shipping delays  
            - Optimize transport schedules  
            """)

        with col2:
            st.markdown("""
            **Supply Chain Optimization**
            - Inventory allocation  
            - Warehouse distribution  
            - Route optimization  
            - Predict maintenance and downtime  

            **AI Insights Layer**
            - Actionable dashboards  
            - AI-based decision support  
            """)

        st.markdown("### ❓ Why It Is Needed")

        st.markdown("""
        **Challenges**
        - Inventory wastage and stockouts  
        - Logistics delays  
        - High operational costs  

        **Solutions by OmniFlow**
        - AI-driven forecasting  
        - Optimized routes and inventory  
        - Real-time proactive decision-making  
        """)

        st.markdown("### 👥 Who Can Use It")

        st.markdown("""
        **Enterprise Roles**
        - Supply Chain Managers  
        - Production Planners  
        - Logistics Coordinators  
        - Data Scientists  
        - Data Analysts  
        - Business Analysts  

        **Industry Sectors**
        - Retail  
        - Manufacturing  
        - E-commerce  
        - FMCG  
        - Pharmaceuticals  
        """)

        st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================================================
    # TAB 2 : APPLICATION (UNCHANGED LOGIC, STYLED OUTPUT)
    # ==================================================================================
    with tab2:

        st.header("Demand Forecasting – Intelligence Module")

        # -------------------------------
        # LOAD DATA
        # -------------------------------
        df = load_data()

        with st.expander("📘 Data Dictionary"):
            st.dataframe(DATA_DICTIONARY, use_container_width=True)

        with st.expander("📊 Data Profiling"):
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
        # ENCODING
        # -------------------------------
        for col in ["product_category", "sales_region", "weather_condition", "season"]:
            df[col] = LabelEncoder().fit_transform(df[col])

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

        # -------------------------------
        # KPIs (STYLED)
        # -------------------------------
        st.subheader("Executive KPIs")
        c1, c2, c3, c4 = st.columns(4)

        kpis = [
            ("Best Model", best_model),
            ("Avg Forecast", int(df_forecast["forecast"].mean())),
            ("RMSE", round(results_df.iloc[0]["RMSE"], 2)),
            ("Volatility", round(sigma, 2))
        ]

        for col, (title, value) in zip([c1, c2, c3, c4], kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        # -------------------------------
        # VISUALS
        # -------------------------------
        st.plotly_chart(
            px.bar(results_df, x="Model", y="RMSE", text="RMSE", title="Model Comparison"),
            use_container_width=True
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["forecast"], name="Forecast"))
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["upper_ci"], name="Upper CI", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["lower_ci"], name="Lower CI", fill="tonexty"))
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # NLP
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


