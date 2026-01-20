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
st.set_page_config(page_title="OmniFlow D2D", layout="wide")

# ======================================================================================
# ADVANCED LIGHT UI CSS
# ======================================================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color:#f8fafc;
    color:#0f172a;
}

/* Fade in */
.block-container {
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Titles */
.page-title {
    font-size:42px;
    font-weight:900;
    color:#0f172a;
}
.page-subtitle {
    font-size:18px;
    color:#475569;
    margin-bottom:25px;
}
.section-title {
    font-size:28px;
    font-weight:800;
    margin-top:30px;
    margin-bottom:12px;
}

/* Cards */
.card {
    background:white;
    padding:22px;
    border-radius:16px;
    border:1px solid #e5e7eb;
    box-shadow:0 8px 24px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow:0 18px 40px rgba(0,0,0,0.14);
}

/* KPI Cards */
.metric-card {
    background:linear-gradient(180deg, #eef4ff, #ffffff);
    padding:18px;
    text-align:center;
    border-radius:16px;
    box-shadow:0 6px 18px rgba(30,58,138,0.18);
    transition:0.25s ease;
}
.metric-card:hover {
    transform:translateY(-6px);
    box-shadow:0 14px 32px rgba(30,58,138,0.28);
}
.metric-label {
    font-size:14px;
    color:#475569;
}
.metric-value {
    font-size:30px;
    font-weight:900;
    color:#1e3a8a;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background:#f1f5f9;
    border-radius:12px;
    padding:10px 18px;
    font-weight:600;
    color:#475569;
}
.stTabs [aria-selected="true"] {
    background:#e0e7ff;
    color:#1e3a8a;
    box-shadow:0 6px 18px rgba(30,58,138,0.25);
}

/* Buttons */
.stButton button, .stDownloadButton button {
    background:linear-gradient(135deg, #2563eb, #1e40af);
    color:white;
    border-radius:14px;
    padding:10px 24px;
    font-weight:700;
    border:none;
    box-shadow:0 10px 24px rgba(37,99,235,0.35);
    transition:0.25s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform:translateY(-3px);
    box-shadow:0 18px 36px rgba(37,99,235,0.5);
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border-radius:16px;
    box-shadow:0 10px 30px rgba(0,0,0,0.12);
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
        "Unique product identifier",
        "Product category",
        "Sales region",
        "Units sold per day (TARGET)",
        "Selling price per unit",
        "Discount percentage",
        "Promotion indicator",
        "Weather condition",
        "Competitor price",
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
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# PROFILING
# ======================================================================================
def data_profiling(df):
    return {
        "Total Records": len(df),
        "Date Range": f"{df.date.min().date()} → {df.date.max().date()}",
        "Unique Stores": df.store_id.nunique(),
        "Unique Products": df.product_id.nunique(),
        "Average Daily Sales": round(df.daily_sales.mean(), 2),
        "Sales Volatility": round(df.daily_sales.std(), 2),
        "Missing Cells": int(df.isnull().sum().sum())
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def feature_engineering(df):
    df = df.sort_values("date")
    df["lag_1"] = df.daily_sales.shift(1)
    df["lag_7"] = df.daily_sales.shift(7)
    df["rolling_7"] = df.daily_sales.rolling(7).mean()
    df["month"] = df.date.dt.month
    df["day_of_week"] = df.date.dt.dayofweek
    df.dropna(inplace=True)
    return df

# ======================================================================================
# NLP
# ======================================================================================
def nlp_answer(df, q):
    q = q.lower()
    if "highest" in q:
        return f"Highest forecasted demand: {df.forecast.max():.0f}"
    if "average" in q:
        return f"Average forecasted demand: {df.forecast.mean():.2f}"
    if "trend" in q:
        return "Demand is increasing 📈" if df.forecast.iloc[-1] > df.forecast.iloc[0] else "Demand is decreasing 📉"
    return "Try: highest demand, average demand, demand trend"

# ======================================================================================
# MAIN APP
# ======================================================================================
def demand_forecasting_page():


    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ==================================================================================
    # TAB 1 : OVERVIEW (FULL CONTENT – NOTHING SKIPPED)
    # ==================================================================================
    with tab1:

        # ---------------- ABSTRACT ----------------
        st.markdown('<div class="section-title">Abstract</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        OmniFlow D2D is an AI-driven end-to-end enterprise platform that integrates marketing demand forecasting,
        supply chain planning, manufacturing optimization, transportation logistics, and Gen-AI decision intelligence
        into a single system.<br><br>
        
        It removes operational silos by ensuring that demand signals directly drive procurement, production schedules,
        and delivery planning. The platform functions as a closed-loop intelligence engine that predicts demand,
        optimizes execution, and continuously improves decisions using real-time data and AI.
        </div>
        """, unsafe_allow_html=True)



        # ---------------- TOOLS ----------------
        st.markdown('<div class="section-title">🛠 Tools</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <ul style="line-height:1.8; padding-left:20px;">
            <li>Python, SQL, Pandas, NumPy</li>
            <li>Scikit-Learn</li>
            <li>Time-Series Models (Prophet / SARIMA)</li>
            <li>Optimization Models (Linear Programming)</li>
            <li>Power BI / Tableau</li>
            <li>Streamlit</li>
            <li>Gen AI (LLMs, RAG)</li>
            <li>Matplotlib / Plotly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


        # ---------------- WHAT CAN BE DONE ----------------
        st.markdown('<div class="section-title">What Can Be Done</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
            <b>Demand Intelligence</b>
            <ul>
                <li>Forecast product demand accurately</li>
                <li>Reduce overstocking and understocking</li>
            </ul>
        
            <b>Predictive Logistics</b>
            <ul>
                <li>Forecast shipping delays</li>
                <li>Optimize transport schedules</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
            <b>Supply Chain Optimization</b>
            <ul>
                <li>Inventory allocation</li>
                <li>Warehouse distribution</li>
                <li>Route optimization</li>
                <li>Predict maintenance and downtime</li>
            </ul>
        
            <b>AI Insights Layer</b>
            <ul>
                <li>Actionable dashboards</li>
                <li>AI-based decision support</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


        # ---------------- WHY NEEDED ----------------
        st.markdown('<div class="section-title">Why It Is Needed</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <b>Challenges</b>
        <ul>
            <li>Inventory wastage and stockouts</li>
            <li>Logistics delays</li>
            <li>High operational costs</li>
        </ul>
        
        <b>Solutions by OmniFlow</b>
        <ul>
            <li>AI-driven forecasting</li>
            <li>Optimized routes and inventory</li>
            <li>Real-time proactive decision-making</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- WHO CAN USE ----------------
        st.markdown('<div class="section-title">Who Can Use It</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <b>Enterprise Roles</b>
        <ul>
            <li>Supply Chain Managers</li>
            <li>Production Planners</li>
            <li>Logistics Coordinators</li>
            <li>Data Scientists</li>
            <li>Data Analysts</li>
            <li>Business Analysts</li>
        </ul>
        
        <b>Industry Sectors</b>
        <ul>
            <li>Retail</li>
            <li>Manufacturing</li>
            <li>E-commerce</li>
            <li>FMCG</li>
            <li>Pharmaceuticals</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


    # ==================================================================================
    # TAB 2 : APPLICATION (FULL CONTENT – NOTHING SKIPPED)
    # ==================================================================================
    with tab2:

        st.markdown('<div class="section-title">Demand Forecasting – Intelligence Module</div>', unsafe_allow_html=True)

        # ---------------- LOAD DATA ----------------
        df = load_data()

        with st.expander("Data Dictionary"):
            st.dataframe(DATA_DICTIONARY, use_container_width=True)

        with st.expander("Data Profiling"):
            profile = data_profiling(df)
            for k, v in profile.items():
                st.write(f"**{k}:** {v}")

        # ---------------- FILTERS ----------------
        store = st.selectbox("Select Store", sorted(df["store_id"].unique()))
        product = st.selectbox(
            "Select Product",
            sorted(df[df["store_id"] == store]["product_id"].unique())
        )

        df = df[(df["store_id"] == store) & (df["product_id"] == product)].copy()

        if len(df) < 30:
            st.warning("Not enough data for this Store–Product combination")
            return

        # ---------------- ENCODING ----------------
        for col in ["product_category", "sales_region", "weather_condition", "season"]:
            df[col] = LabelEncoder().fit_transform(df[col])

        # ---------------- FEATURE ENGINEERING ----------------
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

        # ---------------- TIME SPLIT ----------------
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # ---------------- MODELS ----------------
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

        # ---------------- FINAL FORECAST ----------------
        df_forecast = df.iloc[split:].copy()
        df_forecast["forecast"] = forecasts[best_model]

        sigma = df_forecast["forecast"].std()
        df_forecast["lower_ci"] = df_forecast["forecast"] - 1.96 * sigma
        df_forecast["upper_ci"] = df_forecast["forecast"] + 1.96 * sigma

        # ---------------- KPIs ----------------
        st.markdown('<div class="section-title">Executive KPIs</div>', unsafe_allow_html=True)
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
                <div class="metric-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        # ---------------- MODEL COMPARISON ----------------
        st.plotly_chart(
            px.bar(results_df, x="Model", y="RMSE", text="RMSE", title="Model Comparison"),
            use_container_width=True
        )

        # ---------------- FORECAST VISUAL ----------------
        
        st.markdown('<div class="section-title">Forecast with Confidence Interval</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["forecast"], name="Forecast"))
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["upper_ci"], name="Upper CI", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["lower_ci"], name="Lower CI", fill="tonexty"))
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- NLP ----------------
        st.markdown('<div class="section-title">💬 Demand Analytics Assistant</div>', unsafe_allow_html=True)
        q = st.text_input("Ask: highest demand, average demand, demand trend")
        if q:
            st.info(nlp_answer(df_forecast, q))

        # ---------------- DOWNLOAD ----------------
        st.download_button(
            "⬇ Download Forecast Output",
            df_forecast[
                ["date", "store_id", "product_id", "daily_sales", "forecast", "lower_ci", "upper_ci"]
            ].to_csv(index=False),
            "forecast_demand.csv"
        )












