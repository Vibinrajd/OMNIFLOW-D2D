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

def demand_forecasting_page():

    st.title("Demand Forecasting")

    tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

    # ==================================================================================
    # TAB 1 : OVERVIEW
    # ==================================================================================
    with tab1:

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

        st.markdown("### 🚀 What Can Be Done")

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

    # ==================================================================================
    # TAB 2 : IMPORTANT ATTRIBUTES
    # ==================================================================================
    with tab2:

        st.subheader("📘 Required Column Data Dictionary")
        st.dataframe(DATA_DICTIONARY, use_container_width=True)

        st.subheader("🔀 Variable Classification")

        left, right = st.columns(2)

        with left:
            st.markdown("""
            **Independent Variables**
            - unit_price  
            - discount_rate  
            - promotion_flag  
            - competitor_price  
            - product_category  
            - sales_region  
            - weather_condition  
            - season  
            - lag_1  
            - lag_7  
            - rolling_7  
            - month  
            - day_of_week  
            """)

        with right:
            st.markdown("""
            **Dependent Variable**
            - daily_sales
            """)

    # ==================================================================================
    # TAB 3 : APPLICATION (100% ORIGINAL CODE – NOTHING REMOVED)
    # ==================================================================================
    with tab3:

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

        # ==========================================================
        # SAVE FULL FORECAST FOR INVENTORY MODULE (AUTO PIPELINE)
        # ==========================================================
        FULL_FORECAST_PATH = os.path.join("data", "forecast_demand.csv")

        full_output = df_forecast[
            ["date", "store_id", "product_id", "daily_sales", "forecast", "lower_ci", "upper_ci"]
        ].copy()

        full_output.to_csv(FULL_FORECAST_PATH, index=False)

        st.success("✅ Full demand forecast saved for Inventory module")
        st.info(f"📁 File location: {FULL_FORECAST_PATH}")

        st.subheader("📊 Full Forecast Preview (used by Inventory module)")
        st.dataframe(full_output.head(10), use_container_width=True)

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
