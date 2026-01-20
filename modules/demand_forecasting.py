def demand_forecasting_page():

    st.title("Demand Forecasting")

    tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

    # ==================================================================================
    # TAB 1 : OVERVIEW
    # ==================================================================================
    with tab1:
        st.subheader("OMNIFLOW D2D : Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System")

        st.markdown("""
        **ABSTRACT**

        OmniFlow D2D is an AI-driven end-to-end enterprise platform that integrates marketing demand forecasting,
        supply chain planning, manufacturing optimization, transportation logistics, and Gen-AI decision intelligence
        into a single system.

        It eliminates operational silos by ensuring demand signals directly drive procurement, production schedules,
        and delivery planning. The platform works as a closed-loop intelligence engine that predicts demand,
        optimizes execution, and continuously improves decisions using real-time data and AI.
        """)

        st.markdown("### 🛠 Tools & Technologies")
        st.markdown("""
        - **Python, SQL, Pandas, NumPy** – Data handling and processing  
        - **Scikit-Learn** – Machine learning models  
        - **Time-Series Models (Prophet / SARIMA)** – Demand forecasting  
        - **Optimization Models (Linear Programming)** – Supply chain optimization  
        - **Power BI / Tableau** – Business dashboards  
        - **Streamlit** – Interactive web applications  
        - **Gen AI (LLMs, RAG)** – Predictive insights and recommendations  
        - **Matplotlib / Plotly** – Advanced visualizations  
        """)

        st.markdown("### 🚀 What Can Be Done")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Demand Intelligence**
            - Accurate demand forecasting  
            - Reduced overstock and stockouts  

            **Predictive Logistics**
            - Shipping delay prediction  
            - Transport schedule optimization  
            """)

        with col2:
            st.markdown("""
            **Supply Chain Optimization**
            - Inventory & warehouse optimization  
            - Route optimization  
            - Production efficiency analysis  
            - Predictive maintenance  

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

        **OmniFlow Solutions**
        - AI-driven forecasting  
        - Optimized inventory and routing  
        - Real-time, proactive decision-making  
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

        **Industries**
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
            st.markdown("### Independent Variables")
            st.markdown("""
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
            st.markdown("### Dependent Variable")
            st.markdown("""
            - **daily_sales**  
            """)

    # ==================================================================================
    # TAB 3 : APPLICATION (YOUR EXISTING LOGIC – UNTOUCHED)
    # ==================================================================================
    with tab3:

        st.header("📈 Demand Forecasting – Intelligence Module")

        df = load_data()

        with st.expander("📘 Data Dictionary"):
            st.dataframe(DATA_DICTIONARY, use_container_width=True)

        with st.expander("🔍 Data Profiling"):
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
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        df = feature_engineering(df)

        FEATURES = [
            "unit_price", "discount_rate", "promotion_flag",
            "competitor_price", "product_category", "sales_region",
            "weather_condition", "season", "lag_1", "lag_7",
            "rolling_7", "month", "day_of_week"
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

        df_forecast = df.iloc[split:].copy()
        df_forecast["forecast"] = forecasts[best_model]

        sigma = df_forecast["forecast"].std()
        df_forecast["lower_ci"] = df_forecast["forecast"] - 1.96 * sigma
        df_forecast["upper_ci"] = df_forecast["forecast"] + 1.96 * sigma

        st.success("✅ Demand Forecasting Completed Successfully")
