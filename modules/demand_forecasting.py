# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module with Custom NLP Intelligence
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

# ----------------------------------
# IMPORTS
# ----------------------------------
import os
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# ----------------------------------
# CONFIG
# ----------------------------------
DATA_PATH = "data/sales.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date", "product_id", "daily_sales", "price", "promotion",
        "lag_sales_1", "lag_sales_7", "rolling_mean_7",
        "forecast_demand", "lower_bound", "upper_bound"
    ],
    "Description": [
        "Transaction date",
        "Unique product identifier",
        "Units sold per day",
        "Unit selling price",
        "Promotion flag (0/1)",
        "Previous day sales",
        "Sales 7 days ago",
        "7-day rolling average demand",
        "Predicted future demand",
        "Lower confidence interval",
        "Upper confidence interval"
    ]
})

# ======================================================================================
# DATA LOADING
# ======================================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ======================================================================================
# DATA PROFILING
# ======================================================================================
def data_profiling(df):
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Missing Values (%)": round(df.isnull().mean().mean() * 100, 2),
        "Zero Sales (%)": round((df["daily_sales"] == 0).mean() * 100, 2),
        "Average Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility": round(df["daily_sales"].std(), 2)
    }

# ======================================================================================
# FEATURE ENGINEERING
# ======================================================================================
def feature_engineering(df):
    df = df.sort_values(["product_id", "date"])
    df["lag_sales_1"] = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_sales_7"] = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_mean_7"] = (
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
            n_estimators=300, max_depth=18, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        )
    }

    results, predictions = [], {}

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
# ------------------ CUSTOM NLP ENGINE (CORE PART) ------------------
# ======================================================================================

# ---------- Text Preprocessing ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

# ---------- Knowledge Base Builder ----------
def build_knowledge_base(m):
    docs = [
        f"The best forecasting model is {m['model']} with RMSE {m['rmse']:.2f}.",
        f"Average demand is {m['avg']:.0f} units and peak demand is {m['peak']:.0f} units.",
        f"Demand volatility is {m['vol']:.2f} percent which indicates "
        f"{'high demand risk' if m['vol'] > 30 else 'stable demand'}.",
        f"The confidence interval width is {m['ci']:.2f} units indicating uncertainty.",
        "High demand risk requires safety stock and frequent replenishment.",
        "Low volatility allows lean inventory and cost optimization."
    ]
    return docs

# ---------- Train NLP Model ----------
def train_nlp(docs):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    vectors = vectorizer.fit_transform(
        [preprocess_text(d) for d in docs]
    )
    return vectorizer, vectors

# ---------- NLP Answer Engine ----------
def nlp_answer(question, docs, vectorizer, vectors):
    q_vec = vectorizer.transform([preprocess_text(question)])
    sims = cosine_similarity(q_vec, vectors)
    idx = sims.argmax()
    score = sims[0][idx]

    if score < 0.15:
        return (
            "This question is outside the analytical scope. "
            "Please ask about demand, models, risk, volatility, or confidence intervals."
        )

    return f"{docs[idx]} (confidence: {score:.2f})"

# ======================================================================================
# PDF REPORT
# ======================================================================================
def generate_pdf(metrics, insights):
    path = f"{OUTPUT_DIR}/Demand_Forecast_Report.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("OmniFlow-D2D Demand Forecast Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Analytical Insights", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(ins, styles["Normal"]))

    doc.build(story)
    return path

# ======================================================================================
# MAIN STREAMLIT PAGE
# ======================================================================================
def demand_forecasting_page():

    st.header("📈 Demand Forecasting with NLP Intelligence")

    df_raw = load_data()
    profile = data_profiling(df_raw)

    with st.expander("📘 Data Dictionary"):
        st.dataframe(DATA_DICTIONARY, width="stretch")

    with st.expander("🔍 Data Profiling"):
        for k, v in profile.items():
            st.write(f"**{k}:** {v}")

    df = feature_engineering(df_raw)

    FEATURES = ["price", "promotion", "lag_sales_1", "lag_sales_7", "rolling_mean_7"]
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

    avg = forecast_df["forecast_demand"].mean()
    peak = forecast_df["forecast_demand"].max()
    vol = (forecast_df["forecast_demand"].std() / avg) * 100
    rmse = results_df.iloc[0]["RMSE"]
    ci = (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()

    # KPIs
    st.subheader("📊 Executive KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model)
    c2.metric("Avg Demand", int(avg))
    c3.metric("RMSE", round(rmse, 2))
    c4.metric("Volatility (%)", round(vol, 2))

    # Charts
    st.subheader("📈 Forecast Analysis")
    st.plotly_chart(px.line(forecast_df, x="date", y="forecast_demand",
                            title="Forecast Trend"), width="stretch")
    st.plotly_chart(px.histogram(forecast_df, x="forecast_demand",
                                 title="Demand Distribution"), width="stretch")

    # ---------------- NLP SECTION ----------------
    st.divider()
    st.subheader("🤖 NLP-Based Analytical Assistant")

    metrics = {
        "model": best_model,
        "rmse": rmse,
        "avg": avg,
        "peak": peak,
        "vol": vol,
        "ci": ci
    }

    knowledge_docs = build_knowledge_base(metrics)
    vectorizer, vectors = train_nlp(knowledge_docs)

    if "nlp_chat" not in st.session_state:
        st.session_state.nlp_chat = []

    q = st.chat_input("Ask any analytical question about demand forecasting")

    if q:
        ans = nlp_answer(q, knowledge_docs, vectorizer, vectors)
        st.session_state.nlp_chat.append((q, ans))

    for q, a in st.session_state.nlp_chat[-10:]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
