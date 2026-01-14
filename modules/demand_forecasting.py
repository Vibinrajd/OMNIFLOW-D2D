# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module (STREAMLIT PAGE MODULE)
# MSc Data Science – MAJOR PROJECT
# ======================================================================================

# ------------------------------
# IMPORTS
# ------------------------------
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from openai import OpenAI, RateLimitError

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# ------------------------------
# CONFIG
# ------------------------------
DATA_PATH = "data/sales.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# GEN-AI HELPER FUNCTION (OPENAI GPT-3.5-TURBO)
# ======================================================================================
def genai_response(user_query: str, context: str) -> str:
    """
    Context-aware GenAI assistant using OpenAI GPT-3.5-turbo
    """

    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return "⚠️ OpenAI API key not configured in Streamlit secrets."

    try:
        client = OpenAI(api_key=api_key)

        system_prompt = f"""
        You are a senior supply chain analytics expert.

        Use ONLY the context below.
        Do NOT hallucinate.
        Explain ML results and business risks clearly.
        Keep answers concise and professional.

        CONTEXT:
        {context}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            max_tokens=250
        )

        return response.choices[0].message.content

    except RateLimitError:
        return (
            "⚠️ OpenAI rate limit reached.
