# ============================================================
# OMNIFLOW-D2D : DEMAND FORECASTING INTELLIGENCE MODULE
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ------------------------------------------------------------
# 1. LOAD SALES DATA
# ------------------------------------------------------------
df = pd.read_csv("sales.csv", parse_dates=["date"])

# Sort for time series
df = df.sort_values(["store_id", "product_id", "date"])

# ------------------------------------------------------------
# 2. DATA DICTIONARY (AUTO-GENERATED)
# ------------------------------------------------------------
data_dictionary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing_%": df.isnull().mean() * 100
})

# ------------------------------------------------------------
# 3. DATA QUALITY CHECKS
# ------------------------------------------------------------
assert df["daily_sales"].isnull().sum() == 0, "Target has missing values"
assert (df["daily_sales"] >= 0).all(), "Negative sales detected"

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

# Calendar features
df["weekday"] = df["date"].dt.weekday
df["month"] = df["date"].dt.month

# Lag & rolling features (PER STORE + PRODUCT)
for lag in [1, 7, 14]:
    df[f"lag_{lag}"] = (
        df.groupby(["store_id", "product_id"])["daily_sales"]
        .shift(lag)
    )

df["rolling_7"] = (
    df.groupby(["store_id", "product_id"])["daily_sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["rolling_14"] = (
    df.groupby(["store_id", "product_id"])["daily_sales"]
    .shift(1)
    .rolling(14)
    .mean()
)

df = df.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# 5. FEATURE SETUP
# ------------------------------------------------------------
TARGET = "daily_sales"

NUMERIC_FEATURES = [
    "unit_price",
    "discount_rate",
    "promotion_flag",
    "weekday",
    "month",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7",
    "rolling_14"
]

CATEGORICAL_FEATURES = [
    "store_id",
    "product_id",
    "weather_condition",
    "season",
    "sales_region"
]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ------------------------------------------------------------
# 6. PREPROCESSOR
# ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES)
    ]
)

# ------------------------------------------------------------
# 7. MODELS
# ------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}

# ------------------------------------------------------------
# 8. TIME SERIES CROSS VALIDATION
# ------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
results = []

for model_name, model in models.items():
    mae_scores = []
    rmse_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model)
            ]
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, preds))
        rmse_scores.append(mean_squared_error(y_test, preds, squared=False))

    results.append({
        "model": model_name,
        "MAE": np.mean(mae_scores),
        "RMSE": np.mean(rmse_scores)
    })

results_df = pd.DataFrame(results).sort_values("RMSE")

# ------------------------------------------------------------
# 9. TRAIN BEST MODEL ON FULL DATA
# ------------------------------------------------------------
best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

final_pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", best_model)
    ]
)

final_pipeline.fit(X, y)

# ------------------------------------------------------------
# 10. CONFIDENCE INTERVALS (BOOTSTRAP)
# ------------------------------------------------------------
def bootstrap_ci(model, X, n_boot=100, alpha=0.05):
    preds = []
    for _ in range(n_boot):
        idx = np.random.choice(len(X), len(X), replace=True)
        preds.append(model.predict(X.iloc[idx]))
    preds = np.array(preds)
    return (
        np.percentile(preds, 100 * alpha / 2, axis=0),
        np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    )

lower_ci, upper_ci = bootstrap_ci(final_pipeline, X)

# ------------------------------------------------------------
# 11. FEATURE IMPORTANCE (EXPLAINABILITY)
# ------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    feature_names = (
        final_pipeline.named_steps["prep"]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

# ------------------------------------------------------------
# 12. FORECAST OUTPUT
# ------------------------------------------------------------
df_forecast = df.copy()
df_forecast["forecast"] = final_pipeline.predict(X)
df_forecast["lower_ci"] = lower_ci
df_forecast["upper_ci"] = upper_ci

# ------------------------------------------------------------
# 13. EXECUTIVE KPIs
# ------------------------------------------------------------
kpis = {
    "Total_Stores": df["store_id"].nunique(),
    "Total_Products": df["product_id"].nunique(),
    "Avg_Daily_Sales": df["daily_sales"].mean(),
    "Best_Model": best_model_name,
    "RMSE": results_df.iloc[0]["RMSE"]
}

# ------------------------------------------------------------
# 14. NLP-STYLE ANALYTICS (DATA ONLY)
# ------------------------------------------------------------
def ask_question(question: str):
    q = question.lower()

    if "best store" in q:
        return df.groupby("store_id")["daily_sales"].mean().idxmax()

    if "best product" in q:
        return df.groupby("product_id")["daily_sales"].mean().idxmax()

    if "highest demand" in q:
        return df.sort_values("daily_sales", ascending=False).head(5)

    return "Question not supported."

# ------------------------------------------------------------
# 15. DOWNLOADABLE OUTPUT
# ------------------------------------------------------------
df_forecast.to_csv("demand_forecast_output.csv", index=False)
results_df.to_csv("model_comparison.csv", index=False)

print("✅ Demand Forecasting Module Completed Successfully")
print(results_df)
print("KPIs:", kpis)
