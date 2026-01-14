# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Module
# ======================================================================================
# Project Type : MAJOR MSc Data Science Project
# Module       : Demand Forecasting (Step 1 of D2D Pipeline)
# Purpose      :
#   - Analyze historical sales demand
#   - Detect demand patterns & seasonality
#   - Forecast future demand using ML
#   - Provide explainable & reliable demand signals
#
# Business Context:
#   Demand forecasting is the foundation of supply chain planning.
#   Any error here propagates into production, inventory & logistics.
#
# ======================================================================================

# ----------------------------------
# STANDARD LIBRARIES
# ----------------------------------
import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta

# ----------------------------------
# THIRD PARTY LIBRARIES
# ----------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ======================================================================================
# 1. CONFIGURATION MANAGEMENT
# ======================================================================================

class DemandForecastConfig:
    """
    Centralized configuration class.
    This allows easy tuning and experimentation.
    """

    # Paths
    DATA_PATH = "data/sales.csv"
    OUTPUT_PATH = "data/forecast_demand.csv"
    LOG_PATH = "logs/demand_forecasting.log"

    # Forecast settings
    TEST_SIZE_RATIO = 0.2
    FORECAST_HORIZON_DAYS = 30

    # Lag features
    LAG_DAYS = [1, 7, 14, 30]

    # Rolling windows
    ROLLING_WINDOWS = [7, 14, 30]

    # Random Forest Params
    RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 18,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1
    }

    # Gradient Boosting Params
    GB_PARAMS = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": 42
    }


# ======================================================================================
# 2. LOGGING SETUP (ENTERPRISE PRACTICE)
# ======================================================================================

def setup_logger():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename=DemandForecastConfig.LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logging.info("Demand Forecasting Logger Initialized")


# ======================================================================================
# 3. DATA LOADING & VALIDATION
# ======================================================================================

def load_sales_data(path):
    """
    Loads sales dataset and performs basic validation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sales file not found at {'/content/sales.csv'}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    logging.info(f"Sales data loaded | Rows: {len(df)}")
    return df


def validate_schema(df):
    """
    Ensures required columns are present.
    """
    required_columns = {
        "date",
        "product_id",
        "daily_sales",
        "price",
        "promotion"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    logging.info("Schema validation successful")
    return True


# ======================================================================================
# 4. DATA PROFILING & QUALITY CHECKS
# ======================================================================================

def data_quality_report(df):
    """
    Generates a detailed data quality report.
    """
    report = {}

    report["total_rows"] = len(df)
    report["date_range"] = {
        "start": df["date"].min(),
        "end": df["date"].max()
    }

    report["missing_values"] = df.isnull().sum().to_dict()
    report["negative_sales_count"] = int((df["daily_sales"] < 0).sum())
    report["zero_sales_ratio"] = float((df["daily_sales"] == 0).mean())

    logging.info(f"Data Quality Report: {json.dumps(report, default=str)}")
    return report


# ======================================================================================
# 5. BUSINESS ASSUMPTION LAYER
# ======================================================================================

def apply_business_rules(df):
    """
    Apply domain-specific assumptions.
    """

    # Rule 1: Sales cannot be negative
    df = df[df["daily_sales"] >= 0]

    # Rule 2: Price should be positive
    df = df[df["price"] > 0]

    # Rule 3: Promotion must be binary
    df["promotion"] = df["promotion"].astype(int)

    logging.info("Business rules applied")
    return df


# ======================================================================================
# 6. TIME FEATURE ENGINEERING
# ======================================================================================

def add_time_features(df):
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df

    # ======================================================================================
# 7. ADVANCED FEATURE ENGINEERING (LAG + ROLLING + DEMAND SIGNALS)
# ======================================================================================

def add_lag_features(df, lag_days):
    """
    Adds lag-based demand signals per product.
    """
    df = df.sort_values(["product_id", "date"])

    for lag in lag_days:
        df[f"lag_sales_{lag}"] = (
            df.groupby("product_id")["daily_sales"]
            .shift(lag)
        )

    logging.info(f"Lag features added: {lag_days}")
    return df


def add_rolling_features(df, windows):
    """
    Adds rolling statistics capturing short & medium term trends.
    """
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby("product_id")["daily_sales"]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"rolling_std_{window}"] = (
            df.groupby("product_id")["daily_sales"]
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )

    logging.info(f"Rolling features added: {windows}")
    return df


def add_price_effect_features(df):
    """
    Captures price elasticity & promotion impact.
    """
    df["price_change_pct"] = (
        df.groupby("product_id")["price"]
        .pct_change()
        .fillna(0)
    )

    df["promo_price_interaction"] = df["promotion"] * df["price"]

    return df


# ======================================================================================
# 8. MASTER FEATURE PIPELINE
# ======================================================================================

def build_feature_matrix(df, config):
    """
    Complete feature engineering pipeline.
    """
    df = add_time_features(df)
    df = add_lag_features(df, config.LAG_DAYS)
    df = add_rolling_features(df, config.ROLLING_WINDOWS)
    df = add_price_effect_features(df)

    df.dropna(inplace=True)

    feature_columns = [
        "price",
        "promotion",
        "price_change_pct",
        "promo_price_interaction",
        "day",
        "month",
        "day_of_week",
        "week_of_year",
        "is_weekend"
    ]

    # Dynamically add lag & rolling features
    for lag in config.LAG_DAYS:
        feature_columns.append(f"lag_sales_{lag}")

    for window in config.ROLLING_WINDOWS:
        feature_columns.append(f"rolling_mean_{window}")
        feature_columns.append(f"rolling_std_{window}")

    X = df[feature_columns]
    y = df["daily_sales"]

    logging.info(f"Feature matrix built | Features: {len(feature_columns)}")
    return X, y, df, feature_columns


# ======================================================================================
# 9. TIME SERIES TRAIN / TEST SPLIT
# ======================================================================================

def time_series_split(X, y, test_ratio):
    split_index = int(len(X) * (1 - test_ratio))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    logging.info(
        f"Train size: {len(X_train)} | Test size: {len(X_test)}"
    )

    return X_train, X_test, y_train, y_test


# ======================================================================================
# 10. MODEL DEFINITIONS
# ======================================================================================

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, params):
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ======================================================================================
# 11. MODEL EVALUATION FRAMEWORK
# ======================================================================================

def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

    logging.info(f"Evaluation | {model_name} | {metrics}")
    return metrics, preds


def compare_models(results):
    """
    Compares models based on RMSE.
    """
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["RMSE"].idxmin()]
    return results_df, best_row["Model"]


# ======================================================================================
# 12. CROSS-VALIDATION (TIME SERIES)
# ======================================================================================

def time_series_cross_validation(X, y, model_fn, params=None, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = (
            model_fn(X_train, y_train, params)
            if params else model_fn(X_train, y_train)
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    avg_rmse = np.mean(rmse_scores)
    logging.info(f"CV RMSE: {avg_rmse}")
    return avg_rmse


# ======================================================================================
# 13. TRAIN & SELECT BEST MODEL
# ======================================================================================

def train_and_select_model(X_train, X_test, y_train, y_test, config):
    results = []
    predictions = {}

    # Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics, lr_preds = evaluate_model(
        lr_model, X_test, y_test, "Linear Regression"
    )
    results.append(lr_metrics)
    predictions["Linear Regression"] = lr_preds

    # Random Forest
    rf_model = train_random_forest(
        X_train, y_train, config.RF_PARAMS
    )
    rf_metrics, rf_preds = evaluate_model(
        rf_model, X_test, y_test, "Random Forest"
    )
    results.append(rf_metrics)
    predictions["Random Forest"] = rf_preds

    # Gradient Boosting
    gb_model = train_gradient_boosting(
        X_train, y_train, config.GB_PARAMS
    )
    gb_metrics, gb_preds = evaluate_model(
        gb_model, X_test, y_test, "Gradient Boosting"
    )
    results.append(gb_metrics)
    predictions["Gradient Boosting"] = gb_preds

    results_df, best_model_name = compare_models(results)

    logging.info(f"Best model selected: {best_model_name}")
    return best_model_name, results_df, predictions
# ======================================================================================
# 14. MODEL EXPLAINABILITY (FEATURE IMPORTANCE)
# ======================================================================================

def extract_feature_importance(model, feature_names):
    """
    Extract feature importance for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        logging.info("Feature importance extracted")
        return importance_df

    logging.warning("Model does not support feature importance")
    return pd.DataFrame()


# ======================================================================================
# 15. DEMAND SCENARIO SIMULATION
# ======================================================================================

def simulate_demand_scenarios(base_forecast):
    """
    Generates demand scenarios for risk-aware planning.
    """
    scenarios = {}

    scenarios["expected"] = base_forecast
    scenarios["optimistic"] = base_forecast * 1.15   # +15%
    scenarios["pessimistic"] = base_forecast * 0.85  # -15%

    scenario_df = pd.DataFrame(scenarios)
    logging.info("Demand scenarios generated")

    return scenario_df


# ======================================================================================
# 16. CONFIDENCE INTERVAL ESTIMATION
# ======================================================================================

def calculate_confidence_intervals(predictions, confidence=0.95):
    """
    Approximate confidence bands using residual distribution.
    """
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)

    z_score = 1.96 if confidence == 0.95 else 1.65

    lower = predictions - z_score * std_pred
    upper = predictions + z_score * std_pred

    return lower, upper


# ======================================================================================
# 17. FORECAST HORIZON GENERATION
# ======================================================================================

def generate_future_dates(last_date, horizon):
    return [
        last_date + timedelta(days=i)
        for i in range(1, horizon + 1)
    ]


# ======================================================================================
# 18. FINAL FORECAST ASSEMBLY
# ======================================================================================

def assemble_forecast_output(
    df,
    X_test,
    predictions,
    model_name,
    feature_names
):
    """
    Build final forecast output for downstream modules.
    """
    forecast_df = df.loc[X_test.index].copy()
    forecast_df["forecast_demand"] = predictions
    forecast_df["model_used"] = model_name

    # Confidence bands
    lower, upper = calculate_confidence_intervals(predictions)
    forecast_df["lower_bound"] = lower
    forecast_df["upper_bound"] = upper

    final_output = forecast_df[
        [
            "date",
            "product_id",
            "forecast_demand",
            "lower_bound",
            "upper_bound",
            "model_used"
        ]
    ]

    logging.info("Final forecast assembled")
    return final_output


# ======================================================================================
# 19. MASTER PIPELINE ORCHESTRATOR
# ======================================================================================

def run_demand_forecasting():
    """
    Master function executed by application.py
    """
    setup_logger()
    logging.info("Demand Forecasting Pipeline Started")

    config = DemandForecastConfig()

    # Step 1: Load & Validate
    df = load_sales_data(config.DATA_PATH)
    validate_schema(df)

    # Step 2: Profiling & Business Rules
    quality_report = data_quality_report(df)
    df = apply_business_rules(df)

    # Step 3: Feature Engineering
    X, y, processed_df, feature_names = build_feature_matrix(df, config)

    # Step 4: Train/Test Split
    X_train, X_test, y_train, y_test = time_series_split(
        X, y, config.TEST_SIZE_RATIO
    )

    # Step 5: Model Training & Selection
    best_model_name, results_df, preds_dict = train_and_select_model(
        X_train, X_test, y_train, y_test, config
    )

    best_predictions = preds_dict[best_model_name]

    # Step 6: Train final model on full training set
    if best_model_name == "Random Forest":
        final_model = train_random_forest(
            X_train, y_train, config.RF_PARAMS
        )
    elif best_model_name == "Gradient Boosting":
        final_model = train_gradient_boosting(
            X_train, y_train, config.GB_PARAMS
        )
    else:
        final_model = train_linear_regression(X_train, y_train)

    # Step 7: Explainability
    feature_importance = extract_feature_importance(
        final_model, feature_names
    )

    # Step 8: Final Forecast
    final_forecast = assemble_forecast_output(
        processed_df,
        X_test,
        best_predictions,
        best_model_name,
        feature_names
    )

    # Step 9: Save Outputs
    os.makedirs("data", exist_ok=True)
    final_forecast.to_csv(config.OUTPUT_PATH, index=False)

    if not feature_importance.empty:
        feature_importance.to_csv(
            "data/demand_feature_importance.csv",
            index=False
        )

    logging.info("Demand Forecasting Pipeline Completed")

    return {
        "forecast": final_forecast,
        "model_comparison": results_df,
        "feature_importance": feature_importance,
        "data_quality": quality_report
    }


# ======================================================================================
# 20. MODULE ENTRY POINT
# ======================================================================================

if __name__ == "__main__":
    run_demand_forecasting()

