import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True


def add_features(df):
    """Add time-based features to improve predictions"""
    df = df.copy()

    # Extract time components
    df["hour"] = df["ds"].dt.hour
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day

    # Create cyclical features for hour of day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Create cyclical features for day of week
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Create cyclical features for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def train_prophet_optimized(train_df, future_df, pollutant, station_code):
    """Optimized Prophet model with dynamic parameters"""

    # Pollutant-specific parameter tuning based on characteristics
    params = {
        "SO2": {
            "changepoint": 0.05,
            "seasonality": 10,
            "daily_fourier": 12,
            "weekly_fourier": 10,
        },
        "NO2": {
            "changepoint": 0.08,
            "seasonality": 15,
            "daily_fourier": 20,
            "weekly_fourier": 12,
        },
        "O3": {
            "changepoint": 0.1,
            "seasonality": 20,
            "daily_fourier": 24,
            "weekly_fourier": 12,
        },
        "CO": {
            "changepoint": 0.05,
            "seasonality": 12,
            "daily_fourier": 15,
            "weekly_fourier": 10,
        },
        "PM10": {
            "changepoint": 0.08,
            "seasonality": 15,
            "daily_fourier": 20,
            "weekly_fourier": 12,
        },
        "PM2.5": {
            "changepoint": 0.1,
            "seasonality": 15,
            "daily_fourier": 20,
            "weekly_fourier": 10,
        },
    }

    # Station-specific adjustments
    station_adjustments = {
        206: {"changepoint_mod": 1.2, "seasonality_mod": 1.1},
        211: {"changepoint_mod": 0.9, "seasonality_mod": 1.2},
        217: {
            "changepoint_mod": 1.1,
            "seasonality_mod": 1.3,
        },  # O3 needs stronger seasonality
        219: {"changepoint_mod": 0.8, "seasonality_mod": 1.0},
        225: {"changepoint_mod": 1.0, "seasonality_mod": 1.2},
        228: {"changepoint_mod": 1.1, "seasonality_mod": 1.2},
    }

    # Apply station-specific adjustments
    adj = station_adjustments.get(
        station_code, {"changepoint_mod": 1.0, "seasonality_mod": 1.0}
    )
    changepoint_scale = params[pollutant]["changepoint"] * adj["changepoint_mod"]
    seasonality_scale = params[pollutant]["seasonality"] * adj["seasonality_mod"]

    # Create model with optimized parameters
    model = Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=changepoint_scale,
        seasonality_prior_scale=seasonality_scale,
        holidays_prior_scale=10.0,
        mcmc_samples=0,  # Faster fitting
    )

    # Add custom seasonality components
    model.add_seasonality(
        name="daily", period=1, fourier_order=params[pollutant]["daily_fourier"]
    )
    model.add_seasonality(
        name="weekly", period=7, fourier_order=params[pollutant]["weekly_fourier"]
    )

    # Add yearly seasonality if we have enough data (at least 6 months)
    if len(train_df) >= 365 * 12:
        model.add_seasonality(name="yearly", period=365.25, fourier_order=8)

    # Fit model
    model.fit(train_df)

    # Generate forecast
    forecast = model.predict(future_df)

    # Apply adaptive smoothing based on pollutant volatility
    volatility_factors = {"SO2": 3, "NO2": 5, "O3": 7, "CO": 3, "PM10": 5, "PM2.5": 5}
    window = volatility_factors.get(pollutant, 5)

    # Apply smoothing with weighted moving average
    forecast["yhat"] = (
        forecast["yhat"].rolling(window=window, center=True, min_periods=1).mean()
    )

    # Ensure non-negative predictions
    forecast["yhat"] = np.maximum(0, forecast["yhat"])

    return forecast


def train_rf_model(train_data, future_data, pollutant):
    """Train a Random Forest model as part of ensemble"""

    # Create features for training
    train_features = add_features(train_data)
    future_features = add_features(future_data)

    # Base feature columns that are available for both training and future data
    base_feature_cols = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
    ]
    feature_cols = base_feature_cols.copy()

    # Define lag_cols variable even if not used (avoids potential reference issues)
    lag_cols = []

    # Add lag features to training data only if we have enough data
    if len(train_features) > 24 * 7:
        # Add 24-hour lag
        train_features["lag_24"] = train_features["y"].shift(24)
        # Add 7-day lag
        train_features["lag_168"] = train_features["y"].shift(168)
        # Add 7-day moving average
        train_features["ma_168"] = (
            train_features["y"].rolling(window=168, min_periods=1).mean()
        )

        # Set lag columns and update feature list
        lag_cols = ["lag_24", "lag_168", "ma_168"]
        feature_cols.extend(lag_cols)

    # Clean training data
    train_features = train_features.dropna()

    if len(train_features) < 10:  # Not enough data for RF
        return None

    # Define hyperparameters based on pollutant
    n_estimators = {
        "SO2": 100,
        "NO2": 150,
        "O3": 200,
        "CO": 100,
        "PM10": 150,
        "PM2.5": 200,
    }.get(pollutant, 100)

    max_depth = {"SO2": 10, "NO2": 15, "O3": 20, "CO": 10, "PM10": 15, "PM2.5": 15}.get(
        pollutant, 10
    )

    # Train Random Forest model
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )

    X_train = train_features[feature_cols]
    y_train = train_features["y"]

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    rf.fit(X_train_scaled, y_train)

    # For future data, we only use the base features that don't depend on past values
    X_future = future_features[base_feature_cols].copy()

    # Add placeholder columns for lag features with zeros if required
    if len(feature_cols) > len(base_feature_cols):
        for lag_col in lag_cols:
            X_future[lag_col] = 0

    # Ensure columns are in the same order as training data
    X_future = X_future[feature_cols]

    # Transform using the same scaler
    X_future_scaled = scaler.transform(X_future)

    # Predict
    predictions = rf.predict(X_future_scaled)

    # Ensure non-negative predictions
    predictions = np.maximum(0, predictions)

    return predictions


def ensemble_predictions(prophet_forecast, rf_predictions, pollutant):
    """Combine predictions from multiple models"""

    # Define weights for each model based on pollutant characteristics
    weights = {
        "SO2": {"prophet": 0.7, "rf": 0.3},
        "NO2": {"prophet": 0.6, "rf": 0.4},
        "O3": {"prophet": 0.7, "rf": 0.3},
        "CO": {"prophet": 0.8, "rf": 0.2},
        "PM10": {"prophet": 0.6, "rf": 0.4},
        "PM2.5": {"prophet": 0.65, "rf": 0.35},
    }

    # Get weights for the current pollutant
    w = weights.get(pollutant, {"prophet": 0.7, "rf": 0.3})

    # Combine predictions
    if rf_predictions is not None:
        combined = (w["prophet"] * prophet_forecast["yhat"].values) + (
            w["rf"] * rf_predictions
        )
    else:
        combined = prophet_forecast["yhat"].values

    return combined


def preprocess_training_data(raw_data, pollutant):
    """Advanced preprocessing for training data"""

    # Handle outliers with adaptive thresholds based on pollutant
    outlier_factors = {
        "SO2": 1.8,
        "NO2": 2.0,
        "O3": 2.5,
        "CO": 1.8,
        "PM10": 2.0,
        "PM2.5": 2.0,
    }
    factor = outlier_factors.get(pollutant, 2.0)

    if len(raw_data) > 10:
        q1 = raw_data["y"].quantile(0.1)
        q3 = raw_data["y"].quantile(0.9)
        iqr = q3 - q1

        raw_data["y"] = np.clip(raw_data["y"], q1 - factor * iqr, q3 + factor * iqr)

    # Create hourly time series with proper resampling
    train_df = raw_data.set_index("ds").resample("H").mean()

    # Handle missing values with advanced interpolation
    train_df["y"] = train_df["y"].interpolate(method="linear", limit=24)
    train_df["y"] = train_df["y"].interpolate(method="time", limit_direction="both")
    train_df["y"] = train_df["y"].ffill().bfill()
    train_df["y"] = train_df["y"].fillna(0)

    return train_df.reset_index()


def load_and_preprocess_data():
    """Load measurement and instrument data, then preprocess measurement data"""
    measurement = pd.read_csv("data/raw/measurement_data.csv")
    instrument_data = pd.read_csv("data/raw/instrument_data.csv")

    measurement["Measurement date"] = pd.to_datetime(measurement["Measurement date"])
    instrument_data["Measurement date"] = pd.to_datetime(
        instrument_data["Measurement date"]
    )

    # Filter for normal measurements using instrument data
    normal_instruments = instrument_data[instrument_data["Instrument status"] == 0]

    # You can further filter measurement data based on normal_instruments if needed
    return measurement


def main():
    logging.basicConfig(level=logging.INFO)

    # Load data
    measurement = load_and_preprocess_data()

    # Define targets
    targets = [
        {
            "station": 206,
            "pollutant": "SO2",
            "start": "2023-07-01",
            "end": "2023-07-31",
        },
        {
            "station": 211,
            "pollutant": "NO2",
            "start": "2023-08-01",
            "end": "2023-08-31",
        },
        {"station": 217, "pollutant": "O3", "start": "2023-09-01", "end": "2023-09-30"},
        {"station": 219, "pollutant": "CO", "start": "2023-10-01", "end": "2023-10-31"},
        {
            "station": 225,
            "pollutant": "PM10",
            "start": "2023-11-01",
            "end": "2023-11-30",
        },
        {
            "station": 228,
            "pollutant": "PM2.5",
            "start": "2023-12-01",
            "end": "2023-12-31",
        },
    ]

    predictions = {}

    for target in targets:
        station_code = target["station"]
        pollutant = target["pollutant"]
        start_date = pd.to_datetime(target["start"])
        end_date = pd.to_datetime(target["end"]).replace(hour=23)

        logging.info(f"Processing station {station_code}, pollutant {pollutant}")

        # Map pollutant to column name in measurement data
        station_data = measurement[measurement["Station code"] == station_code][
            ["Measurement date", pollutant]
        ].copy()
        station_data.columns = ["ds", "y"]

        # Only use training data (before prediction period)
        training_data = station_data[station_data["ds"] < start_date].copy()

        # Apply preprocessing
        processed_data = preprocess_training_data(training_data, pollutant)

        # Create future dataframe for predictions
        future_dates = pd.date_range(start=start_date, end=end_date, freq="H")
        future_df = pd.DataFrame({"ds": future_dates})

        # Generate Prophet forecast
        prophet_forecast = train_prophet_optimized(
            processed_data, future_df, pollutant, station_code
        )

        # Generate Random Forest forecast if we have enough data
        rf_predictions = train_rf_model(processed_data, future_df, pollutant)

        # Combine predictions
        final_predictions = ensemble_predictions(
            prophet_forecast, rf_predictions, pollutant
        )

        # Store predictions in the required format
        predictions[str(station_code)] = {
            dt.strftime("%Y-%m-%d %H:%M:%S"): round(float(val), 5)
            for dt, val in zip(future_df["ds"], final_predictions)
        }

    # Ensure output directory exists
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions to JSON
    output_path = os.path.join(output_dir, "predictions_task_2.json")
    with open(output_path, "w") as f:
        json.dump({"target": predictions}, f, indent=4)

    logging.info(f"Predictions successfully saved to {output_path}")


if __name__ == "__main__":
    main()
