import json
import logging
import os

import pandas as pd

logging.getLogger("prophet.plot").disabled = True
from prophet import Prophet
from sklearn.metrics import r2_score

# Disable a warning from Prophet that was bothering me

# Load dataset
measurement = pd.read_csv("data/raw/measurement_data.csv")
measurement["Measurement date"] = pd.to_datetime(measurement["Measurement date"])

# Define station and pollutant combinations
targets = [
    {"station": 206, "pollutant": "SO2", "start": "2023-07-01", "end": "2023-07-31"},
    {"station": 211, "pollutant": "NO2", "start": "2023-08-01", "end": "2023-08-31"},
    {"station": 217, "pollutant": "O3", "start": "2023-09-01", "end": "2023-09-30"},
    {"station": 219, "pollutant": "CO", "start": "2023-10-01", "end": "2023-10-31"},
    {"station": 225, "pollutant": "PM10", "start": "2023-11-01", "end": "2023-11-30"},
    {"station": 228, "pollutant": "PM2.5", "start": "2023-12-01", "end": "2023-12-31"},
]

predictions = {}
r2_scores = {}

for target in targets:
    station_code = target["station"]
    pollutant = target["pollutant"]
    start_date = target["start"]
    end_date = pd.to_datetime(target["end"]).replace(hour=23, minute=0, second=0)
    end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")

    # Filter and clean data
    df = measurement[measurement["Station code"] == station_code][
        ["Measurement date", pollutant]
    ].copy()
    df.columns = ["ds", "y"]
    df = df.dropna()
    df = df[df["y"] >= 0]

    if df.empty:
        print(
            f"Warning: No valid data for station {station_code}, pollutant {pollutant}."
        )
        continue

    # interpolate missing values
    df = df.set_index("ds").resample("h").asfreq()
    df["y"] = df["y"].interpolate(method="time")
    df = df.reset_index()

    # Train model with optimized parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        holidays_prior_scale=10.0,
    )
    model.fit(df)

    # Predict future values
    future = model.make_future_dataframe(
        periods=24 * 31, freq="h", include_history=False
    )
    future = future[(future["ds"] >= start_date) & (future["ds"] <= end_date)]
    forecast = model.predict(future)

    # Store predictions
    station_preds = {
        row["ds"].strftime("%Y-%m-%d %H:%M:%S"): round(row["yhat"], 5)
        for _, row in forecast.iterrows()
    }
    predictions[str(station_code)] = station_preds

    # R2 score if actual data is available
    actual_data = df[(df["ds"] >= start_date) & (df["ds"] <= end_date)]
    if not actual_data.empty:
        y_true = actual_data["y"].values
        y_pred = forecast["yhat"][: len(y_true)].values
        r2_scores[str(station_code)] = round(r2_score(y_true, y_pred), 5)

# Save to json
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "predictions_task_2.json"), "w") as f:
    json.dump({"target": predictions, "r2_scores": r2_scores}, f, indent=4)
