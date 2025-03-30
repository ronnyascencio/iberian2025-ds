import json
import logging

import pandas as pd

# disable a warning from ploty in prophet was bodering me
logging.getLogger("prophet.plot").disabled = True
from prophet import Prophet

# data set
measurement = pd.read_csv("data/raw/measurement_data.csv")
measurement["Measurement date"] = pd.to_datetime(measurement["Measurement date"])

# station combinations
targets = [
    {"station": 206, "pollutant": "SO2", "start": "2023-07-01", "end": "2023-07-31"},
    {"station": 211, "pollutant": "NO2", "start": "2023-08-01", "end": "2023-08-31"},
    {"station": 217, "pollutant": "O3", "start": "2023-09-01", "end": "2023-09-30"},
    {"station": 219, "pollutant": "CO", "start": "2023-10-01", "end": "2023-10-31"},
    {"station": 225, "pollutant": "PM10", "start": "2023-11-01", "end": "2023-11-30"},
    {"station": 228, "pollutant": "PM2.5", "start": "2023-12-01", "end": "2023-12-31"},
]

predictions = {}

for target in targets:
    station_code = target["station"]
    pollutant = target["pollutant"]
    start_date = target["start"]
    end_date = pd.to_datetime(target["end"]).replace(hour=23, minute=0, second=0)
    end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")

    # data filtering
    df = measurement[(measurement["Station code"] == station_code)][
        ["Measurement date", pollutant]
    ].copy()
    df.columns = ["ds", "y"]

    # interpolation of the values
    df = df.set_index("ds").resample("h").asfreq()
    df["y"] = df["y"].interpolate(method="time")
    df = df.reset_index()

    # model training
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
    )
    model.fit(df)

    # future dates for prediction max 31 days month
    future = model.make_future_dataframe(
        periods=24 * 31,
        freq="h",
        include_history=False,
    )
    future = future[(future["ds"] >= start_date) & (future["ds"] <= end_date)]

    # result formating
    forecast = model.predict(future)
    station_preds = {
        row["ds"].strftime("%Y-%m-%d %H:%M:%S"): round(row["yhat"], 5)
        for _, row in forecast.iterrows()
    }

    predictions[str(station_code)] = station_preds

# save to json
with open("predictions/predictions_task_2.json", "w") as f:
    json.dump({"target": predictions}, f, indent=4)
