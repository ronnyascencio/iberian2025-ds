import json
import os

import numpy as np
import pandas as pd

# data load
instrument = pd.read_csv("data/raw/instrument_data.csv")
pollutant = pd.read_csv("data/raw/pollutant_data.csv")

# converting date time
instrument["Measurement date"] = pd.to_datetime(instrument["Measurement date"])

# mapping
item_to_pollutant = dict(zip(pollutant["Item code"], pollutant["Item name"]))

# Filtering
normal_measurements = instrument[instrument["Instrument status"] == 0].copy()
normal_measurements["Pollutant"] = normal_measurements["Item code"].map(
    item_to_pollutant
)

# Q1:
so2_data = normal_measurements[normal_measurements["Pollutant"] == "SO2"].copy()
so2_data["Date"] = so2_data["Measurement date"].dt.date
q1 = (
    so2_data.groupby(["Station code", "Date"])["Average value"]
    .mean()
    .groupby("Station code")
    .mean()
    .round(5)
    .to_dict()
)

# Q2:
co_data = normal_measurements[
    (normal_measurements["Pollutant"] == "CO")
    & (normal_measurements["Station code"] == 209)
].copy()
co_data["Month"] = co_data["Measurement date"].dt.month
seasons = {
    12: 1,
    1: 1,
    2: 1,  # Invierno
    3: 2,
    4: 2,
    5: 2,  # Primavera
    6: 3,
    7: 3,
    8: 3,  # Verano
    9: 4,
    10: 4,
    11: 4,
}  # Otoño
co_data["Season"] = co_data["Month"].map(seasons)
q2 = co_data.groupby("Season")["Average value"].mean().round(5).to_dict()

# Q3:
o3_data = normal_measurements[normal_measurements["Pollutant"] == "O3"].copy()
o3_data["Hour"] = o3_data["Measurement date"].dt.hour
q3 = o3_data.groupby("Hour")["Average value"].std().idxmax()

# Q4:
abnormal_data = instrument[instrument["Instrument status"] == 9]
q4 = abnormal_data["Station code"].value_counts().idxmax()

# Q5:
not_normal = instrument[instrument["Instrument status"] != 0]
q5 = not_normal["Station code"].value_counts().idxmax()

# Q6:
pm25_data = normal_measurements[normal_measurements["Pollutant"] == "PM2.5"].copy()

pm25_data["Category"] = pd.cut(
    pm25_data["Average value"],
    bins=[-np.inf, 15, 35, 75, np.inf],
    labels=["Good", "Normal", "Bad", "Very bad"],
)
q6 = pm25_data["Category"].value_counts().to_dict()

# Formating
answers = {
    "target": {
        "Q1": q1,
        "Q2": q2,
        "Q3": int(q3),
        "Q4": int(q4),
        "Q5": int(q5),
        "Q6": q6,
    }
}

# SavingJSON
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "questions.json"), "w") as f:
    json.dump(answers, f, indent=4)

print("¡Answer saved in  predictions/questions.json!")
