"""Task 3: Detect anomalies in data measurements
Detect instrument anomalies for the following stations and periods:



Station code: 205 | pollutant: SO2  
 | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00
Station code: 209 | pollutant: NO2  
 | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00
Station code: 223 | pollutant: O3   
 | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00
Station code: 224 | pollutant: CO  
  | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00
Station code: 226 | pollutant: PM10 
 | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00
Station code: 227 | pollutant: PM2.5 
| Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier



# load measurement data
measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Merge measurement and instrument data
merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])

# prepare input/s
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")
input_list = [ "Station code: 205 | pollutant: SO2  | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00",
"Station code: 209 | pollutant: NO2  | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00",
"Station code: 223 | pollutant: O3   | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00",
"Station code: 224 | pollutant: CO  | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00",
"Station code: 226 | pollutant: PM10 | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00",
"Station code: 227 | pollutant: PM2.5 | Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00",
]



def input_preparer(line, pollutant_data):
    parts = line.split("|")

    station_code = int(parts[0].split(":")[1].strip())
    pollutant = parts[1].split(":")[1].strip()
    start_date, end_date = parts[2].split("Period:")[1].strip().split(" - ")

    pollutant_code = pollutant_data[pollutant_data["Item name"]==pollutant]["Item code"].values[0]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    return station_code, pollutant_code, start_date, end_date


print(input_preparer("Station code: 209 | pollutant: NO2  | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00", pollutant_df))


def data_filter(StatCode, ItCode, start_date, end_date):
    # filter for each station code:

    return merged_df[
    (merged_df["Station code"].dropna() == StatCode) &
    (merged_df["Item code"].dropna() == ItCode) &
    (merged_df["Measurement date"] >= start_date) &
    (merged_df["Measurement date"] <= end_date)
]


def machine_learning(df_filtered):
    df_filtered["hour"] = df_filtered["Measurement date"].dt.hour
    df_filtered["dayofweek"] = df_filtered["Measurement date"].dt.dayofweek

    X = df_filtered[["Average value", "hour", "dayofweek"]]
    y = df_filtered["Instrument status"]

    model = RandomForestClassifier()
    model.fit(X, y)
    print("Modelo con ", len(X), "instancias")

    return




for each in input_list:
    station_code, pollutant_code, start_date, end_date = input_preparer(each, pollutant_df)
    df_filtered = data_filter(station_code, pollutant_code, start_date, end_date)
    machine_learning(df_filtered)

