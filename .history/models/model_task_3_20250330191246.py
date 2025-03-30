
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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report





# load measurement data
measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Fusionar datos de medición e instrumento
merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])
print("Fechas únicas en merged_df:")
print(merged_df["Measurement date"].unique())
print(merged_df.columns)
# Preparar entradas
input_list = [
    "Station code: 205 | pollutant: SO2   | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00",
    "Station code: 209 | pollutant: NO2   | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00",
    "Station code: 223 | pollutant: O3    | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00",
    "Station code: 224 | pollutant: CO    | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00",
    "Station code: 226 | pollutant: PM10  | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00",
    "Station code: 227 | pollutant: PM2.5 | Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00",
]

def input_preparer(line, pollutant_data):
    parts = line.split("|")
    
    station_code = int(parts[0].split(":")[1].strip())
    pollutant = parts[1].split(":")[1].strip()
    start_date, end_date = parts[2].split("Period:")[1].strip().split(" - ")

    
    # Buscar el código de contaminante y manejar posibles errores
    matching_pollutants = pollutant_data[pollutant_data["Item name"] == pollutant.strip()]
    if matching_pollutants.empty:
        print(f"¡ADVERTENCIA! No se encontró el contaminante: '{pollutant}' en los datos.")
        print("Contaminantes disponibles:", pollutant_data["Item name"].unique())
        return None, None, None, None
    
    pollutant_code = matching_pollutants["Item code"].values[0]
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    return station_code, pollutant_code, start_date, end_date



iter_str = iter(input_string.splitlines()[1:])

for line in iter_str:
    station_code, pollutant_code, start_date, end_date = input_preparer(line, pollutant_df)



def data_filter(StatCode, ItCode, start_date, end_date):
    print("Filtrando con:")
    print(f"  Estación: {StatCode}")
    print(f"  Contaminante (código): {ItCode}")
    print(f"  Fechas: {start_date} -> {end_date}")
    print("Tipos:")
    print(type(StatCode), type(ItCode))
    print(merged_df["Station code"].dtype, merged_df["Item code"].dtype)

    print("\n\n------------------------------\n\n")





