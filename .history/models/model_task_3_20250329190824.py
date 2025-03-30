import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar datos
print("Cargando datos...")
measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Mostrar información de diagnóstico
print("\n=== DIAGNÓSTICO DE DATOS ===")
print(f"measurement_df: {len(measurement_df)} filas")
print(f"Estaciones únicas: {sorted(measurement_df['Station code'].unique())}")
print(f"Fechas: {measurement_df['Measurement date'].min()} a {measurement_df['Measurement date'].max()}")

print(f"\ninstrument_df: {len(instrument_df)} filas")
print(f"Estaciones únicas: {sorted(instrument_df['Station code'].unique())}")

print(f"\npollutant_df: {len(pollutant_df)} filas")
print("Contaminantes disponibles:")
print(pollutant_df[['Item code', 'Item name']])

# Crear tabla de mapeo directa
pollutant_map = {
    "SO2": None,    # Se actualizará con códigos reales
    "NO2": None,
    "O3": None,
    "CO": None,
    "PM10": None, 
    "PM2.5": None
}

# Actualizar códigos basados en datos reales
for pollutant in pollutant_map.keys():
    matches = pollutant_df[pollutant_df["Item name"].str.contains(pollutant, case=False)]
    if not matches.empty:
        pollutant_map[pollutant] = matches["Item code"].iloc[0]
        print(f"Mapeado: {pollutant} -> {pollutant_map[pollutant]}")
    else:
        print(f"¡ADVERTENCIA! No se encontró coincidencia para: {pollutant}")

# Preparar lista de tareas
tasks = [
    {"station": 205, "pollutant": "SO2", "start": "2023-11-01", "end": "2023-11-30"},
    {"station": 209, "pollutant": "NO2", "start": "2023-09-01", "end": "2023-09-30"},
    {"station": 223, "pollutant": "O3", "start": "2023-07-01", "end": "2023-07-31"},
    {"station": 224, "pollutant": "CO", "start": "2023-10-01", "end": "2023-10-31"},
    {"station": 226, "pollutant": "PM10", "start": "2023-08-01", "end": "2023-08-31"},
    {"station": 227, "pollutant": "PM2.5", "start": "2023-12-01", "end": "2023-12-31"}
]

# Función mejorada para fusionar datos
def get_filtered_data(station, pollutant_name, start_date, end_date):
    # Obtener código de contaminante
    pollutant_code = pollutant_map.get(pollutant_name)
    if pollutant_code is None:
        print(f"No se encontró código para contaminante: {pollutant_name}")
        return None
    
    # Convertir fechas
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date + " 23:59:59")  # Incluir todo el último día
    
    # Filtrar datos de mediciones
    measurements = measurement_df[
        (measurement_df["Station code"] == station) &
        (measurement_df["Item code"] == pollutant_code) &
        (measurement_df["Measurement date"] >= start) &
        (measurement_df["Measurement date"] <= end)
    ]
    
    if measurements.empty:
        print(f"No hay mediciones para estación {station}, contaminante {pollutant_name}")
        # Verificar si hay datos para la estación con cualquier contaminante
        station_data = measurement_df[measurement_df["Station code"] == station]
        if not station_data.empty:
            print(f"  - La estación {station} tiene {len(stati