import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar datos
measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Fusionar datos de medición e instrumento
merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])
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


station_code, pollutant_code, start_date, end_date = input_preparer("Station code: 205 | pollutant: SO2   | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00", pollutant_df)

print
def data_filter(StatCode, ItCode, start_date, end_date):
    filtered_data = merged_df[
        (merged_df["Station code"] == StatCode) &
        (merged_df["Item code"] == ItCode) &
        (merged_df["Measurement date"] >= start_date) &
        (merged_df["Measurement date"] <= end_date)
    ]
    
    # Verificar si hay datos después del filtrado
    if filtered_data.empty:
        print(f"¡ADVERTENCIA! No hay datos para: Estación {StatCode}, Contaminante {ItCode}, Periodo {start_date} - {end_date}")
    
    return filtered_data

def prepare_features(df):
    # Crear características temporales
    df_features = df.copy()
    df_features["hour"] = df_features["Measurement date"].dt.hour
    df_features["dayofweek"] = df_features["Measurement date"].dt.dayofweek
    df_features["month"] = df_features["Measurement date"].dt.month
    df_features["day"] = df_features["Measurement date"].dt.day
    
    # Características adicionales que pueden ayudar a detectar anomalías
    df_features["rolling_mean_3h"] = df_features["Average value"].rolling(window=3, min_periods=1).mean()
    df_features["rolling_std_3h"] = df_features["Average value"].rolling(window=3, min_periods=1).std()
    
    # Llenar valores NaN que pueden resultar de las operaciones rolling
    df_features = df_features.fillna(method="bfill").fillna(method="ffill")
    
    return df_features

def train_anomaly_detector(df_filtered):
    if df_filtered.empty:
        print("No se puede entrenar el modelo: conjunto de datos vacío.")
        return None
    
    # Preparar características
    df_features = prepare_features(df_filtered)
    
    # Definir características y objetivo
    features = ["Average value", "hour", "dayofweek", "month", "day", 
               "rolling_mean_3h", "rolling_std_3h"]
    
    # Asegurarse de que todas las características existen
    X = df_features[[col for col in features if col in df_features.columns]]
    y = df_features["Instrument status"]
    
    print(f"Entrenando modelo con {len(X)} instancias y {X.shape[1]} características")
    
    # Dividir datos en entrenamiento y prueba
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = model.predict(X_test)
        print("Reporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Identificar y mostrar anomalías
        anomalies = df_features[model.predict(X) == 1]  # Suponiendo que 1 representa anomalías
        print(f"Se encontraron {len(anomalies)} anomalías")
        
        if not anomalies.empty:
            print("\nPrimeras 5 anomalías detectadas:")
            print(anomalies[["Measurement date", "Average value", "Instrument status"]].head())
        
        return model, anomalies
    except ValueError as e:
        print(f"Error al dividir los datos: {e}")
        if len(np.unique(y)) < 2:
            print("El conjunto de datos tiene solo una clase. Se necesitan al menos dos clases para entrenar el modelo.")
        return None, None

# Procesar cada combinación estación-contaminante-periodo
results = {}

for entry in input_list:
    print("\n" + "="*50)
    print(f"Procesando: {entry}")
    print("="*50)
    
    station_code, pollutant_code, start_date, end_date = input_preparer(entry, pollutant_df)
    
    if station_code is None:
        continue
    
    df_filtered = data_filter(station_code, pollutant_code, start_date, end_date)
    
    if not df_filtered.empty:
        model, anomalies = train_anomaly_detector(df_filtered)
        if model is not None:
            results[(station_code, pollutant_code)] = {
                'model': model,
                'anomalies': anomalies,
                'data_size': len(df_filtered)
            }
    else:
        print(f"No se encontraron datos para esta combinación.")

# Resumen final
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS")
print("="*50)

for (station, pollutant), result in results.items():
    pollutant_name = pollutant_df[pollutant_df["Item code"] == pollutant]["Item name"].values[0]
    print(f"Estación {station}, Contaminante {pollutant_name}:")
    print(f"  - Total de datos: {result['data_size']}")
    print(f"  - Anomalías detectadas: {len(result['anomalies'])}")
    print("  - Periodos con anomalías:")
    
    if not result['anomalies'].empty:
        # Agrupar anomalías por día para un resumen más compacto
        anomalies_by_day = result['anomalies'].groupby(result['anomalies']["Measurement date"].dt.date).size()
        for day, count in anomalies_by_day.items():
            print(f"      * {day}: {count} anomalías")
    else:
        print("      * No se encontraron anomalías")
    print()

