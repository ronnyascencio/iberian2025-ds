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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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


def data_filter(StatCode, ItCode, start_date, end_date):
    print("Filtrando con:")
    print(f"  Estación: {StatCode}")
    print(f"  Contaminante (código): {ItCode}")
    print(f"  Fechas: {start_date} -> {end_date}")
    print("Tipos:")
    print(type(StatCode), type(ItCode))
    print(merged_df["Station code"].dtype, merged_df["Item code"].dtype)

    print("\n\n------------------------------\n\n")



def prepare_features(df):

    # Crear características temporales
    df_features = df.copy()
    df_features["station_code"] = df_features["Station code"]
    df_features["latitude"] = df_features["Latitude"]
    df_features["longitude"] = df_features["Longitude"]
    df_features["SO2"] = df_features["SO2"]
    df_features["NO2"] = df_features["NO2"]
    df_features["O3"] = df_features["O3"]
    df_features["CO"] = df_features["CO"]
    df_features["PM10"] = df_features["PM10"]
    df_features["PM2.5"] = df_features["PM2.5"]
    df_features["item_code"] = df_features["Item code"]
    df_features["avg_value"] = df_features["Average value"]
    df_features["hour"] = df_features["Measurement date"].dt.hour
    df_features["dayofweek"] = df_features["Measurement date"].dt.dayofweek
    df_features["month"] = df_features["Measurement date"].dt.month
    df_features["day"] = df_features["Measurement date"].dt.day
    df_features["pollutant_code"] = df_features["Item code"]
    df_features["station"] = df_features["Station code"]
    df_features["latitude_abs"] = df_features["Latitude"].abs()
    df_features["longitude"] = df_features["Longitude"]

    # Características adicionales que pueden ayudar a detectar anomalías
    df_features["rolling_mean_2h"] = df_features["Average value"].rolling(window=2, min_periods=1).mean()
    df_features["rolling_std_2h"] = df_features["Average value"].rolling(window=2, min_periods=1).std()

    df_features["rolling_mean_3h"] = df_features["Average value"].rolling(window=3, min_periods=1).mean()
    df_features["rolling_std_3h"] = df_features["Average value"].rolling(window=3, min_periods=1).std()

    df_features["rolling_std_12h"] = df_features["Average value"].rolling(window=12, min_periods=1).std()
    df_features["rolling_mean_12h"] = df_features["Average value"].rolling(window=12, min_periods=1).mean()

    df_features["rolling_mean_10h"] = df_features["Average value"].rolling(window=10, min_periods=1).mean()
    df_features["rolling_std_10h"] = df_features["Average value"].rolling(window=10, min_periods=1).std()
    
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
    # hour y month añaden 0.02 de precision
    # pollutant code no da casi nada, station code tampoco


    features = ["Average value", "CO", "PM10", "rolling_mean_10h", "rolling_std_10h", "rolling_std_3h"]

    # Asegúrate de que Measurement date e Instrument status NO estén en X
    #X = df_filtered[features].fillna(0)  # Si hay NaNs
    y = (df_features["Instrument status"] != 0).astype(int) 
    

    # Asegurarse de que todas las características existen
    X = df_features[[col for col in features if col in df_features.columns]]
    df_features["is_anomaly"] = (df_features["Instrument status"] != 0).astype(int) # lo hago binario, el tipo 0 es sin anomalía
    y = df_features["is_anomaly"]
    
    print(f"Entrenando modelo con {len(X)} instancias y {X.shape[1]} características")
    
    # Dividir datos en entrenamiento y prueba
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print("\n\n\n=== ENTRENANDO LOGISTIC REGRESSION ===")
        model_log = LogisticRegression(max_iter=100, n_jobs=-1)
        start_time = time.time()
        model_log.fit(X_train, y_train)
        y_pred_log = model_log.predict(X_test)
        print("Reporte de clasificación (Logistic Regression):")
        print(classification_report(y_test, y_pred_log))

        print("\n=== ENTRENANDO GRADIENT BOOSTING (sklearn) ===")
        model_gb = GradientBoostingClassifier(n_estimators=5, random_state=42)
        start_time = time.time()
        model_gb.fit(X_train, y_train)
        y_pred_gb = model_gb.predict(X_test)
        print("Reporte de clasificación (Gradient Boosting):")
        print(classification_report(y_test, y_pred_gb))

        print("\n=== ENTRENANDO HISTOGRAM GRADIENT BOOSTING (sklearn) ===")
        model_hgb = HistGradientBoostingClassifier(random_state=42)
        start_time = time.time()
        model_hgb.fit(X_train, y_train)
                print("Tiempo de entrenamiento: {:.2f} segundos".format(time.time() - start_time))

        y_pred_hgb = model_hgb.predict(X_test)
        print("Reporte de clasificación (HistGradientBoosting):")
        print(classification_report(y_test, y_pred_hgb))

        print("\n=== ENTRENANDO LIGHTGBM ===")
        model_lgbm = LGBMClassifier(n_estimators=5, random_state=42)
        start_time = time.time()
        model_lgbm.fit(X_train, y_train)
        y_pred_lgbm = model_lgbm.predict(X_test)
        print("Reporte de clasificación (LightGBM):")
        print(classification_report(y_test, y_pred_lgbm))

        print("\n=== ENTRENANDO XGBOOST ===")
        start_time = time.time()
        model_xgb = XGBClassifier(n_estimators=5, use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_xgb.fit(X_train, y_train)
        y_pred_xgb = model_xgb.predict(X_test)
        print("Reporte de clasificación (XGBoost):")
        print(classification_report(y_test, y_pred_xgb))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Entrenar el modelo
        model = RandomForestClassifier(n_estimators=2, n_jobs=-1, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        print("Tiempo de entrenamiento: {:.2f} segundos".format(time.time() - start_time))

        importances = model.feature_importances_
        for name, importance in zip(X.columns, importances):
            print(f"{name}: {importance:.4f}")

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


train_anomaly_detector(merged_df)










""" test_df = merged_df[
(merged_df["Station code"] == 205) &
(merged_df["Item code"] == 0)
]
print(test_df["Measurement date"].min(), "->", test_df["Measurement date"].max())
print(test_df.tail(5))"""

   
"""    filtered_data = merged_df[
        (merged_df["Station code"] == StatCode) &
        (merged_df["Item code"] == ItCode) &
        (merged_df["Measurement date"] >= start_date) &
        (merged_df["Measurement date"] <= end_date)]"""
    
"""   print("\n\n------\n\n")
print("Fechas mín y máx en merged_df:", merged_df["Measurement date"].min(), "->", merged_df["Measurement date"].max())
print("Estaciones únicas:", merged_df["Station code"].unique())
print("Contaminantes únicos:", merged_df["Item code"].unique())"""
    
    
"""  # Verificar si hay datos después del filtrado
    if filtered_data.empty:
        print(f"¡ADVERTENCIA! No hay datos para: Estación {StatCode}, Contaminante {ItCode}, Periodo {start_date} - {end_date}")
    
    return filtered_data"""

"""df_filtered = data_filter(station_code, pollutant_code, start_date, end_date)
print(df_filtered.head())
print(f"Filas filtradas: {len(df_filtered)}")"""
