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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns
import matplotlib.pyplot as plt
import shap

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
    df_features = df_features.bfill().ffill()
    
    return df_features


def train_anomaly_detector(df_filtered):

    if df_filtered.empty:
        print("No se puede entrenar el modelo: conjunto de datos vacío.")
        return None
    
    # Preparar características
    df_features = prepare_features(df_filtered)
    
    le = LabelEncoder()
    y = le.fit_transform(df_features["Instrument status"])
    
    # Definir características y objetivo
    # hour y month añaden 0.02 de precision
    # pollutant code no da casi nada, station code tampoco


    features = ["Average value", "CO", "PM10", "rolling_std_10h","SO2","PM2.5"]

    # Asegúrate de que Measurement date e Instrument status NO estén en X
    #X = df_filtered[features].fillna(0)  # Si hay NaNs
    
    # Asegurarse de que todas las características existen
    X = df_features[[col for col in features if col in df_features.columns]]
    # df_features["is_anomaly"] = (df_features["Instrument status"] != 0).astype(int) # lo hago binario, el tipo 0 es sin anomalía
    
    
    print(f"Entrenando modelo con {len(X)} instancias y {X.shape[1]} características")
    
    # Dividir datos en entrenamiento y prueba
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print("\n=== ENTRENANDO XGBOOST ===")
        model = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
        #vamos a intentar balancear la clase...
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

        start_time = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred_xgb = model.predict(X_test)
        print("Tiempo de entrenamiento: {:.2f} segundos".format(time.time() - start_time))
        y_pred = model.predict(X_test)

        unique_classes = np.unique(y_train)

        # Mostrar clases originales más difíciles de predecir
        original_preds = le.inverse_transform(y_pred_xgb)
        original_true = le.inverse_transform(y_test)

        print("\nDistribución de clases predichas (originales):")
        print(pd.Series(original_preds).value_counts().sort_index())

        print("\nDistribución de clases verdaderas (originales):")
        print(pd.Series(original_true).value_counts().sort_index())

        importances = model.feature_importances_
        for name, importance in zip(X.columns, importances):
            print(f"{name}: {importance:.4f}")

        # Identificar y mostrar anomalías
        anomalies = df_features[model.predict(X) == 1]  # Suponiendo que 1 representa anomalías
        print(f"Se encontraron {len(anomalies)} anomalías")
        
        if not anomalies.empty:
            print("\nPrimeras 5 anomalías detectadas:")
            print(anomalies[["Measurement date", "Average value", "Instrument status"]].head())

        print("\n=== IMPORTANCIA DE VARIABLES POR TIPO DE FALLO ===")
        
        print("\n=== EVALUACIÓN POR CLASE DE FALLO CON MODELOS BINARIOS ===")


        eachfeatures = [
            # Clase 1
            ["station_code", "SO2", "O3", "avg_value", "month"],
            # Clase 2
            ["longitude", "SO2", "O3", "item_code", "avg_value", "hour", "dayofweek", "rolling_mean_12h"],
            # Clase 4
            ["station_code", "latitude", "longitude", "NO2", "month"],
            # Clase 8
            ["station_code", "longitude", "item_code", "month", "day", "rolling_std_12h", "rolling_mean_12h"],
            # Clase 9
            ["station_code", "SO2", "O3", "item_code", "avg_value", "day", "rolling_std_12h", "rolling_mean_12h"]
        ]


        i=0
        for clase in np.unique(y_train):
            class_error = le.inverse_transform([clase])[0]
            if clase == 0:
                continue  # saltar clase 'normal'
            
                

            y_bin = (y == clase).astype(int)
            X_bin = df_features[[col for col in eachfeatures[i] if col in df_features.columns]]

            X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

            model_bin = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss', random_state=42)



            model_bin.fit(X_train_bin, y_train_bin)
            y_pred_bin = model_bin.predict(X_test_bin)
            
            print(f"\n=== EXPLICABILIDAD SHAP PARA CLASE {class_error} ===")
            explainer_bin = shap.Explainer(model_bin, X_train_bin)
            shap_values_bin = explainer_bin(X_test_bin)
            shap.summary_plot(shap_values_bin, X_test_bin, plot_type="bar")

            

            print(f"\n--- Modelo para clase {class_error} vs resto ---")
            print("Reporte de clasificación (modelo binario XGBoost):")
            print(classification_report(y_test_bin, y_pred_bin))

            importances = model_bin.feature_importances_
            print("Importancia de características:")
            for name, importance in zip(X_bin.columns, importances):
                print(f"{name}: {importance:.4f}")
            i+=1

        print("\n=== MATRIZ DE CORRELACIÓN ENTRE FEATURES ===")
        plt.figure(figsize=(12, 10))
        corr = df_features.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de correlación")
        plt.tight_layout()
        plt.show()

        print("\n=== EXPLICABILIDAD CON SHAP (modelo global) ===")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")

        return model, anomalies
    except ValueError as e:
        print(f"Error al dividir los datos: {e}")
        if len(np.unique(y)) < 2:
            print("El conjunto de datos tiene solo una clase. Se necesitan al menos dos clases para entrenar el modelo.")
        return None, None


model, _ = train_anomaly_detector(merged_df)
if model is None:
    print("No se entrenó el modelo. Se detiene la ejecución.")
    exit()
    
print("\n=== PREDICCIÓN DE ANOMALÍAS EN LOS PERIODOS DE INPUT ===")
for input_line in input_list:
    print("\n" + "=" * 50)
    print(f"Procesando: {input_line}")
    print("=" * 50)

    station_code, pollutant_code, start_date, end_date = input_preparer(input_line, pollutant_df)
    if station_code is None:
        continue

    df_input = merged_df[
        (merged_df["Station code"] == station_code) &
        (merged_df["Item code"] == pollutant_code) &
        (merged_df["Measurement date"] >= start_date) &
        (merged_df["Measurement date"] <= end_date)
    ]

    if df_input.empty:
        # Intentar usar todos los datos disponibles de esa estación y contaminante
        df_input = merged_df[
            (merged_df["Station code"] == station_code) &
            (merged_df["Item code"] == pollutant_code)
        ]

    df_features_input = prepare_features(df_input)
    X_input = df_features_input[[col for col in ["Average value", "CO", "PM10", "rolling_std_10h","SO2","PM2.5"] if col in df_features_input.columns]]

    y_pred_input = model.predict(X_input)
    n_anomalies = sum(pred != 0 for pred in y_pred_input)
    print(f"Predicción: se esperan aproximadamente {n_anomalies} anomalías en el periodo analizado.")

import pickle

filename = "xgboost_model_task3.pickle"
with open(filename, "wb") as file:
    pickle.dump(model, file)

print(f"\nModelo guardado en '{filename}'")
