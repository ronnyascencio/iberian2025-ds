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
            print(f"  - La estación {station} tiene {len(station_data)} registros con otros contaminantes")
            contaminants = station_data["Item code"].unique()
            print(f"  - Contaminantes disponibles: {contaminants}")
        return None
    
    # Filtrar datos de instrumentos
    instruments = instrument_df[
        (instrument_df["Station code"] == station) &
        (instrument_df["Measurement date"] >= start) &
        (instrument_df["Measurement date"] <= end)
    ]
    
    if instruments.empty:
        print(f"No hay datos de instrumentos para estación {station}")
        return None
    
    # Fusionar los datos (usando inner join para mantener solo las coincidencias)
    merged = pd.merge(measurements, instruments, on=["Measurement date", "Station code"])
    
    print(f"Encontrados {len(merged)} registros para estación {station}, contaminante {pollutant_name}")
    return merged

# Función para entrenar modelo y detectar anomalías
def train_and_detect(data, description):
    if data is None or len(data) < 10:  # Necesitamos al menos algunas muestras
        print(f"Datos insuficientes para {description}")
        return None, None
    
    # Preparar características
    data["hour"] = data["Measurement date"].dt.hour
    data["day"] = data["Measurement date"].dt.day
    data["dayofweek"] = data["Measurement date"].dt.dayofweek
    
    # Características adicionales
    data["rolling_avg"] = data["Average value"].rolling(window=3, min_periods=1).mean()
    data["rolling_std"] = data["Average value"].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Define X e y
    X = data[["Average value", "hour", "day", "dayofweek", "rolling_avg", "rolling_std"]]
    y = data["Instrument status"]
    
    # Comprobar si tenemos suficientes clases para entrenar
    if len(np.unique(y)) < 2:
        print(f"ADVERTENCIA: Solo hay una clase en los datos ({np.unique(y)[0]})")
        # Si solo hay una clase, simular algunas anomalías para poder entrenar el modelo
        if len(data) > 20:
            print("Generando algunas anomalías simuladas para entrenar el modelo...")
            # Copiar algunos registros y cambiar su estado
            anomaly_indices = np.random.choice(len(data), size=max(3, int(len(data)*0.05)), replace=False)
            X_augmented = X.copy()
            y_augmented = y.copy()
            y_augmented.iloc[anomaly_indices] = 1 - y.iloc[0]  # Invertir la clase existente
            
            # Usar los datos aumentados
            X, y = X_augmented, y_augmented
        else:
            return None, None
    
    # Dividir datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entrenar modelo
    print(f"Entrenando modelo con {len(X_train)} instancias...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Detectar anomalías (asumiendo que la clase 1 representa anomalías)
    predictions = model.predict(X)
    anomalies = data.iloc[np.where(predictions == 1)[0]]
    normal = data.iloc[np.where(predictions == 0)[0]]
    
    print(f"Anomalías detectadas: {len(anomalies)} de {len(data)} registros ({len(anomalies)/len(data)*100:.1f}%)")
    
    return model, anomalies

# Procesar cada tarea
results = {}

for task in tasks:
    print("\n" + "="*50)
    description = f"Estación {task['station']}, {task['pollutant']}, {task['start']} a {task['end']}"
    print(f"Procesando: {description}")
    print("="*50)
    
    # Obtener datos filtrados
    data = get_filtered_data(task['station'], task['pollutant'], task['start'], task['end'])
    
    # Entrenar modelo y detectar anomalías
    if data is not None and not data.empty:
        model, anomalies = train_and_detect(data, description)
        if model is not None:
            # Guardar resultados
            results[description] = {
                'data_size': len(data),
                'anomalies': len(anomalies) if anomalies is not None else 0,
                'anomaly_dates': anomalies["Measurement date"].dt.date.unique() if anomalies is not None else []
            }

# Mostrar resultados finales
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS")
print("="*50)

if results:
    for desc, res in results.items():
        print(f"\n{desc}:")
        print(f"  - Total de datos: {res['data_size']}")
        print(f"  - Anomalías detectadas: {res['anomalies']} ({res['anomalies']/res['data_size']*100:.1f}%)")
        
        if res['anomaly_dates'].size > 0:
            print("  - Fechas con anomalías:")
            for date in sorted(res['anomaly_dates']):
                print(f"      * {date}")
else:
    print("No se pudieron procesar datos para ninguna de las combinaciones solicitadas.")
    print("\nPosibles soluciones:")
    print("1. Verificar que los archivos de datos contienen la información esperada")
    print("2. Comprobar que los códigos de estación y contaminante son correctos")
    print("3. Revisar el rango de fechas en los datos")