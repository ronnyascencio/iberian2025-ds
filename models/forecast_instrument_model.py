"""
Partimos de la hipótesis de que todas las variables afectan al estado del instrumento.
Ya que 'Instrument Status' es de categoría, intentaremos un modelo de clasificación.
"""
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np

def input_preparer(line, pollutant_data):
    parts = line.split("|")

    station_code = int(parts[0].split(":")[1].strip())
    pollutant = parts[1].split(":")[1].strip()
    start_date, end_date = parts[2].split("Period:")[1].strip().split(" - ")

    pollutant_code = pollutant_data[pollutant_data["Item name"]==pollutant]["Item code"].values[0]

    start_date = pd.to_datetime(str(start_date))
    end_date = pd.to_datetime(str(end_date))

    return station_code, pollutant_code, start_date, end_date

def filter(merged_df, StatCode, ItCode):
    """
    Filter the merged dataframe based on station code, item code, and date range

    Args:
        merged_df: DataFrame with merged measurement and instrument data
        StatCode: Station code to filter
        ItCode: Item (pollutant) code to filter
        end_date: End date of the period
        start_date: Start date of the period

    Returns:
        DataFrame: Filtered dataframe
    """
    return merged_df[
        (merged_df["Station code"] == StatCode) &
        (merged_df["Item code"] == ItCode)
    ]

def classification_forecast(filtered_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Forecast the instrument status using a classification model.

    Args:
        filtered_df (pd.DataFrame): Filtered dataframe with measurement and instrument data
        start_date (pd.Timestamp): Start date of the period
        end_date (pd.Timestamp): End date of the period

    Returns:
        pd.DataFrame: Dataframe with forecasted instrument status
    """
    # Copy dataframe to avoid modifications
    df = filtered_df.copy()

    # Get available pollutants
    pollutants = [col for col in ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
                 if col in df.columns]

    # Temporal features
    df['hour'] = df['Measurement date'].dt.hour
    df['day_of_week'] = df['Measurement date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Lag features
    lags = [1, 3, 6, 12]
    for lag in lags:
        for col in pollutants:
            df[f'{col}_lag_{lag}'] = df.groupby(['Station code', 'Item code'])[col].shift(lag)

    # Rolling statistics
    for col in pollutants:
        df[f'{col}_rolling_mean_24h'] = df.groupby(['Station code', 'Item code'])[col].transform(lambda x: x.rolling(24).mean())
        df[f'{col}_rolling_std_24h'] = df.groupby(['Station code', 'Item code'])[col].transform(lambda x: x.rolling(24).std())

    # Dynamic feature list - Remove 'Average value' from base_features
    base_features = ['hour', 'day_of_week', 'is_weekend', 'Station code', 'Item code',
                    'Latitude', 'Longitude']  # Removed 'Average value'
    lag_features = [f'{col}_lag_{lag}' for col in pollutants for lag in lags]
    rolling_features = [f'{col}_{stat}' for col in pollutants
                       for stat in ['rolling_mean_24h', 'rolling_std_24h']]

    features = base_features + pollutants + lag_features + rolling_features

    # Train-test split
    split_date = df['Measurement date'].max() - pd.Timedelta(days=90)
    train = df[df['Measurement date'] < split_date]
    test = df[df['Measurement date'] >= split_date]
    recent_data = test.iloc[-1]

    X_train = train[features]
    y_train = train['Instrument status']
    X_test = test[features]
    y_test = test['Instrument status']

    # Class mapping
    unique_classes = np.sort(y_train.unique())
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_train_mapped = y_train.map(class_mapping)
    y_test_mapped = y_test.map(class_mapping)

    # Train model
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_classes),
        random_state=42
    )
    model.fit(X_train, y_train_mapped)

    # Initialize future values using the last known values
    last_values = df.iloc[-1].copy()

    # Create future predictions dataframe with all necessary features
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    future_df = pd.DataFrame(index=future_dates)

    # Add temporal features
    future_df['hour'] = future_df.index.hour
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)

    # Add static features - make sure to use only the features we defined
    static_cols = ['Station code', 'Item code', 'Latitude', 'Longitude']
    for col in static_cols:
        future_df[col] = last_values[col]

    # Propagate values forward with some random variation
    for col in pollutants:
        base_value = last_values[col]
        std_dev = df[col].std() * 0.1  # 10% of historical standard deviation
        future_df[col] = np.random.normal(base_value, std_dev, len(future_df))

        # Update lag features
        for lag in lags:
            future_df[f'{col}_lag_{lag}'] = future_df[col].shift(lag)

        # Update rolling features
        future_df[f'{col}_rolling_mean_24h'] = future_df[col].rolling(24).mean()
        future_df[f'{col}_rolling_std_24h'] = future_df[col].rolling(24).std()

    # Fill NaN values with last known values
    future_df = future_df.fillna(method='ffill')

    # Make predictions using only available features
    available_features = [f for f in features if f in future_df.columns]
    predictions = model.predict(future_df[available_features])

    # Map predictions back to original classes
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    future_df['predicted_status'] = [reverse_mapping[p] for p in predictions]

    return future_df

measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])

input_string = """
Station code: 205 | pollutant: SO2   | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00
Station code: 209 | pollutant: NO2   | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00
Station code: 223 | pollutant: O3    | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00
Station code: 224 | pollutant: CO    | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00
Station code: 226 | pollutant: PM10  | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00
Station code: 227 | pollutant: PM2.5 | Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00
"""

iter_str = iter(input_string.splitlines()[1:])

predictions = {}

for line in iter_str:
    station_code, pollutant_code, start_date, end_date = input_preparer(line, pollutant_df)
    filtered_df = filter(merged_df, station_code, pollutant_code)

    # Forecast instrument status
    forecasted_df = classification_forecast(filtered_df, start_date, end_date)

    # Print forecasted instrument status
    future_status_dict = {
        index.strftime('%Y-%m-%d %H:%M:%S'): status
        for index, status in forecasted_df['predicted_status'][forecasted_df['predicted_status'] != 0].items()
    }
    predictions[str(station_code)] = future_status_dict

# save to json
with open("predictions/predictions_task_3.json", "w") as f:
    json.dump({"target": predictions}, f, indent=4)