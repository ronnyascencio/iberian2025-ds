import pandas as pd
import json

"""
First part: Loading data.
"""
measurement_df = pd.read_csv(
    "data/raw/measurement_data.csv", parse_dates=["Measurement date"]
)
instrument_df = pd.read_csv(
    "data/raw/instrument_data.csv", parse_dates=["Measurement date"]
)
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Filter only 'Normal' instrument status from measurment df before merging.
filtered_instrument_df = instrument_df[instrument_df["Instrument status"] == 0]

# Merge measurement and instrument data
merged_df = pd.merge(
    measurement_df, filtered_instrument_df, on=["Measurement date", "Station code"]
)
merged_df.dropna(inplace=True)


# Let's try to normalize the data:
# Normalize pollutant levels using Robust Scaling
for col in ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']:
    col_min = merged_df[col].min()
    col_max = merged_df[col].max()
    merged_df[col] = (merged_df[col] - col_min) / (col_max - col_min)
# Note: This only changes the output for Q1

"""
Second part: Answering questions.
"""

# Create results dict:
results = {}

"""Q1 - Average daily SO2 concentration across all districts"""
### Raul
def q1_daily_explicit(df):
    merged_data = df.copy()
    # Paso 1: Calcular los promedios diarios de SO₂ por estación
    daily_station_avg = merged_data.groupby(["Station code", "Measurement date"])["SO2"].mean()

    # Paso 2: Calcular el promedio de los promedios diarios para cada estación
    station_daily_avg = daily_station_avg.groupby("Station code").mean()

    # Paso 3: Calcular el promedio general entre todas las estaciones
    overall_daily_avg_so2 = station_daily_avg.mean()

    # Redondear a 5 decimales
    return round(overall_daily_avg_so2, 5)

### Ronald
def q1_daily_implicit(df):
    merged_data = df.copy()
    daily_station_avg = merged_data.groupby('Station code')['SO2'].mean().mean()
    q1_result = round(daily_station_avg, 5)
    return float(q1_result)

q1_result_ex = q1_daily_explicit(merged_df)
q1_result_im = q1_daily_implicit(merged_df)
results["Q1"] = q1_result_ex



"""Q2 - Average CO concentration in station 209 by season"""
### Ronny

# "normal meditions"(status=0)
instrument_normal = instrument_df[instrument_df["Instrument status"] == 0]

merged_df = pd.merge(
    measurement_df, instrument_df, on=["Measurement date", "Station code"]
)

merged_df = merged_df[merged_df["Instrument status"] == 0]


# mix measurement and instrument_normal
merged_df["Measurement date"] = pd.to_datetime(
    merged_df["Measurement date"], format="%Y-%m-%d %H:%M:%S"
)
co_code = pollutant_df[pollutant_df["Item name"] == "CO"]["Item code"].values[0]

station_209_co = merged_df[
    (merged_df["Station code"] == 209) & (merged_df["Item code"] == co_code)
].copy()

season_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
station_209_co["month"] = station_209_co["Measurement date"].dt.month

station_209_co["season"] = station_209_co["month"].map(season_map)
# debug
result = station_209_co.groupby("season")["CO"].mean().round(5)
q2_result = {str(k): float(v) for k, v in result.to_dict().items()}
results["Q2"] = q2_result



"""Q3: Which hour presents the highest variability (Standard Deviation)
for the pollutant O3? Treat all stations as equal."""
### Raul

#get hour
merged_df["hour"]= merged_df["Measurement date"].dt.hour

#Group hours and get std deviation
hour_std = merged_df.groupby("hour")["O3"].std()

q3_result = int(hour_std.idxmax()) # get the maximum std deviation hour
results["Q3"] = q3_result


"""Q4 Which is the station code with more measurements labeled as "Abnormal data"?"""
### Raul

# !! Abnormal data code -> 9 (Instrument status); only instrument_data has "Instrument status"
# No need to use the merged

df_abn = instrument_df[instrument_df["Instrument status"]==9]
q4_result = df_abn["Station code"].value_counts().idxmax()
results["Q4"] = int(q4_result)


"""Q5: Station with more "not normal" measurements..."""
### Raul
# we only need instrument_data..., we filter it for "not normal"-->!=0
df_nnorm = instrument_df[instrument_df["Instrument status"] != 0]

# count how many times a station appears and gets the one with most
q5_result = int(df_nnorm["Station code"].value_counts().idxmax())
results["Q5"] = q5_result


"""Q6 - PM2.5 classification"""
### Raul
# Get item code for PM2.5
pm25_code = pollutant_df[pollutant_df["Item name"] == "PM2.5"]["Item code"].values[0]
df_pm25 = merged_df[merged_df["Item code"] == pm25_code]

# Get classification thresholds
row = pollutant_df[pollutant_df["Item name"] == "PM2.5"]
good = row["Good"].values[0]
normal = row["Normal"].values[0]
bad = row["Bad"].values[0]
very_bad = row["Very bad"].values[0]


# Classification function
def classify_pm25(val):
    if val <= good:
        return "Good"
    elif val <= normal:
        return "Normal"
    elif val <= bad:
        return "Bad"
    else:
        return "Very bad"


df_pm25["quality"] = df_pm25["Average value"].apply(classify_pm25)
q6_counts = df_pm25["quality"].value_counts().to_dict()

# Step 4: Format results for Q6
results["Q6"] = q6_counts

"""
Save results to JSON file
"""
with open('predictions/questions.json', 'w') as f:
    json.dump({'target': results}, f, indent=4)