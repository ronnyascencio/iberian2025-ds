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

"""
Second part: Answering questions.
"""

# Create results dict:
results = {}

"""Q1 - Average daily SO2 concentration across all districts"""
### Ronald
daily_station_avg = merged_df.groupby('Station code')['SO2'].mean().mean()
q1_result = round(daily_station_avg, 5)
results["Q1"] = float(q1_result)


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
### Ronald
def classify_pm25_levels(row, thresholds):
    """Classify PM2.5 levels based on thresholds."""
    if row['PM2.5'] <= thresholds['Good']:
        return 'Good'
    elif row['PM2.5'] <= thresholds['Normal']:
        return 'Normal'
    elif row['PM2.5'] <= thresholds['Bad']:
        return 'Bad'
    else:
        return 'Very bad'


# Step 1: Extract PM2.5 thresholds from pollutant_data
pm25_thresholds = pollutant_df[pollutant_df['Item name'] == "PM2.5"].iloc[0]
thresholds = {
    'Good': pm25_thresholds['Good'],
    'Normal': pm25_thresholds['Normal'],
    'Bad': pm25_thresholds['Bad'],
    'Very bad': pm25_thresholds['Very bad']
}

# Step 2: Classify PM2.5 levels in measurement_data
measurement_data = measurement_df.copy()
measurement_data['PM2.5 Status'] = measurement_data.apply(classify_pm25_levels, axis=1, thresholds=thresholds)

# Step 3: Count occurrences of each status
pm25_status_counts = measurement_data['PM2.5 Status'].value_counts().to_dict()

# Step 4: Format results for Q6
results["Q6"] = {status: int(count) for status, count in pm25_status_counts.items()}

"""
Save results to JSON file
"""
with open('predictions/questions.json', 'w') as f:
    json.dump({'target': results}, f, indent=4)