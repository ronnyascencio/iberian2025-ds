import pandas as pd

# Load data
measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Merge measurement and instrument data
merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])

# Filter only 'Normal' instrument status
df = merged_df[merged_df["Instrument status"] == 0]

### Q1 - Average daily SO2 concentration across all districts
df_so2 = df[["Measurement date", "Station code", "SO2"]].dropna()
df_so2["date"] = df_so2["Measurement date"].dt.date
daily_station_avg = df_so2.groupby(["Station code", "date"])["SO2"].mean()
station_avg = daily_station_avg.groupby("Station code").mean()
q1_result = round(station_avg.mean(), 5)



"""Q2: Analyse how pollution levels vary by season. 
Return the average levels of CO per season at the station 209. 
(Take the whole month of December as part of winter, March as spring, and so on.) 
Provide the answer with 5 decimals.

Q3: Which hour presents the highest variability (Standard Deviation) 
for the pollutant O3? Treat all stations as equal."""



# Q3 
# filter null O3 vals
df_o3 = df[["Measurement date", "O3"]].dropna()

#get hour
df_o3["hour"]= df_o3["Measurement date"].dt.hour

#Group hours and get std deviation
hour_std = df_o3.groupby("hour")["O3"].std()

q3_result = int(hour_std.idxmax()) # get the maximum std deviation hour

print("Q3:", q3_result)

#Q4 Which is the station code with more measurements labeled as "Abnormal data"?

# !! Abnormal data code -> 9 (Instrument status); only instrument_data has "Instrument status"
# No need to use the merged

df_abn = instrument_df[instrument_df["Instrument status"]==9]
q4_result = df_abn["Station code"].value_counts().idxmax()
print("Q4:", q4_result)




#Q5: Station with more "not normal" measurements...
# we only need measurement_data..., we filter it for "not normal"















### Q6 - PM2.5 classification
# Get item code for PM2.5
pm25_code = pollutant_df[pollutant_df["Item name"] == "PM2.5"]["Item code"].values[0]
df_pm25 = df[df["Item code"] == pm25_code]

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

# Print outputs
print("Q1:", q1_result)

print("Q6:", q6_counts)

### Q2 - Average CO concentration in station 209 by season

# aqui tu merged_df

# measurement = pd.read_csv("data/raw/measurement_data.csv")
# instrument = pd.read_csv("data/raw/instrument_data.csv")
# pollutant = pd.read_csv("data/raw/pollutant_data.csv")

# "normal meditions"(status=0)
instrument_normal = instrument_df[instrument_df["Instrument status"] == 0]

merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])

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
print(f"season debug: {station_209_co['season'].value_counts()}")
result = station_209_co.groupby("season")["CO"].mean().round(5)

"""Outer"""

output = {"target": {"Q2": {str(k): float(v) for k, v in result.to_dict().items()}}}
print(f"output: {output}")
