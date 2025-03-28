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