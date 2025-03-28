import pandas as pd

# datasets
measurement = pd.read_csv("data/raw/measurement_data.csv")
instrument = pd.read_csv("data/raw/instrument_data.csv")
pollutant = pd.read_csv("data/raw/pollutant_data.csv")

# "normal meditions"(status=0)
instrument_normal = instrument[instrument["Instrument status"] == 0]

merged = pd.merge(
    measurement, instrument_normal, on=["Station code", "Measurement date"], how="inner"
)
"""Q2"""

# mix measurement and instrument_normal
merged["Measurement date"] = pd.to_datetime(
    merged["Measurement date"], format="%Y-%m-%d %H:%M:%S"
)
co_code = pollutant[pollutant["Item name"] == "CO"]["Item code"].values[0]

station_209_co = merged[
    (merged["Station code"] == 209) & (merged["Item code"] == co_code)
].copy()

season_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
station_209_co["month"] = station_209_co["Measurement date"].dt.month

station_209_co["season"] = station_209_co["month"].map(season_map)
# debug
print(station_209_co["season"].value_counts())
result = station_209_co.groupby("season")["CO"].mean().round(5)

"""Outer"""

output = {"target": {"Q2": {str(k): float(v) for k, v in result.to_dict().items()}}}
print(output)
