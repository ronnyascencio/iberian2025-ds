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
from sklearn.ensemble import RandomForestClassifier



# load measurement data
measurement_df = pd.read_csv("data/raw/measurement_data.csv", parse_dates=["Measurement date"])
instrument_df = pd.read_csv("data/raw/instrument_data.csv", parse_dates=["Measurement date"])
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")

# Merge measurement and instrument data
merged_df = pd.merge(measurement_df, instrument_df, on=["Measurement date", "Station code"])

# prepare input/s
pollutant_df = pd.read_csv("data/raw/pollutant_data.csv")
input_list = [ "Station code: 205 | pollutant: SO2  | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00",
"Station code: 209 | pollutant: NO2  | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00",
"Station code: 223 | pollutant: O3   | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00",
"Station code: 224 | pollutant: CO  | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00",
"Station code: 226 | pollutant: PM10 | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00",
"Station code: 227 | pollutant: PM2.5 | Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00",
""]



def input_preparer(line, pollutant_data):
    parts = line.split("|")

    station_code = int(parts[0].split(":")[1].strip())
    pollutant = parts[1].split(":")[1].strip()
    start_date, end_date = parts[2].split("Period:")[1].strip().split(" - ")

    pollutant_code = pollutant_data[pollutant_data["Item name"]==pollutant]["Item code"].values[0]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    return station_code, pollutant_code, start_date, end_date


print(input_preparer("Station code: 209 | pollutant: NO2  | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00", pollutant_df))


raularmasserina@air-de-raul iberian2025-ds % /usr/local/bin/python3 /Users/raularmasserina/Desktop/HAC
KATHON/iberian2025-ds/models/model_task_3.py
(209, np.int64(2), Timestamp('2023-09-01 00:00:00'), Timestamp('2023-09-30 23:00:00'))
Traceback (most recent call last):
  File "/Users/raularmasserina/Desktop/HACKATHON/iberian2025-ds/models/model_task_3.py", line 93, in <module>
    df_filtered = data_filter(station_code, pollutant_code, start_date, end_date)
  File "/Users/raularmasserina/Desktop/HACKATHON/iberian2025-ds/models/model_task_3.py", line 69, in data_filter
    return ((merged_df[merged_df["Station code"] == StatCode]) & 
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/ops/common.py", line 76, in new_method
    return method(self, other)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/arraylike.py", line 70, in __and__
    return self._logical_method(other, operator.and_)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/frame.py", line 7913, in _arith_method
    new_data = self._dispatch_frame_op(other, op, axis=axis)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/frame.py", line 7956, in _dispatch_frame_op
    bm = self._mgr.operate_blockwise(
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/internals/managers.py", line 1511, in operate_blockwise
    return operate_blockwise(self, other, array_op)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/internals/ops.py", line 65, in operate_blockwise
    res_values = array_op(lvals, rvals)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 442, in logical_op
    res_values = op(lvalues, rvalues)
TypeError: unsupported operand type(s) for &: 'DatetimeArray' and 'DatetimeArray'