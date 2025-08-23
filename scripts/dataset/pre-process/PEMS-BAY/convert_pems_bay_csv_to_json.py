import pandas as pd
import json

# 路徑
meta_path = "PEMS-BAY-META.csv"
data_path = "活頁簿11.csv"

# 讀取 Meta 資料
meta_fields = ['sensor_id', 'Lanes', 'Length', 'Latitude', 'Longitude', 'Dir']
meta_df = pd.read_csv(meta_path)[meta_fields]
meta_df.set_index("sensor_id", inplace=True)  # 方便查詢

# 讀取交通數據
data_df = pd.read_csv(data_path)

# 第一欄是時間
data_df = data_df.rename(columns={"Unnamed: 0": "date"})

# 建立輸出結構
output = {}

for _, row in data_df.iterrows():
    date = row["date"]
    sensors_info = []
    
    for sensor_id in row.index[1:]:  # 跳過 date 欄
        speed = row[sensor_id]
        sensor_id_int = int(sensor_id)

        # 取 meta 資料
        if sensor_id_int in meta_df.index:
            meta = meta_df.loc[sensor_id_int].to_dict()
        else:
            meta = {"Lanes": None, "Length": None, "Latitude": None, "Longitude": None, "Dir": None}

        # 整合
        sensor_entry = {
            "sensor_id": sensor_id_int,
            "speed": None if pd.isna(speed) else float(speed),
            **meta
        }
        sensors_info.append(sensor_entry)
    
    output[date] = sensors_info

# 輸出成 JSON
with open("pemsbay_timeseries.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("轉換完成！輸出檔案：pemsbay_timeseries.json")
