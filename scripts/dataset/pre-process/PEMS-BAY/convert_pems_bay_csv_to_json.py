import pandas as pd
import json

# 讀取 CSV 檔案
csv_path = '/home/GP/repo/Social-xLSTM/blob/dataset/pre-processed/PEMS-BAY/PEMS-BAY-META.csv'
df = pd.read_csv(csv_path)

# 篩選所需欄位
fields = ['sensor_id', 'Lanes', 'Length', 'Latitude', 'Longitude', 'Dir']
filtered_df = df[fields]

# 轉成指定格式的 JSON list
sensor_list = []
for _, row in filtered_df.iterrows():
    sensor = {
        "sensor_id": int(row["sensor_id"]),
        "Lanes": int(row["Lanes"]),
        "Length": float(row["Length"]),
        "Latitude": float(row["Latitude"]),
        "Longitude": float(row["Longitude"]),
        "Dir": row["Dir"]
    }
    sensor_list.append(sensor)

# 寫入 JSON 檔
output_path = 'pemsbay_sensors.json'
with open(output_path, 'w') as f:
    json.dump(sensor_list, f, indent=2)

print(f"轉換完成，已儲存為 {output_path}")
