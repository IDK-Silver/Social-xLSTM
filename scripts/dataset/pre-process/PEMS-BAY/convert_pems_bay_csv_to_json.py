import pandas as pd
import json

# 讀取 CSV 檔案
csv_path = '/home/GP/repo/Social-xLSTM/blob/dataset/pre-processed/PEMS-BAY/PEMS-BAY-META.csv'  # 替換成你的路徑
df = pd.read_csv(csv_path)

# 只保留需要的欄位
selected_columns = ['sensor_id', 'Lanes', 'Length', 'Latitude', 'Longitude', 'Dir']
df_filtered = df[selected_columns]

# 以 sensor_id 為 key，轉為字典格式
sensor_dict = df_filtered.set_index('sensor_id').to_dict(orient='index')

# 儲存為 JSON 檔案
json_path = 'pemsbay_sensors.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(sensor_dict, f, ensure_ascii=False, indent=2)

print(f"成功儲存 JSON 檔案至：{json_path}")
