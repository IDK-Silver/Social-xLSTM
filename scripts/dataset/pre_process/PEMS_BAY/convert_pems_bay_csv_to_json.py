import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

# 路徑
meta_path = "/home/GP/repo/Social-xLSTM/blob/dataset/raw/PEMS-BAY/PEMS-BAY-META.csv"
data_path = "/home/GP/repo/Social-xLSTM/blob/dataset/raw/PEMS-BAY/PEMS-BAY.csv"
output_path = "pems_bay_features_test.h5"

# 讀取 meta
meta_fields = ['sensor_id', 'Lanes', 'Length', 'Latitude', 'Longitude', 'Dir']
meta_df = pd.read_csv(meta_path)[meta_fields]
meta_df.set_index("sensor_id", inplace=True)

# 讀取交通數據
data_df = pd.read_csv(data_path)
data_df = data_df.rename(columns={"Unnamed: 0": "date"})

# 只取前三天 (假設每 5 分鐘一筆 → 一天 288 筆)
rows_per_day = 288
subset_df = data_df.iloc[:rows_per_day * 3]

# 感測器欄位 (保持字串，避免 KeyError)
sensor_ids = [c for c in subset_df.columns if c != "date"]

# 初始化 feature array: (time, sensor, features)
num_time = len(subset_df)
num_nodes = len(sensor_ids)
num_features = 6  # speed + meta 五個欄位
features = np.full((num_time, num_nodes, num_features), np.nan)

# Dir 映射
dir_map = {"N": 0, "S": 1, "E": 2, "W": 3}

# 填資料
print("轉換中...")
for j, sid in enumerate(tqdm(sensor_ids, desc="Sensors")):
    sid_int = int(sid)  # 對 meta 用整數
    # 取 meta
    if sid_int in meta_df.index:
        meta = meta_df.loc[sid_int].to_dict()
    else:
        meta = {"Lanes": np.nan, "Length": np.nan, "Latitude": np.nan, "Longitude": np.nan, "Dir": None}

    for i, row in enumerate(subset_df[sid]):  # sid 是字串
        features[i, j, 0] = row if not pd.isna(row) else np.nan
        features[i, j, 1] = meta.get("Lanes", np.nan)
        features[i, j, 2] = meta.get("Length", np.nan)
        features[i, j, 3] = meta.get("Latitude", np.nan)
        features[i, j, 4] = meta.get("Longitude", np.nan)
        features[i, j, 5] = dir_map.get(meta.get("Dir", None), np.nan)

# 存成 H5
with h5py.File(output_path, "w") as f:
    f.create_dataset("features", data=features)
    f.create_dataset("dates", data=subset_df["date"].astype("S"))  # 存字串日期
    f.create_dataset("sensors", data=np.array(sensor_ids, dtype="S"))

print(f"✅ 轉換完成！輸出檔案：{output_path}")
