import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """計算兩點間的地球表面距離（km）"""
    R = 6371  # 地球半徑 km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def find_nearest_sensors(csv_path, center_id, n=10):
    # 讀取 CSV
    df = pd.read_csv(csv_path)
    
    # 找出中心點
    if center_id not in df["sensor_id"].values:
        raise ValueError(f"sensor_id {center_id} not found in CSV.")
    
    center = df[df["sensor_id"] == center_id].iloc[0]
    
    # 計算距離
    df["distance"] = df.apply(
        lambda row: haversine(center.latitude, center.longitude, row.latitude, row.longitude), axis=1
    )
    
    # 排序並取最近 n 個（排除自己）
    nearest = df[df["sensor_id"] != center_id].sort_values("distance").head(n)
    return nearest

# 使用範例
csv_file = "/home/GP/repo/Social-xLSTM/blob/dataset/raw/METR-LA/metr-la_sensor_locations.csv"  # 你的 CSV 檔案
center_sensor = 773869
nearest_10 = find_nearest_sensors(csv_file, center_sensor, n=10)
print(nearest_10)

