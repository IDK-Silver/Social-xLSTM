# ? 這測試怪怪的，為什麼要轉成 csv
import pandas as pd

# 讀取 H5 檔
df = pd.read_hdf("metr-la.h5")

# 存成 CSV
df.to_csv("metr-la.csv")

print("轉換完成！輸出 metr-la.csv")
