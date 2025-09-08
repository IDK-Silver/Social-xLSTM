import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from .convert_coords import mercator_projection
from social_xlstm.dataset.utils.json_utils import VDLiveList, VDInfo
# python graph.py --VDListJson /path/to/VDList.json




def plot_vd_coordinates(json_path, lat_origin=23.9150, lon_origin=120.6846, radius=6378137, output_path: str | None = None):
    """
    根據 VDList.json 檔案，繪製所有 VDID 設備的墨卡托投影座標圖
    """
    #check if json_path is none
    if json_path is None:
        raise ValueError("json_path cannot be None")
    
    # try:
    #     with open(json_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    # except Exception as e:
    #     print(f"無法讀取 JSON 檔案：{e}")
    #     return
    
    vd_info=VDInfo.load_from_json(json_path)

    x_coords = []
    y_coords = []

    for vd in vd_info.VDList:
        try:
            lat = vd.PositionLat
            lon = vd.PositionLon
            x, y = mercator_projection(lat, lon, lat_origin, lon_origin, radius)
            x_coords.append(x)
            y_coords.append(y)
        except Exception as e:
            print(f"轉換座標失敗(VDID={vd.VDID}): {e}")

    if not x_coords:
        print("沒有成功取得任何座標，無法繪圖")
        return

    # 繪製圖形
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, s=10, color='blue', alpha=0.6)
    plt.title("VDID")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = Path("blob/plots/vdid_map.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=1000)
    return str(output_path)
