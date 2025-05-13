import argparse
import json
import matplotlib.pyplot as plt
from convert_coords import mercator_projection
from social_xlstm.dataset.json_utils import VDLiveList, VDInfo
#python graph.py --VDListJson /home/GP/repo/Social-xLSTM/blob/lab/2025-03-18_00-49-00/VDList.json


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--VDListJson", required=True, help="Json檔絕對位置")

    return parser.parse_args()


def plot_vd_coordinates(json_path, lat_origin=23.9150, lon_origin=120.6846, radius=6378137):
    """
    根據 VDList.json 檔案，繪製所有 VDID 設備的墨卡托投影座標圖
    """
    #check if json_path is none
    if json_path is None:
        raise ValueError("json_path cannot be None")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"無法讀取 JSON 檔案：{e}")
        return

    x_coords = []
    y_coords = []

    for vd in data.get("VDList", []):
        try:
            lat = vd["PositionLat"]
            lon = vd["PositionLon"]
            x, y = mercator_projection(lat, lon, lat_origin, lon_origin, radius)
            x_coords.append(x)
            y_coords.append(y)
        except Exception as e:
            print(f"轉換座標失敗(VDID={vd.get('VDID')}): {e}")

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
    plt.savefig("/home/GP/repo/Social-xLSTM/src/social_xlstm/utils/vdid_map.png", dpi=1000)

def main():
    args = parse_arguments()
    plot_vd_coordinates(args.VDListJson)

if __name__ == "__main__":
    main()