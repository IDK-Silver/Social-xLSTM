import argparse
import math
from typing import Union, Tuple

#python convert_coords.py --latitude 25.0330 --longitude 121.5654

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--latitude", type=float, required=True, help="目標緯度")
    parser.add_argument("--longitude", type=float, required=True, help="目標經度")
    parser.add_argument("--lat_origin",type=float,default=23.9150,help="Taiwan Nantou 緯度")
    parser.add_argument("--lon_origin",type=float,default=120.6846,help="Taiwan Nantou 經度")

    return parser.parse_args()

def mercator_projection(latitude: Union[float, int],longitude: Union[float, int],lat_origin: Union[float, int],lon_origin: Union[float, int],radius: float = 6378137) -> Tuple[float, float]:
    """
    將經緯度轉為以自訂原點為基準的墨卡托投影平面座標（相對位置）
    :return: (x, y) 相對原點的平面座標（公尺）
    """

    if latitude is None:
        raise ValueError("latitude cannot be None")
    if longitude is None:
        raise ValueError("longitude cannot be None")

    # 轉為弧度
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    lat0_rad = math.radians(lat_origin)
    lon0_rad = math.radians(lon_origin)

    # 墨卡托投影原點和目標點的 x, y
    x = radius * (lon_rad - lon0_rad)
    y = radius * (math.log(math.tan(math.pi / 4 + lat_rad / 2)) -
                  math.log(math.tan(math.pi / 4 + lat0_rad / 2)))

    return x, y

def main():
    args = parse_arguments()

    # 計算相對座標 # lat_origin, lon_origin 可自行設置
    x, y = mercator_projection(
        latitude=args.latitude,
        longitude=args.longitude,
        lat_origin=args.lat_origin,
        lon_origin=args.lon_origin
    )

    print(f"相對原點的 XY 座標為: x = {x:.2f} m, y = {y:.2f} m")

if __name__ == '__main__':
    main()