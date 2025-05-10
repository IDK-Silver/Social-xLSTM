import argparse
import os
import xml.etree.ElementTree as ET
import json
import pathlib
from typing import Union

#python XML_to_Json.py --xml_file_path /home/GP/repo/Social-xLSTM/blob/lab/zip/2025-04-09_00:00:00_to_2025-04-10_00:00:00/2025-04-09_00-01-00/VDList.xml --output_json_path /home/GP/repo/Social-xLSTM/blob/lab/Json/2025-04-09_00:00:00_to_2025-04-10_00:00:00/2025-04-09_00-01-00/VDList
#python XML_to_Json.py --xml_file_path /home/GP/repo/Social-xLSTM/blob/lab/zip/2025-04-09_00:00:00_to_2025-04-10_00:00:00/2025-04-09_00-01-00/VDLiveList.xml --output_json_path /home/GP/repo/Social-xLSTM/blob/lab/Json/2025-04-09_00:00:00_to_2025-04-10_00:00:00/2025-04-09_00-01-00/VDLiveList

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """

    # 建立命令列參數解析器
    parser = argparse.ArgumentParser(description="Process an XML file")
    #xml檔存放的絕對路徑
    #ex : --xml_file_path /home/GP/repo/Social-xLSTM/blob/lab/zip/2025-04-09_00:00:00_to_2025-04-10_00:00:00/2025-04-09_00-01-00/VDList.xml  or VDLiveList.xml
    parser.add_argument("--xml_file_path", type=str, required=True, help="XML 檔案絕對路徑")
    #輸出json檔的存放位址
    #ex : --output_json_path /home/GP/repo/Social-xLSTM/blob/lab/Json/2025-04-09_00:00:00_to_2025-04-10_00:00:00/2025-04-09_00-01-00/VDList or VDLiveList
    parser.add_argument("--output_json_path", type=str, required=True, help="輸出檔案到資料夾的絕對路徑")

    return parser.parse_args()


def VDList_xml_to_Json(xml_file_path :Union[str,pathlib.Path],output_json_path:Union[str,pathlib.Path])->None:
    
    #check if xml_file_path is None
    if xml_file_path is None:
        raise ValueError("Input file path cannot be None")
    
    #check if output_json_path is None
    if output_json_path is None:
        raise ValueError("Output file path cannot be None")
    
    # Convert input path to pathlib.Path object
    xml_file_path = pathlib.Path(xml_file_path)

    # Validate input file exists
    if not xml_file_path.exists():
        raise FileNotFoundError(f"Input file not found : {xml_file_path}")
    
    # Convert output path to pathlib.Path object
    output_json_path = pathlib.Path(output_json_path)

    # Create output directory if it doesn't exist
    output_json_path.parent.mkdir(parents=True,exist_ok=True)

    try:

        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # XML 命名空間 研究看看
        namespace = {"ns": "http://traffic.transportdata.tw/standard/traffic/schema/"}

        # 解析 UpdateTime 和 UpdateInterval
        update_time = root.find("ns:UpdateTime", namespace)
        update_interval = root.find("ns:UpdateInterval", namespace)

        update_info = {
            "UpdateTime": update_time.text if update_time is not None else "",
            "UpdateInterval": int(update_interval.text) if update_interval is not None else 0
        }

        # 解析 VDs 資料
        vd_list = []
        for vd in root.findall("ns:VDs/ns:VD", namespace):
            try:
                vd_data = {
                    "VDID": vd.find("ns:VDID", namespace).text,
                    "PositionLon": float(vd.find("ns:PositionLon", namespace).text),
                    "PositionLat": float(vd.find("ns:PositionLat", namespace).text),
                    "RoadID": vd.find("ns:RoadID", namespace).text,
                    "RoadName": vd.find("ns:RoadName", namespace).text,
                    "LaneNum": int(vd.find("ns:DetectionLinks/ns:DetectionLink/ns:LaneNum", namespace).text),
                    "DetectionLinks": []
                }

                for link in vd.findall("ns:DetectionLinks/ns:DetectionLink", namespace):
                    link_data = {
                        "LinkID": link.find("ns:LinkID", namespace).text,
                        "RoadDirection": link.find("ns:RoadDirection", namespace).text
                    }
                    vd_data["DetectionLinks"].append(link_data)

                vd_list.append(vd_data)
            except AttributeError as e:
                print(f"跳過 VD 元素，因為有缺少欄位：{e}")
                continue
        # 合併 Update 資訊與 VD 清單
        dataset = {
            "UpdateInfo": update_info,
            "VDList": vd_list
        }

        # 自動產生檔名 VDList.json 放入指定資料夾
        # output_file_path = output_json_path / "VDList.json"
        output_file_path = output_json_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
            
        print(f"轉換完成！JSON 檔案已儲存為 {output_json_path}")

    except ET.ParseError as e:
        raise RuntimeError(f"XML 解析錯誤：{e}")
    except Exception as e:
        raise RuntimeError(f"處理 VDList 時發生錯誤：{e}")

def VDLiveList_xml_to_Json(xml_file_path :Union[str,pathlib.Path],output_json_path:Union[str,pathlib.Path])->None:
    
    #check if xml_file_path is None
    if xml_file_path is None:
        raise ValueError("Input file path cannot be None")
    
    #check if output_json_path is None
    if output_json_path is None:
        raise ValueError("Output file path cannot be None")
    
    # Convert input path to pathlib.Path object
    xml_file_path = pathlib.Path(xml_file_path)

    # Validate input file exists
    if not xml_file_path.exists():
        raise FileNotFoundError(f"Input file not found : {xml_file_path}")
    
    # Convert output path to pathlib.Path object
    output_json_path = pathlib.Path(output_json_path)

    # Create output directory if it doesn't exist
    output_json_path.parent.mkdir(parents=True,exist_ok=True)

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 定義命名空間
        namespace = {"ns": "http://traffic.transportdata.tw/standard/traffic/schema/"}

        # 解析 XML 並轉換為 JSON 結構
        data_list = []

        for vd_live in root.find("ns:VDLives", namespace).findall("ns:VDLive", namespace):
            try:
                vd_data = {
                    "VDID": vd_live.find("ns:VDID", namespace).text,
                    "DataCollectTime": vd_live.find("ns:DataCollectTime", namespace).text,
                    "UpdateInterval": int(root.find("ns:UpdateInterval", namespace).text),
                    "AuthorityCode": root.find("ns:AuthorityCode", namespace).text,
                    "LinkFlows": []
                }

                for link_flow in vd_live.findall("ns:LinkFlows/ns:LinkFlow", namespace):
                    link_data = {
                        "LinkID": link_flow.find("ns:LinkID", namespace).text,
                        "Lanes": []
                    }

                    for lane in link_flow.findall("ns:Lanes/ns:Lane", namespace):
                        lane_data = {
                            "LaneID": int(lane.find("ns:LaneID", namespace).text),
                            "LaneType": int(lane.find("ns:LaneType", namespace).text),
                            "Speed": float(lane.find("ns:Speed", namespace).text),
                            "Occupancy": float(lane.find("ns:Occupancy", namespace).text),
                            "Vehicles": []
                        }

                        for vehicle in lane.findall("ns:Vehicles/ns:Vehicle", namespace):
                            vehicle_data = {
                                "VehicleType": vehicle.find("ns:VehicleType", namespace).text,
                                "Volume": int(vehicle.find("ns:Volume", namespace).text)
                            }
                            lane_data["Vehicles"].append(vehicle_data)

                        link_data["Lanes"].append(lane_data)

                    vd_data["LinkFlows"].append(link_data)

                data_list.append(vd_data)
            except AttributeError as e:
                print(f"跳過 VDLive 元素，因為有缺少欄位：{e}")
                continue
        
        # 自動產生檔名 VDList.json 放入指定資料夾
        # output_file_path = output_json_path / "VDLiveList.json"
        output_file_path = output_json_path

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

        print(f"轉換完成！JSON 檔案已儲存為 {output_json_path}")
    except ET.ParseError as e:
        raise RuntimeError(f"XML 解析錯誤：{e}")
    except Exception as e:
        raise RuntimeError(f"處理 VDLiveList 時發生錯誤：{e}")

def main():
    args=parse_arguments()

    xml_file_path=pathlib.Path(args.xml_file_path)
    output_json_path=pathlib.Path(args.output_json_path)

    #取得xml檔名 (ex : VDList.xml or VDLiveList.xml)
    xml_filename=xml_file_path.name
    #print(xml_filename)

    try:
        # 判斷 xml_filename 是否為 VDList.xml
        if xml_filename == "VDList.xml":
            VDList_xml_to_Json(xml_file_path, output_json_path)
        # 判斷 xml_filename 是否為 VDLiveList.xml
        elif xml_filename == "VDLiveList.xml":
            VDLiveList_xml_to_Json(xml_file_path, output_json_path)
        else:
            print(f"檔案 {xml_filename} 不是 VDList.xml or VDLiveList.xml，跳過轉換。")
    except Exception as e:
        print(f"執行轉換過程中發生錯誤：{e}")

    
if __name__ == "__main__":
    main()