from pathlib import Path
from enum import Enum
import xml.etree.ElementTree as ET
from typing import Union, List, Dict, Any
import json
import pathlib

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


class DatasetType(Enum):
    VDLIST = "VDList"
    VDLIVELIST = "VDLiveList"

class TrafficDatasetParser:
    @staticmethod
    def identify_dataset_type(input_path: str | Path) -> DatasetType:
        """
        Identifies the type of traffic dataset from the filename or XML content.
        
        Args:
            input_path (str | Path): Path to the XML file
            
        Returns:
            DatasetType: Enum indicating whether the dataset is VDList or VDLiveList
            
        Raises:
            ValueError: If the dataset type cannot be determined
        """
        input_path = Path(input_path)
        
        # First try to identify by filename
        filename = input_path.stem.lower()  # Get filename without extension
        if "vdlist" in filename:
            return DatasetType.VDLIST
        elif "vdlivelist" in filename:
            return DatasetType.VDLIVELIST
        
        # If filename doesn't help, check XML content
        try:
            tree = ET.parse(str(input_path))
            root = tree.getroot()
            
            if root.find('.//VDList') is not None:
                return DatasetType.VDLIST
            elif root.find('.//VDLiveList') is not None:
                return DatasetType.VDLIVELIST
            else:
                raise ValueError("Unknown dataset type: XML must contain either VDList or VDLiveList")
        except ET.ParseError:
            raise ValueError("Invalid XML file")

    @staticmethod
    def parse_dataset(input_path: str | Path, output_path: str | Path) -> None:
        """
        Parses the dataset and converts it to JSON based on its type.
        
        Args:
            input_path (str | Path): Path to the input XML file
            output_path (str | Path): Path to save the output JSON file
        """
        dataset_type = TrafficDatasetParser.identify_dataset_type(input_path)
        
        if dataset_type == DatasetType.VDLIST:
            VDList_xml_to_Json(input_path, output_path)
        else:
            VDLiveList_xml_to_Json(input_path, output_path)


def raw_xml_to_json(
    input_file_path:   str | Path,
    output_file_path : str | Path,
    dataset_type: DatasetType = None,
    unknown_ignore: bool = False,
):
    """
    Converts XML file to JSON format based on the dataset type.
    
    Args:
        input_file_path (str | Path): Path to the input XML file to convert
        output_file_path (str | Path): Path where the output JSON file will be saved
        dataset_type (DatasetType, optional): Type of dataset (VDList or VDLiveList). If None, will attempt to identify from file
    
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If dataset type is unknown or cannot be determined
    """
    
    input_file_path = Path(input_file_path)
    output_file_path = Path(output_file_path)
    
    # Check if input file exists
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Check if output folder exists, if not create it
    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset_type is None
    if dataset_type is None:
        # Identify the dataset type
        try:
            dataset_type = TrafficDatasetParser.identify_dataset_type(input_file_path)
        except ValueError as e:
            if unknown_ignore:
                print(f"Warning: {e}. Ignoring.")
                return
            else:
                raise ValueError(f"Could not identify dataset type: {e}")
        print(f"Identified dataset type: {dataset_type}")
    
    
    # Parse the dataset and convert to JSON
    if dataset_type == DatasetType.VDLIST:
        VDList_xml_to_Json(input_file_path, output_file_path)
    elif dataset_type == DatasetType.VDLIVELIST:
        VDLiveList_xml_to_Json(input_file_path, output_file_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")