import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Union
from pathlib import Path

@dataclass
class UpdateInfo:
    UpdateTime: str
    UpdateInterval: int

@dataclass
class DetectionLink:
    LinkID: str
    RoadDirection: str

@dataclass
class VD:
    VDID: str
    PositionLon: float
    PositionLat: float
    RoadID: str
    RoadName: str
    LaneNum: int
    DetectionLinks: List[DetectionLink]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VD':
        detection_links = [DetectionLink(**link) for link in data.get("DetectionLinks", [])]
        return cls(
            VDID=data["VDID"],
            PositionLon=data["PositionLon"],
            PositionLat=data["PositionLat"],
            RoadID=data["RoadID"],
            RoadName=data["RoadName"],
            LaneNum=data["LaneNum"],
            DetectionLinks=detection_links
        )
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["DetectionLinks"] = [asdict(link) for link in self.DetectionLinks]
        return data

@dataclass
class VDInfo:
    UpdateInfo: UpdateInfo
    VDList: List[VD]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VDInfo':
        update_info_data = data.get("UpdateInfo", {})
        update_info = UpdateInfo(**update_info_data)
        
        vd_list_data = data.get("VDList", [])
        vd_list = [VD.from_dict(vd_item) for vd_item in vd_list_data]
        
        return cls(UpdateInfo=update_info, VDList=vd_list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "UpdateInfo": asdict(self.UpdateInfo),
            "VDList": [vd.to_dict() for vd in self.VDList]
        }

    @classmethod
    def load_from_json(cls, file_path: Path) -> 'VDInfo':
        """Loads VDInfo data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_json(self, file_path: Path) -> None:
        """Saves VDInfo data to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

@dataclass
class VehicleData:
    VehicleType: str
    Volume: int

@dataclass
class LaneData:
    LaneID: int
    LaneType: int
    Speed: float
    Occupancy: float
    Vehicles: List[VehicleData]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LaneData':
        vehicles = [VehicleData(**vehicle) for vehicle in data.get("Vehicles", [])]
        return cls(
            LaneID=data["LaneID"],
            LaneType=data["LaneType"],
            Speed=data["Speed"],
            Occupancy=data["Occupancy"],
            Vehicles=vehicles
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["Vehicles"] = [asdict(vehicle) for vehicle in self.Vehicles]
        return data

@dataclass
class LinkFlowData:
    LinkID: str
    Lanes: List[LaneData]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinkFlowData':
        lanes = [LaneData.from_dict(lane) for lane in data.get("Lanes", [])]
        return cls(
            LinkID=data["LinkID"],
            Lanes=lanes
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["Lanes"] = [lane.to_dict() for lane in self.Lanes]
        return data

@dataclass
class VDLiveDetail:
    VDID: str
    DataCollectTime: str
    UpdateInterval: int
    AuthorityCode: str
    LinkFlows: List[LinkFlowData]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VDLiveDetail':
        link_flows = [LinkFlowData.from_dict(flow) for flow in data.get("LinkFlows", [])]
        return cls(
            VDID=data["VDID"],
            DataCollectTime=data["DataCollectTime"],
            UpdateInterval=data["UpdateInterval"],
            AuthorityCode=data["AuthorityCode"],
            LinkFlows=link_flows
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["LinkFlows"] = [flow.to_dict() for flow in self.LinkFlows]
        return data

@dataclass
class VDLiveList:
    LiveTrafficData: List[VDLiveDetail]

    @classmethod
    def from_list_of_dicts(cls, data: List[Dict[str, Any]]) -> 'VDLiveList':
        live_traffic_data = [VDLiveDetail.from_dict(item) for item in data]
        return cls(LiveTrafficData=live_traffic_data)

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.LiveTrafficData]

    @classmethod
    def load_from_json(cls, file_path: Path) -> 'VDLiveList':
        """Loads a list of VDLiveDetail data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON root is not a list for VDLiveList")
        return cls.from_list_of_dicts(data)

    def save_to_json(self, file_path: Path) -> None:
        """Saves the list of VDLiveDetail data to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_list_of_dicts(), f, ensure_ascii=False, indent=4)

