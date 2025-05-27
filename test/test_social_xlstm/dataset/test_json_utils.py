import pytest
import json
from pathlib import Path
from social_xlstm.dataset.json_utils import VDInfo, VDLiveList, UpdateInfo, VD, DetectionLink, VDLiveDetail, LinkFlowData, LaneData, VehicleData

# Fixture to create a temporary directory for test files
@pytest.fixture
def temp_json_dir(tmp_path: Path) -> Path:
    return tmp_path

# --- Tests for VDInfo ---

@pytest.fixture
def sample_vd_info_data() -> dict:
    return {
        "UpdateInfo": {
            "UpdateTime": "2025-03-17T02:20:33+08:00",
            "UpdateInterval": 86400
        },
        "VDList": [
            {
                "VDID": "VD-11-0020-002-01",
                "PositionLon": 121.459501,
                "PositionLat": 25.149709,
                "RoadID": "300020",
                "RoadName": "臺2線",
                "LaneNum": 3,
                "DetectionLinks": [
                    {"LinkID": "3000200000176F", "RoadDirection": "E"},
                    {"LinkID": "3000200100155F", "RoadDirection": "W"}
                ]
            },
            {
                "VDID": "VD-11-0020-008-01",
                "PositionLon": 121.44674,
                "PositionLat": 25.19163,
                "RoadID": "300020",
                "RoadName": "臺2線",
                "LaneNum": 1, # Changed LaneNum for variety
                "DetectionLinks": [
                    {"LinkID": "3000200100771F", "RoadDirection": "W"}
                ]
            }
        ]
    }

def test_vd_info_load_from_json(temp_json_dir: Path, sample_vd_info_data: dict):
    file_path = temp_json_dir / "vd_info.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_vd_info_data, f, ensure_ascii=False, indent=4)

    vd_info = VDInfo.load_from_json(file_path)

    assert isinstance(vd_info, VDInfo)
    assert vd_info.UpdateInfo.UpdateTime == sample_vd_info_data["UpdateInfo"]["UpdateTime"]
    assert vd_info.UpdateInfo.UpdateInterval == sample_vd_info_data["UpdateInfo"]["UpdateInterval"]
    assert len(vd_info.VDList) == len(sample_vd_info_data["VDList"])
    assert vd_info.VDList[0].VDID == sample_vd_info_data["VDList"][0]["VDID"]
    assert vd_info.VDList[0].LaneNum == sample_vd_info_data["VDList"][0]["LaneNum"]
    assert len(vd_info.VDList[0].DetectionLinks) == len(sample_vd_info_data["VDList"][0]["DetectionLinks"])
    assert vd_info.VDList[0].DetectionLinks[0].LinkID == sample_vd_info_data["VDList"][0]["DetectionLinks"][0]["LinkID"]

def test_vd_info_save_to_json(temp_json_dir: Path, sample_vd_info_data: dict):
    file_path = temp_json_dir / "vd_info_output.json"
    
    # Create VDInfo object from sample data
    update_info = UpdateInfo(**sample_vd_info_data["UpdateInfo"])
    vd_list = [VD.from_dict(vd_data) for vd_data in sample_vd_info_data["VDList"]]
    vd_info_to_save = VDInfo(UpdateInfo=update_info, VDList=vd_list)

    vd_info_to_save.save_to_json(file_path)

    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    # Basic check, more thorough comparison can be done by comparing dicts
    assert saved_data["UpdateInfo"]["UpdateTime"] == sample_vd_info_data["UpdateInfo"]["UpdateTime"]
    assert len(saved_data["VDList"]) == len(sample_vd_info_data["VDList"])
    assert saved_data["VDList"][0]["VDID"] == sample_vd_info_data["VDList"][0]["VDID"]

def test_vd_info_load_save_consistency(temp_json_dir: Path, sample_vd_info_data: dict):
    original_file_path = temp_json_dir / "original_vd_info.json"
    saved_file_path = temp_json_dir / "resaved_vd_info.json"

    with open(original_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_vd_info_data, f, ensure_ascii=False, indent=4)

    # Load original
    vd_info_loaded = VDInfo.load_from_json(original_file_path)
    # Save it
    vd_info_loaded.save_to_json(saved_file_path)
    # Load the saved one
    vd_info_resaved = VDInfo.load_from_json(saved_file_path)

    assert vd_info_loaded == vd_info_resaved # Dataclasses implement __eq__

def test_vd_info_load_file_not_found(temp_json_dir: Path):
    non_existent_file = temp_json_dir / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        VDInfo.load_from_json(non_existent_file)

# --- Tests for VDLiveList ---

@pytest.fixture
def sample_vd_live_list_data() -> list:
    return [
        {
            "VDID": "VD-11-0020-002-002",
            "DataCollectTime": "2025-03-18T00:46:11+08:00",
            "UpdateInterval": 60,
            "AuthorityCode": "THB",
            "LinkFlows": [
                {
                    "LinkID": "3000200000176F",
                    "Lanes": [
                        {
                            "LaneID": 0, "LaneType": 1, "Speed": 59.0, "Occupancy": 1.0,
                            "Vehicles": [{"VehicleType": "S", "Volume": 1}, {"VehicleType": "L", "Volume": 0}]
                        },
                        {
                            "LaneID": 1, "LaneType": 1, "Speed": 49.0, "Occupancy": 7.3,
                            "Vehicles": [{"VehicleType": "S", "Volume": 6}]
                        }
                    ]
                }
            ]
        },
        {
            "VDID": "VD-11-0020-008-001",
            "DataCollectTime": "2025-03-18T00:47:11+08:00", # Different time
            "UpdateInterval": 60,
            "AuthorityCode": "THB",
            "LinkFlows": [] # Empty LinkFlows
        }
    ]

def test_vd_live_list_load_from_json(temp_json_dir: Path, sample_vd_live_list_data: list):
    file_path = temp_json_dir / "vd_live_list.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_vd_live_list_data, f, ensure_ascii=False, indent=4)

    vd_live_list = VDLiveList.load_from_json(file_path)

    assert isinstance(vd_live_list, VDLiveList)
    assert len(vd_live_list.LiveTrafficData) == len(sample_vd_live_list_data)
    
    first_item_loaded = vd_live_list.LiveTrafficData[0]
    first_item_sample = sample_vd_live_list_data[0]
    assert first_item_loaded.VDID == first_item_sample["VDID"]
    assert first_item_loaded.DataCollectTime == first_item_sample["DataCollectTime"]
    assert len(first_item_loaded.LinkFlows) == len(first_item_sample["LinkFlows"])
    if first_item_sample["LinkFlows"]: # Check nested if exists
        assert first_item_loaded.LinkFlows[0].LinkID == first_item_sample["LinkFlows"][0]["LinkID"]
        assert len(first_item_loaded.LinkFlows[0].Lanes) == len(first_item_sample["LinkFlows"][0]["Lanes"])
        assert first_item_loaded.LinkFlows[0].Lanes[0].Speed == first_item_sample["LinkFlows"][0]["Lanes"][0]["Speed"]
        assert len(first_item_loaded.LinkFlows[0].Lanes[0].Vehicles) == len(first_item_sample["LinkFlows"][0]["Lanes"][0]["Vehicles"])
        assert first_item_loaded.LinkFlows[0].Lanes[0].Vehicles[0].Volume == first_item_sample["LinkFlows"][0]["Lanes"][0]["Vehicles"][0]["Volume"]

def test_vd_live_list_save_to_json(temp_json_dir: Path, sample_vd_live_list_data: list):
    file_path = temp_json_dir / "vd_live_list_output.json"
    
    vd_live_list_to_save = VDLiveList.from_list_of_dicts(sample_vd_live_list_data)
    vd_live_list_to_save.save_to_json(file_path)

    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert len(saved_data) == len(sample_vd_live_list_data)
    assert saved_data[0]["VDID"] == sample_vd_live_list_data[0]["VDID"]
    if sample_vd_live_list_data[0]["LinkFlows"]:
         assert saved_data[0]["LinkFlows"][0]["LinkID"] == sample_vd_live_list_data[0]["LinkFlows"][0]["LinkID"]


def test_vd_live_list_load_save_consistency(temp_json_dir: Path, sample_vd_live_list_data: list):
    original_file_path = temp_json_dir / "original_vd_live_list.json"
    saved_file_path = temp_json_dir / "resaved_vd_live_list.json"

    with open(original_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_vd_live_list_data, f, ensure_ascii=False, indent=4)

    vd_live_loaded = VDLiveList.load_from_json(original_file_path)
    vd_live_loaded.save_to_json(saved_file_path)
    vd_live_resaved = VDLiveList.load_from_json(saved_file_path)

    assert vd_live_loaded == vd_live_resaved # Dataclasses implement __eq__

def test_vd_live_list_load_file_not_found(temp_json_dir: Path):
    non_existent_file = temp_json_dir / "non_existent_live.json"
    with pytest.raises(FileNotFoundError):
        VDLiveList.load_from_json(non_existent_file)

def test_vd_live_list_load_invalid_json_root(temp_json_dir: Path):
    file_path = temp_json_dir / "invalid_root_live.json"
    # VDLiveList expects a list at the root, provide a dict instead
    invalid_data = {"key": "value"} 
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(invalid_data, f)
    
    with pytest.raises(ValueError, match="JSON root is not a list for VDLiveList"):
        VDLiveList.load_from_json(file_path)

def test_vd_live_list_empty_json_array(temp_json_dir: Path):
    file_path = temp_json_dir / "empty_live_list.json"
    empty_list_data = []
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(empty_list_data, f)

    vd_live_list = VDLiveList.load_from_json(file_path)
    assert isinstance(vd_live_list, VDLiveList)
    assert len(vd_live_list.LiveTrafficData) == 0

def test_vd_live_list_with_empty_linkflows(temp_json_dir: Path):
    # Using the provided VDLiveList.json content
    file_path = temp_json_dir / "vd_live_list_empty_flows.json"
    data_with_empty_flows = json.loads("""
    [
        {
            "VDID": "VD-11-0020-002-002",
            "DataCollectTime": "2025-03-18T00:46:11+08:00",
            "UpdateInterval": 60,
            "AuthorityCode": "THB",
            "LinkFlows": []
        },
        {
            "VDID": "VD-11-0020-008-001",
            "DataCollectTime": "2025-03-18T00:46:11+08:00",
            "UpdateInterval": 60,
            "AuthorityCode": "THB",
            "LinkFlows": []
        }
    ]
    """)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_with_empty_flows, f, ensure_ascii=False, indent=4)

    vd_live_list = VDLiveList.load_from_json(file_path)
    assert isinstance(vd_live_list, VDLiveList)
    assert len(vd_live_list.LiveTrafficData) == 2
    assert vd_live_list.LiveTrafficData[0].VDID == "VD-11-0020-002-002"
    assert vd_live_list.LiveTrafficData[0].LinkFlows == []
    assert vd_live_list.LiveTrafficData[1].VDID == "VD-11-0020-008-001"
    assert vd_live_list.LiveTrafficData[1].LinkFlows == []
