from social_xlstm.dataset.json_utils import VDLiveList, VDInfo


if __name__ == "__main__":
    vd_info = VDInfo.load_from_json("blob/lab/2025-03-18_00-49-00/VDList.json")
    print(vd_info)
    
    for vd in vd_info.VDList:
        print(f"VDID: {vd.VDID}, PositionLon: {vd.PositionLon}, PositionLat: {vd.PositionLat}")
    
    
    for vd in vd_info.VDList:
        print(f"VDID: {vd.VDID}, PositionLon: {vd.PositionLon}, PositionLat: {vd.PositionLat}")  
    
    vd_live_list = VDLiveList.load_from_json("blob/lab/2025-03-18_00-49-00/VDLiveList.json")
    print(vd_live_list)