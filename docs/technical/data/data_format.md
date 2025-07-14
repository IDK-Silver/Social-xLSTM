# 交通資料格式說明

## 概述

本專案使用台灣交通部提供的車輛偵測器 (VD, Vehicle Detector) 資料，包含即時交通流量、速度、佔有率等資訊。

## 資料來源結構

### 原始資料格式
```
ZIP 壓縮檔 → XML 檔案 → JSON 格式 → HDF5 格式
```

### 時間戳記目錄結構
```
blob/dataset/pre-processed/unzip_to_json/
├── 2025-03-18_00-49-00/
│   ├── VDList.json          # VD 元資料 (座標、車道數等)
│   └── VDLiveList.json      # VD 即時交通資料
├── 2025-03-18_00-50-00/
└── ...
```

## JSON 資料結構

### VDList.json (VD 元資料)
```json
{
  "UpdateInfo": {...},
  "VDList": [
    {
      "VDID": "VD-11-0020-002-001",
      "PositionLon": 121.459501,
      "PositionLat": 25.149709,
      "RoadID": "300020",
      "RoadName": "臺2線",
      "LaneNum": 3
    }
  ]
}
```

### VDLiveList.json (即時交通資料)
```json
{
  "LiveTrafficData": [
    {
      "VDID": "VD-11-0020-002-001",
      "DataCollectTime": "2025-03-18T00:46:11+08:00",
      "UpdateInterval": 60,
      "AuthorityCode": "THB",
      "LinkFlows": [                    // 📍 關鍵：資料在這裡
        {
          "LinkID": "3000200000176F",
          "Lanes": [                    // 📍 每個 Link 包含多個 Lanes
            {
              "LaneID": 0,
              "LaneType": 1,
              "Speed": 59.0,            // 速度 (km/h)
              "Occupancy": 1.0,         // 佔有率 (%)
              "Vehicles": [             // 📍 車輛資料按車型分類
                {
                  "VehicleType": "L",   // L=大型車, S=小型車
                  "Volume": 0           // 該車型的流量
                },
                {
                  "VehicleType": "S",
                  "Volume": 1
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## 資料路徑解析

### 正確的資料存取路徑
```
VD → LinkFlows → [LinkID] → Lanes → [LaneID] → Speed/Occupancy/Vehicles
```

### 程式碼範例
```python
for vd in vd_live_list.LiveTrafficData:
    print(f"VD ID: {vd.VDID}")
    
    for link_flow in vd.LinkFlows:
        print(f"  Link ID: {link_flow.LinkID}")
        
        for lane in link_flow.Lanes:
            print(f"    Lane {lane.LaneID}: Speed={lane.Speed}, Occ={lane.Occupancy}")
            
            for vehicle in lane.Vehicles:
                print(f"      {vehicle.VehicleType}: {vehicle.Volume} 輛")
```

## 錯誤碼與異常值

### 常見錯誤碼
| 錯誤碼 | 含義 | 處理方式 |
|--------|------|----------|
| `-99` | 感測器故障或資料缺失 | 轉換為 NaN |
| `-1` | 初始化錯誤或未知狀態 | 轉換為 NaN |
| `255` | 數值溢出或系統錯誤 | 轉換為 NaN |
| `0` | 可能是真實零值或無車輛 | 保留 (需上下文判斷) |

### 異常值檢測規則
```python
def is_valid_traffic_value(value, feature_type):
    if value is None or np.isnan(value):
        return False
    
    # 錯誤碼過濾
    if value in [-99, -1, 255]:
        return False
    
    # 範圍檢查
    if feature_type == 'speed':
        return 0 <= value <= 200      # km/h
    elif feature_type == 'occupancy':
        return 0 <= value <= 100      # %
    elif feature_type == 'volume':
        return value >= 0             # 非負數
    
    return True
```

## 特徵定義

### 提取的交通特徵
| 特徵名稱 | 英文名稱 | 計算方式 | 正常範圍 | 單位 |
|----------|----------|----------|----------|------|
| 平均速度 | `avg_speed` | 所有車道速度的平均值 | 0-200 | km/h |
| 總流量 | `total_volume` | 所有車道所有車型流量總和 | ≥0 | 輛/分鐘 |
| 平均佔有率 | `avg_occupancy` | 所有車道佔有率的平均值 | 0-100 | % |
| 速度標準差 | `speed_std` | 各車道速度的標準差 | ≥0 | km/h |
| 車道數 | `lane_count` | 該 VD 的有效車道總數 | 1-10 | 個 |

### 聚合規則
```python
# 單一 VD 可能有多個 LinkFlows，每個 LinkFlow 有多個 Lanes
all_lanes = []
for link_flow in vd.LinkFlows:
    all_lanes.extend(link_flow.Lanes)

# 特徵計算
avg_speed = np.mean([lane.Speed for lane in all_lanes if is_valid(lane.Speed)])
total_volume = np.sum([sum(v.Volume for v in lane.Vehicles) for lane in all_lanes])
```

## 資料品質指標

### 預期的資料品質
| 指標 | 正常範圍 | 說明 |
|------|----------|------|
| VD 匹配率 | 80-90% | VDList 與 VDLiveList 的 ID 匹配比例 |
| 有效資料比例 | 60-90% | 過濾錯誤碼後的有效數值比例 |
| NaN 比例 | 10-40% | 包含錯誤碼過濾和缺失資料 |
| 時間覆蓋率 | >95% | 連續時間步的資料完整性 |

### 資料品質檢查範例
```python
def check_data_quality(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        features = f['data/features']
        
        # 計算各特徵的有效率
        for i, feature_name in enumerate(['avg_speed', 'total_volume', ...]):
            feature_data = features[:, :, i]
            valid_ratio = np.sum(~np.isnan(feature_data)) / feature_data.size
            print(f"{feature_name}: {valid_ratio:.1%} 有效")
```

## 常見問題與解決方案

### Q1: 為什麼 VD 匹配率不是 100%？
**A**: 正常現象。原因包括：
- 部分 VD 在特定時段可能無資料（維護、故障等）
- VDList 包含所有已註冊的 VD，但不是所有都實時運作
- VDLiveList 可能包含測試中的新 VD，尚未加入正式清單

### Q2: 為什麼有這麼多 NaN 值？
**A**: 主要原因：
- 錯誤碼過濾（-99, 255 等）
- 感測器故障或維護期間
- 特定時段交通量極低（如深夜）
- 新建或除役中的 VD

### Q3: Speed 為 0 是正常的嗎？
**A**: 需要判斷：
- 如果 Volume 也為 0：可能是真實的無車狀態
- 如果 Volume > 0 但 Speed = 0：可能是塞車或感測器異常
- 建議結合 Occupancy 一起判斷

### Q4: 如何處理車道數不一致？
**A**: 
- VDList 中的 LaneNum 是理論車道數
- 實際 LinkFlows 中的 Lanes 是當前運作的車道數
- 建議使用實際運作的車道數作為特徵

## 資料使用建議

### 模型訓練前處理
1. **錯誤碼過濾**: 移除 -99, -1, 255 等錯誤值
2. **異常值檢測**: 檢查超出合理範圍的數值
3. **缺失值處理**: 根據模型需求決定插值或遮罩策略
4. **正規化**: 各特徵使用適當的正規化方法
5. **時間對齊**: 確保時間戳記的一致性

### 空間分析建議
- 使用 VD 的經緯度座標進行空間相關性分析
- 考慮道路網絡結構（相同 RoadID 的 VD 可能相關）
- 注意不同方向的車道可能有不同的 LinkID

### 時間序列分析
- 資料更新間隔通常為 1-5 分鐘
- 考慮交通的週期性模式（小時、日、週）
- 注意節假日和特殊事件的影響

## 相關文檔

- [H5 轉換器使用指南](../scripts/dataset/pre-process/README_h5_converter.md)
- [模型訓練指南](./guides/training_scripts_guide.md)
- [座標系統說明](./spatial_coords.md)