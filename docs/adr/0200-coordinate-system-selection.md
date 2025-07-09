# ADR-0200: 座標系統選擇與 Social Pooling 實現

**狀態**: Accepted  
**日期**: 2025-01-08  
**決策者**: 專案團隊  
**技術故事**: Social Pooling 需要固定座標系統支援

## 背景與問題陳述

Social Pooling 機制需要將鄰居節點根據空間位置分配到網格中，這需要：
1. 將 GPS 座標轉換為平面座標系統
2. 計算節點間的歐幾里得距離
3. 進行相對位移計算 (Δx, Δy)
4. 網格劃分和節點分配

原始問題：不確定如何處理 VD 監測站的 (x, y) 座標。

## 決策驅動因素

- **Social Pooling 需求**: 需要準確的平面座標系統
- **現有系統發現**: 專案中已有完整的座標處理實現
- **台灣地理特性**: 需要適合台灣地區的投影系統
- **計算效率**: 座標轉換應該高效且準確

## 考慮的選項

### 選項 1: 重新設計座標系統（UTM 投影）

- **優點**:
  - UTM Zone 51N 對台灣地區精度更高
  - 保持距離和角度的準確性
  - 國際標準，便於比較
- **缺點**:
  - 需要重新實現所有座標轉換功能
  - 開發時間長
  - 可能引入新的 bug

### 選項 2: 使用現有墨卡托投影系統

- **優點**:
  - 已有完整實現 (`spatial_coords.py`)
  - 雙向轉換、距離計算都已測試
  - 以台灣南投為原點，適合台灣地區
  - 立即可用，無需額外開發
- **缺點**:
  - 墨卡托投影在高緯度有輕微變形
  - 精度略低於 UTM

### 選項 3: 混合方案

- **優點**:
  - 保留現有系統，同時添加 UTM 支援
  - 靈活性高
- **缺點**:
  - 複雜度增加
  - 維護成本高

## 決策結果

**選擇**: 選項 2 - 使用現有墨卡托投影系統

**理由**: 
1. **現有系統完整**：`CoordinateSystem` 類別功能齊全，包含所有需要的功能
2. **滿足需求**：墨卡托投影在台灣地區的精度足夠 Social Pooling 使用
3. **開發效率**：立即可用，無需額外開發時間
4. **測試完善**：現有系統已有完整的測試和文檔
5. **專案聚焦**：可以專注於 Social Pooling 算法本身

## 實施細節

### 座標系統規格
```python
# 原點：台灣南投
LAT_ORIGIN = 23.9150°N
LON_ORIGIN = 120.6846°E
EARTH_RADIUS = 6378137 meters (WGS84)
```

### Social Pooling 網格配置
```python
GRID_SIZE = (8, 8)          # 8×8 網格
GRID_RADIUS_X = 25000       # 25 公里
GRID_RADIUS_Y = 25000       # 25 公里
# 每個網格格子約 6.25km × 6.25km
```

### 使用方式
```python
from social_xlstm.utils.spatial_coords import CoordinateSystem

# 預處理 VD 座標
def preprocess_vd_coordinates(vd_locations):
    vd_coords = {}
    for vd_id, (lat, lon) in vd_locations.items():
        coord = CoordinateSystem.create_from_latlon(lat, lon)
        x, y = coord.to_xy()
        vd_coords[vd_id] = (x, y)
    return vd_coords

# Social Pooling 網格分配
def assign_to_grid(target_x, target_y, neighbor_x, neighbor_y):
    delta_x = neighbor_x - target_x
    delta_y = neighbor_y - target_y
    
    m = int((delta_x + GRID_RADIUS_X) / (2 * GRID_RADIUS_X / GRID_SIZE[0]))
    n = int((delta_y + GRID_RADIUS_Y) / (2 * GRID_RADIUS_Y / GRID_SIZE[1]))
    
    return max(0, min(GRID_SIZE[0] - 1, m)), max(0, min(GRID_SIZE[1] - 1, n))
```

## 後果

### 正面後果

- **立即可用**: 無需等待座標系統開發
- **穩定可靠**: 現有系統已經過測試
- **文檔完整**: `spatial_coords.py` 有完整的使用範例
- **功能豐富**: 支援距離計算、方位角等額外功能

### 負面後果

- **精度限制**: 墨卡托投影精度略低於 UTM
- **未來擴展**: 如需極高精度可能需要重新考慮

### 風險與緩解措施

- **精度不足風險**: 墨卡托投影精度不滿足需求 / **緩解**: 在高屏地區測試，驗證精度是否足夠
- **座標轉換錯誤**: 現有系統可能有 bug / **緩解**: 增加座標轉換的單元測試
- **效能問題**: 大量座標轉換可能較慢 / **緩解**: 預先轉換並快取所有 VD 座標

## 相關決策

- [ADR-0100: Social Pooling vs Graph-based 方法](0100-social-pooling-vs-graph-networks.md)
- [ADR-0201: 數據管線格式選擇](0201-data-pipeline-xml-json-hdf5.md)

## 註記

### 現有系統功能概覽
- **雙向轉換**: GPS ↔ 平面座標
- **距離計算**: 歐幾里得距離
- **方位角計算**: 兩點間方位角
- **靜態方法**: 便於快速計算
- **錯誤處理**: 完整的異常處理

### 精度評估
台灣地區的墨卡托投影變形極小（< 0.1%），對於 Social Pooling 的網格劃分精度完全足夠。

### 未來考量
如果後續研究需要更高精度（如精確到公尺級別的應用），可以考慮添加 UTM 投影支援，但當前需求下現有系統已足夠。