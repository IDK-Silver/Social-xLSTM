# 模組功能說明

本文檔詳細說明 Social-xLSTM 專案各模組的功能與實現。專案基於社交池化與 xLSTM 技術，實現無拓撲依賴的交通流量預測。

## 核心套件結構 (`src/social_xlstm/`)

### 1. Dataset 模組 (`dataset/`) - 已重構為結構化子包

Dataset 模組已重構為清晰的子包結構，提供更好的可維護性：

#### `config/` - 配置管理
- **`base.py`**：
  - `TrafficDatasetConfig`: 數據集配置
  - `TrafficHDF5Config`: HDF5 轉換配置

#### `core/` - 核心數據操作
- **`processor.py`**：
  - `TrafficDataProcessor`: 數據前處理工具
  - 支援正規化、缺失值處理、時間特徵提取
  
- **`timeseries.py`**：
  - `TrafficTimeSeries`: 時間序列數據集類
  - 處理時間序列窗口和數據分割
  - 支援正規化器共享機制
  
- **`datamodule.py`**：
  - `TrafficDataModule`: PyTorch Lightning 數據模組
  - 管理訓練/驗證/測試數據載入器

#### `storage/` - 存儲與持久化
- **`h5_converter.py`**：
  - `TrafficHDF5Converter`: 執行 JSON 到 HDF5 轉換
  - `TrafficFeatureExtractor`: 從原始數據提取特徵
  - 智能文件檢查和批量處理
  
- **`h5_reader.py`**：
  - `TrafficHDF5Reader`: 讀取 HDF5 數據
  - 支援數據切片和選擇性載入
  
- **`feature.py`**：
  - `TrafficFeature`: 標準交通特徵定義
  - 包含 avg_speed, total_volume, avg_occupancy 等

#### `utils/` - 工具函數
- **`json_utils.py`**：
  - `VDInfo`, `VDLiveList`: JSON 數據結構
  - 作為 XML 和 HDF5 之間的中間格式
  
- **`xml_utils.py`**：
  - 解析原始 XML 交通數據
  - 處理流程：XML → Python 對象 → JSON
  
- **`zip_utils.py`**：
  - 壓縮檔案處理工具

#### 使用範例

```python
# 主要 API 使用（推薦）
from social_xlstm.dataset import TrafficDatasetConfig, TrafficTimeSeries, TrafficDataModule

# 或者使用子包（更明確）
from social_xlstm.dataset.config import TrafficDatasetConfig
from social_xlstm.dataset.core import TrafficTimeSeries, TrafficDataModule
from social_xlstm.dataset.storage import TrafficHDF5Reader
```

### 2. Models 模組 (`models/`)

#### `traffic_xlstm.py` (開發中)
- **功能**：基於 Social xLSTM 的交通預測模型
- **核心架構**：
  - **社交池化層 (Social Pooling)**：
    - 座標驅動的網格劃分機制
    - 隱式捕捉鄰近節點互動
    - 無需預先定義鄰接矩陣
  - **Hybrid xLSTM**：
    - sLSTM：標量記憶，支援指數門控
    - mLSTM：矩陣記憶，提供高容量存儲
    - 交錯堆疊於殘差骨幹中
- **輸入特徵**：
  - 當前流量 (qt)
  - 車道佔用率 (pt)
  - 車道速度 (st)
  - 車道數 (ℓi)
  - 時段特徵 (ti)
- **創新特點**：
  - 無拓撲依賴設計
  - 自動學習空間互動關係
  - 適應不規則節點分佈

#### `traffic_lstm.py`
- **功能**：傳統 LSTM 基準模型
- **架構組件**：
  - 標準 LSTM 單元（遺忘門、輸入門、輸出門）
  - 細胞狀態記憶機制
  - 多層堆疊支援
- **用途**：
  - 作為性能比較基準
  - 驗證 Social xLSTM 的改進效果
  - 評估無拓撲依賴的價值

### 3. Evaluation 模組 (`evaluation/`)

#### `evaluator.py`
- **功能**：模型評估指標計算
- **支援指標**：
  - MAE (平均絕對誤差)
  - MSE (均方誤差)
  - RMSE (均方根誤差)
  - MAPE (平均絕對百分比誤差)
  - R² (決定係數)

### 4. Utils 模組 (`utils/`)

#### `convert_coords.py`
- **功能**：座標系統轉換
- **應用**：處理不同地圖投影系統

#### `graph.py`
- **功能**：圖結構處理
- **用途**：建立交通網路的圖表示

#### `spatial_coords.py`
- **功能**：空間座標處理
- **特點**：支援地理空間運算

### 5. Visualization 模組 (`visualization/`)

#### `model.py`
- **功能**：模型視覺化
- **包含**：
  - 預測結果圖表
  - 訓練過程視覺化
  - 特徵重要性分析

## 腳本工具 (`scripts/`)

### 數據預處理 (`dataset/pre-process/`)

1. **`list_all_zips.py`**
   - **功能**：掃描並列出所有待處理的 ZIP 檔案
   - **參數**：
     - `--input_folder_list`: 輸入資料夾列表
     - `--output_file_path`: 輸出檔案路徑
   - **輸出**：ZIP 檔案清單，用於後續批量處理

2. **`unzip_and_to_json.py`**
   - **功能**：批量解壓縮檔案並轉換為 JSON
   - **參數**：
     - `--input_zip_list_path`: ZIP 清單檔案路徑
     - `--output_folder_path`: 輸出資料夾路徑
     - `--status_file`: 處理狀態檔案
   - **處理流程**：
     - 解壓縮 ZIP 檔案
     - 解析 XML 交通數據
     - 轉換為 JSON 格式
     - 追蹤處理狀態

3. **`create_h5_file.py`**
   - **功能**：將 JSON 數據轉換為 HDF5 高效存儲格式
   - **參數**：
     - `--source_dir`: 來源目錄
     - `--output_path`: 輸出路徑
     - `--selected_vdids`: 選定的 VD IDs（可選）
     - `--overwrite`: 覆寫選項
   - **特點**：
     - 按地區分塊最佳化
     - 支援選擇性處理特定監測站
     - 智能檢查避免重複轉換
     - 包含數據完整性驗證

### 實用工具 (`utils/`)

**`plot_vd_point.py`**
- **功能**：視覺化 VD 檢測器位置分布
- **參數**：
  - `--VDListJson`: VD 清單 JSON 檔案路徑
- **輸出**：
  - 監測站地理分布圖
  - 以南投為中心點的相對位置圖
  - 高屏交界區域監測站分布
- **應用**：
  - 選擇研究目標區域
  - 驗證監測站覆蓋範圍
  - 分析節點分布特性

## 訓練腳本 (`lab/train/` 與 `scripts/train/`)

### 基準模型訓練

1. **`lstm.py`**
   - **功能**：LSTM 基準模型訓練腳本
   - **特點**：
     - 完整的訓練循環實現
     - 支援 PyTorch Lightning
     - 內建評估指標計算
     - GPU 加速支援

2. **`pure_lstm.py`**
   - **功能**：簡化版 LSTM 實現
   - **用途**：
     - 快速原型開發
     - 概念驗證
     - 輕量級測試

### Social xLSTM 訓練 (開發中)

**`social_xlstm.py`** (計劃中)
- **功能**：Social xLSTM 模型訓練與驗證
- **預期特點**：
  - 座標驅動社交池化訓練
  - Hybrid xLSTM 記憶機制
  - 與基準模型性能比較
  - 無拓撲依賴驗證

## 工作流程管理

### Snakemake 管線
- **`Snakefile`**：定義完整數據處理流程
- **`config.yaml`**：集中配置管理
- **自動化步驟**：
  1. **資料蒐集階段**：
     - 列出台灣公路總局數據 ZIP 檔案
     - 驗證檔案完整性
  2. **資料預處理階段**：
     - 批量解壓縮檔案
     - XML 到 JSON 格式轉換
     - 資料清理與驗證
  3. **資料最佳化階段**：
     - JSON 到 HDF5 高效存儲
     - 按地區分塊最佳化
     - 建立時空索引
  4. **模型訓練階段**：
     - LSTM 基準模型訓練
     - Social xLSTM 模型訓練（開發中）
     - 模型評估與比較
- **配置特點**：
  - 支援部分處理和增量更新
  - 自動依賴管理
  - 並行處理最佳化
  - 錯誤處理與重試機制

## 測試架構 (`test/`)

### 測試組織結構

- **`test_social_xlstm/`**：核心套件測試
  - `dataset/`：數據處理模組測試
    - `test_json_utils.py`：JSON 數據處理測試
    - `test_zip_utils.py`：檔案解壓縮測試
    - `test_extract_archive.py`：歸檔提取測試
  - `test_loader_xlstm.py`：xLSTM 載入器測試

- **`project/`**：專案級測試
  - `gpu_test.py`：GPU 功能測試
  - `xlstm/basic_stack.py`：xLSTM 基本堆疊測試

### 測試特點

- **框架**：使用 pytest 測試框架
- **執行模式**：
  - 支援並行測試執行 (`pytest -n auto`)
  - 單一檔案測試
  - 完整測試套件
- **覆蓋範圍**：
  - 數據處理管線驗證
  - 模型載入與初始化
  - GPU 相容性測試
  - 檔案 I/O 操作測試
- **測試數據**：
  - 使用模擬交通數據
  - 小規模真實數據樣本
  - 邊界條件測試案例