# Dataset 模組文檔

## 模組概述

`social_xlstm.dataset` 模組負責交通數據的完整處理流程，從原始資料載入到深度學習模型的輸入準備。此模組實現了高效的數據處理管線，支援大規模時間序列交通數據的預處理、特徵提取和批次載入。

## 📦 模組架構

本模組採用結構化設計，分為四個主要子套件：

```
social_xlstm.dataset/
├── config/          # 配置管理
│   └── base.py     # TrafficDatasetConfig, TrafficHDF5Config
├── core/           # 核心數據操作
│   ├── processor.py    # TrafficDataProcessor
│   ├── timeseries.py   # TrafficTimeSeries
│   └── datamodule.py   # TrafficDataModule
├── storage/        # 存儲與持久化
│   ├── h5_converter.py # HDF5 轉換器
│   ├── h5_reader.py    # HDF5 讀取器
│   └── feature.py      # TrafficFeature dataclass
└── utils/          # 工具函數
    ├── json_utils.py   # JSON 處理
    ├── xml_utils.py    # XML 轉換
    └── zip_utils.py    # 壓縮檔案處理
```

## 🚀 快速開始

### 基本使用範例

```python
# 1. 導入核心類別
from social_xlstm.dataset import TrafficDatasetConfig, TrafficDataModule
from social_xlstm.dataset.storage import create_traffic_hdf5

# 2. 創建 HDF5 數據文件
create_traffic_hdf5(
    source_dir="data/json",
    output_path="data/traffic.h5",
    selected_vdids=["VD-001", "VD-002"]
)

# 3. 配置數據集
config = TrafficDatasetConfig(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    normalize=True,
    batch_size=32
)

# 4. 創建數據模組
data_module = TrafficDataModule(config)
data_module.setup()

# 5. 獲取數據載入器
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

## 核心組件

### 1. 配置管理 (`config/`)

#### TrafficDatasetConfig
**位置**: `social_xlstm.dataset.config.base`

**功能**: 統一管理數據集配置參數

**主要配置項**:
- `hdf5_path`: HDF5 數據文件路徑
- `sequence_length`: 輸入序列長度
- `prediction_length`: 預測序列長度
- `normalize`: 是否正規化數據
- `batch_size`: 批次大小
- `num_workers`: 數據載入器工作線程數
- `train_split`: 訓練集比例
- `val_split`: 驗證集比例

**使用範例**:
```python
from social_xlstm.dataset.config import TrafficDatasetConfig

config = TrafficDatasetConfig(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    normalize=True,
    normalization_method="standard",
    batch_size=32,
    num_workers=4
)
```

#### TrafficHDF5Config
**功能**: HDF5 轉換配置管理

**主要配置項**:
- `compression`: 壓縮方式 ("gzip", "lzf")
- `chunk_size`: 數據塊大小
- `selected_vdids`: 選定的 VD ID 列表
- `time_range`: 時間範圍過濾 (詳見下方說明)

#### time_range 參數詳解

**用途**: 指定數據處理的時間範圍過濾，控制 HDF5 轉換過程中處理的數據時間窗口。

**類型**: `Optional[Tuple[str, str]]`

**格式要求**: `"YYYY-MM-DD_HH-MM-SS,YYYY-MM-DD_HH-MM-SS"`
- 開始時間和結束時間用逗號分隔
- 時間格式：年-月-日_時-分-秒
- 範例：`"2025-03-18_00-00-00,2025-03-18_23-59-59"`

**行為說明**:
- `null` 或 `None`: 處理所有可用數據（無時間過濾）- 推薦用於生產環境
- 具體時間範圍: 僅處理指定時間段內的數據 - 適合開發測試

**實現位置**: `src/social_xlstm/dataset/storage/h5_converter.py:419-425`

**配置範例**:
```yaml
# 開發環境 - 處理特定時間範圍
time_range: "2025-03-18_00-00-00,2025-03-18_23-59-59"

# 生產環境 - 處理所有數據
time_range: null
```

**相關配置文件**:
- 開發配置: `cfgs/snakemake/dev.yaml`
- 生產配置: `cfgs/snakemake/default.yaml`

### 2. 數據特徵 (`storage/feature.py`)

**功能**: 定義標準化的交通特徵數據結構

**核心類別**:
- `TrafficFeature`: 交通特徵的數據類別

**主要功能**:
- 封裝五種核心交通指標：平均速度、總交通量、平均占有率、速度標準差、車道數
- 提供字段名稱常量，避免字串硬編碼
- 支援字典轉換和字段名稱查詢

**使用場景**:
```python
from social_xlstm.dataset.storage import TrafficFeature

# 創建交通特徵
feature = TrafficFeature(
    avg_speed=65.5,
    total_volume=120,
    avg_occupancy=15.2,
    speed_std=8.3,
    lane_count=3
)

# 獲取字段名稱
field_names = TrafficFeature.get_field_names()
feature_dict = feature.to_dict()
```

**設計評估**:
- ✅ **優點**: 結構清晰，類型安全，易於維護
- ✅ **優點**: 字段名稱常量避免拼寫錯誤
- ✅ **改進**: 已移至 storage 子套件，職責更清晰

### 3. 核心數據操作 (`core/`)

#### TrafficDataProcessor (`core/processor.py`)
**功能**: 數據預處理和正規化

**主要功能**:
- 多種正規化方法（標準化、最小-最大）
- 缺失值處理（零填充、前向填充、插值）
- 時間特徵工程（小時、星期、月份的循環編碼）
- 序列數據窗口切片

**使用範例**:
```python
from social_xlstm.dataset.core import TrafficDataProcessor

processor = TrafficDataProcessor(
    normalize=True,
    normalization_method="standard",
    missing_value_strategy="forward_fill",
    add_time_features=True
)

# 處理數據
processed_data = processor.process_data(raw_data)
```

#### TrafficTimeSeries (`core/timeseries.py`)
**功能**: PyTorch 時間序列數據集

**主要功能**:
- 時間序列窗口切片
- 動態序列長度支援
- 記憶體高效的數據載入
- 多VD數據支援

**使用範例**:
```python
from social_xlstm.dataset.core import TrafficTimeSeries

dataset = TrafficTimeSeries(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    vdids=["VD-001", "VD-002"],
    processor=processor
)
```

#### TrafficDataModule (`core/datamodule.py`)
**功能**: PyTorch Lightning 數據模組

**主要功能**:
- 自動訓練/驗證/測試集分割
- 數據載入器管理
- 分散式訓練支援
- 動態批次大小調整

**使用範例**:
```python
from social_xlstm.dataset.core import TrafficDataModule
from social_xlstm.dataset.config import TrafficDatasetConfig

config = TrafficDatasetConfig(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    batch_size=32
)

data_module = TrafficDataModule(config)
data_module.setup()

# 獲取數據載入器
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

**設計評估**:
- ✅ **優點**: 職責分離，結構清晰
- ✅ **優點**: 支援 PyTorch Lightning 最佳實踐
- ✅ **改進**: 已重構為獨立模組，易於維護

### 4. 存儲與持久化 (`storage/`)

#### TrafficHDF5Converter (`storage/h5_converter.py`)
**功能**: JSON 到 HDF5 格式轉換

**核心類別**:
- `TrafficHDF5Converter`: HDF5 轉換器
- `TrafficFeatureExtractor`: 特徵提取器

**主要功能**:
- 批次處理 JSON 數據轉換為 HDF5
- 智能檢查（配置一致性、文件更新時間）
- 壓縮存儲和增量更新
- 車道級別特徵聚合

**使用範例**:
```python
from social_xlstm.dataset.storage import TrafficHDF5Converter, TrafficHDF5Config

config = TrafficHDF5Config(
    compression="gzip",
    chunk_size=1000,
    selected_vdids=["VD-001", "VD-002"]
)

converter = TrafficHDF5Converter(config)
converter.convert_directory(
    source_dir="data/json",
    output_path="data/traffic.h5",
    overwrite=False
)
```

#### TrafficHDF5Reader (`storage/h5_reader.py`)
**功能**: HDF5 數據讀取和查詢

**主要功能**:
- 高效的 HDF5 數據讀取
- 時間範圍查詢
- VD 級別數據過濾
- 記憶體效率優化

**使用範例**:
```python
from social_xlstm.dataset.storage import TrafficHDF5Reader

reader = TrafficHDF5Reader("data/traffic.h5")

# 讀取特定 VD 數據
data = reader.read_vd_data(
    vdid="VD-001",
    start_time="2023-01-01",
    end_time="2023-01-31"
)

# 獲取可用 VD 列表
available_vdids = reader.get_available_vdids()
```

#### 便捷函數
```python
from social_xlstm.dataset.storage import create_traffic_hdf5

# 一鍵創建 HDF5 文件
create_traffic_hdf5(
    source_dir="data/json",
    output_path="data/traffic.h5",
    selected_vdids=["VD-001", "VD-002"],
    overwrite=False
)
```

**設計評估**:
- ✅ **優點**: 轉換器和讀取器分離，職責清晰
- ✅ **優點**: 智能檢查機制，避免重複轉換
- ✅ **優點**: 支援大規模數據處理和壓縮
- ✅ **改進**: 已分離為獨立模組，易於維護

### 5. 工具函數 (`utils/`)

#### JSON 工具 (`utils/json_utils.py`)
**功能**: 交通數據 JSON 格式處理

**核心類別**:
- `VDInfo`: 車輛偵測器資訊
- `VDLiveList`: 即時交通數據列表
- `VD`, `VehicleData`, `LaneData`: 數據結構類別

**主要功能**:
- 結構化解析交通 JSON 數據
- VDID 正規化處理
- 車道和車輛數據封裝

**使用範例**:
```python
from social_xlstm.dataset.utils import VDInfo, VDLiveList

# 解析 VD 資訊
vd_info = VDInfo.from_json(vd_info_json)
vd_dict = vd_info.to_dict()

# 處理即時數據
live_list = VDLiveList.from_json(live_data_json)
filtered_data = live_list.filter_by_vdids(["VD-001", "VD-002"])
```

#### XML 工具 (`utils/xml_utils.py`)
**功能**: XML 到 JSON 格式轉換

**主要功能**:
- XML 數據解析
- 結構化轉換為 JSON
- 錯誤處理和驗證

**使用範例**:
```python
from social_xlstm.dataset.utils import VDList_xml_to_Json

# 轉換 XML 到 JSON
json_data = VDList_xml_to_Json(xml_file_path)
```

#### ZIP 工具 (`utils/zip_utils.py`)
**功能**: 壓縮檔案處理

**主要功能**:
- 支援多種壓縮格式（ZIP, 7z）
- 自動解壓縮
- 時間戳解析
- 錯誤處理

**使用範例**:
```python
from social_xlstm.dataset.utils import extract_archive, parse_time_from_filename

# 解壓縮檔案
extract_archive("data.zip", "output_dir")

# 解析時間戳
timestamp = parse_time_from_filename("data_20230101_1200.zip")
```

**設計評估**:
- ✅ **優點**: 完整的數據結構定義
- ✅ **優點**: 支援 JSON 序列化/反序列化
- ✅ **優點**: 工具函數組織清晰，易於使用
- ✅ **改進**: 已移至 utils 子套件，職責更明確

## 🔄 API 參考

### 主要導入方式

```python
# 主要類別 (從主模組導入)
from social_xlstm.dataset import (
    TrafficDatasetConfig,
    TrafficDataModule,
    TrafficTimeSeries,
    TrafficDataProcessor
)

# 存儲相關 (從子模組導入)
from social_xlstm.dataset.storage import (
    TrafficFeature,
    TrafficHDF5Converter,
    TrafficHDF5Reader,
    create_traffic_hdf5
)

# 工具函數 (從子模組導入)
from social_xlstm.dataset.utils import (
    VDInfo,
    VDLiveList,
    extract_archive,
    VDList_xml_to_Json
)
```

### 完整工作流程

```python
# 1. 數據預處理
from social_xlstm.dataset.storage import create_traffic_hdf5
from social_xlstm.dataset.utils import extract_archive

# 解壓縮原始數據
extract_archive("traffic_data.zip", "data/raw")

# 轉換為 HDF5
create_traffic_hdf5(
    source_dir="data/raw",
    output_path="data/traffic.h5",
    selected_vdids=["VD-001", "VD-002", "VD-003"]
)

# 2. 配置數據集
from social_xlstm.dataset import TrafficDatasetConfig

config = TrafficDatasetConfig(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    normalize=True,
    normalization_method="standard",
    missing_value_strategy="forward_fill",
    add_time_features=True,
    batch_size=32,
    num_workers=4,
    train_split=0.7,
    val_split=0.2
)

# 3. 創建數據模組
from social_xlstm.dataset import TrafficDataModule

data_module = TrafficDataModule(config)
data_module.setup()

# 4. 使用於訓練
import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1
)

# 假設有一個模型
trainer.fit(model, data_module)
```

## 📊 效能與優化

### 記憶體優化
- **HDF5 壓縮**: 使用 gzip 壓縮，節省 60-70% 儲存空間
- **分塊讀取**: 按需載入數據，降低記憶體使用
- **數據預取**: 使用多線程預先載入數據

### 處理效能
- **批次處理**: 支援大批次數據轉換
- **並行載入**: 多進程數據載入器
- **快取機制**: 智能檢查避免重複轉換

### 可擴展性
- **多 VD 支援**: 同時處理多個車輛偵測器
- **時間範圍過濾**: 靈活的時間範圍選擇
- **動態配置**: 運行時調整參數

## 🔧 故障排除

### 常見問題

#### 1. HDF5 文件損壞
```python
# 檢查 HDF5 文件完整性
from social_xlstm.dataset.storage import TrafficHDF5Reader

reader = TrafficHDF5Reader("data/traffic.h5")
try:
    vdids = reader.get_available_vdids()
    print(f"可用的 VD IDs: {vdids}")
except Exception as e:
    print(f"HDF5 文件可能損壞: {e}")
```

#### 2. 記憶體不足
```python
# 減少批次大小和工作線程
config = TrafficDatasetConfig(
    batch_size=16,  # 減少批次大小
    num_workers=2,  # 減少工作線程
    sequence_length=30  # 減少序列長度
)
```

#### 3. 正規化問題
```python
# 檢查正規化統計
from social_xlstm.dataset.core import TrafficDataProcessor

processor = TrafficDataProcessor(normalize=True)
processor.fit(train_data)
print(f"均值: {processor.mean_}")
print(f"標準差: {processor.std_}")
```

## 整體評估

### ✅ 優點
1. **結構化架構**: 清晰的模組分離和職責劃分
2. **完整的數據流程**: 從原始數據到模型輸入的完整處理
3. **高效存儲**: HDF5 格式支援大規模數據
4. **靈活配置**: 豐富的配置選項滿足不同需求
5. **PyTorch 整合**: 完整的 PyTorch Lightning 支援
6. **易於使用**: 清晰的 API 和豐富的使用範例

### ✅ 已解決的問題
1. **✅ 代碼結構**: 已重構為結構化子套件
2. **✅ 職責分離**: 轉換器和讀取器已分離
3. **✅ 正規化器**: 已實現正確的正規化器共享機制
4. **✅ 文檔完整**: 提供詳細的使用指南和範例

### 🔄 持續改進
1. **效能優化**: 持續優化數據載入速度
2. **錯誤處理**: 增強異常處理和日誌記錄
3. **測試覆蓋**: 提升測試覆蓋率
4. **文檔更新**: 隨功能更新保持文檔同步

## 🎯 使用建議

### 最佳實踐
1. **配置管理**: 使用 `TrafficDatasetConfig` 統一管理所有配置
2. **HDF5 優先**: 對大數據集使用 HDF5 格式存儲，提升載入速度
3. **智能檢查**: 利用轉換器的智能檢查避免重複轉換
4. **正規化選擇**: 根據數據特性選擇合適的正規化方法
5. **批次大小**: 根據 GPU 記憶體調整批次大小
6. **多進程載入**: 使用 `num_workers` 提升數據載入效率

### 開發階段建議

#### 開發階段
```python
# 使用較小的數據集進行快速測試
config = TrafficDatasetConfig(
    hdf5_path="data/traffic_small.h5",
    sequence_length=30,
    prediction_length=5,
    batch_size=16,
    num_workers=2
)
```

#### 實驗階段
```python
# 使用完整配置進行實驗
config = TrafficDatasetConfig(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    normalize=True,
    normalization_method="standard",
    add_time_features=True,
    batch_size=32,
    num_workers=4
)
```

#### 生產階段
```python
# 優化配置用於生產環境
config = TrafficDatasetConfig(
    hdf5_path="data/traffic.h5",
    sequence_length=60,
    prediction_length=15,
    normalize=True,
    batch_size=64,
    num_workers=8,
    pin_memory=True  # GPU 訓練時啟用
)
```

## 📈 下一步發展

### 即將實現的功能
1. **Space-Time 特徵**: 結合座標系統的空間特徵
2. **Social Pooling**: 支援 Social Pooling 算法的數據準備
3. **多模態數據**: 支援更多交通數據類型
4. **實時處理**: 支援實時數據流處理

### 與其他模組的整合
- **與 `models/` 模組**: 為 xLSTM 和 Social Pooling 提供數據
- **與 `utils/spatial_coords.py`**: 整合座標系統進行空間特徵提取
- **與 `evaluation/` 模組**: 提供評估數據集

## 📝 總結

**Dataset 模組**已完成重構，提供了完整、高效、結構化的交通數據處理解決方案。模組架構清晰，功能完整，API 易用，準備好支援下一階段的 **Social Pooling** 和 **xLSTM** 功能開發。

**主要成就**:
- ✅ 結構化重構完成
- ✅ 完整的 API 設計
- ✅ 高效的數據處理流程
- ✅ 豐富的配置選項
- ✅ 完整的文檔和範例

**準備狀態**: 🚀 **準備就緒**，可進行核心功能開發。