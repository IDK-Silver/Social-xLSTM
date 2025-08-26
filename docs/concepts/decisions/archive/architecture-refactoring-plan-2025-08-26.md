# 📚 已歸檔：Social-xLSTM 架構簡化計劃（基於 YAGNI 原則）

> **歷史文檔**  
> 原位置：`docs/_work/task.md`  
> 歸檔日期：2025-08-26  
> 狀態：已完成重構階段

此文檔記錄了 Social-xLSTM 項目在 2025-08-26 進行的大規模架構重構計劃和執行過程，基於 YAGNI 原則清理過度設計。

---

# Social-xLSTM 架構簡化計劃（基於 YAGNI 原則）

## ✅ **2025-08-26 完成：DataModule 統一重構**

**成功完成 PR-2：DataModule 統一整合**

### 已實現功能
1. **TrafficDatasetConfig 增強**：添加 `batch_format` 欄位支援 'centralized' | 'distributed' 
2. **TrafficDataModule 統一**：整合分散式邏輯，根據配置自動選擇批次格式
3. **DistributedCollator 類**：可序列化的分散式 collate function，支援多進程 DataLoader
4. **相容性保持**：DistributedTrafficDataModule 改為相容性 shim，發出棄用警告
5. **100% 向後兼容**：train_multi_vd.py 繼續正常工作

### 程式碼減少統計
- 移除重複邏輯：~180 行 (DistributedTrafficDataModule 的重複實現)
- 新增核心功能：~120 行 (collators.py + datamodule.py 整合)
- **淨減少約 60 行**，同時提升程式碼品質與維護性

### 技術實現
- **配置驅動**：透過 `config.batch_format` 控制行為
- **條件性 collate_fn**：在 setup() 階段根據配置準備適當的 collate function  
- **統一 DataLoader 創建**：`_make_dataloader()` 方法消除重複程式碼
- **Pickle 相容性**：DistributedCollator 類設計為頂層可序列化類

---


## 問題分析總結

經過深度代碼分析確認：**代碼中存在嚴重的過度設計和重複實現**，大量不必要的抽象層導致開發困惑。

### 核心問題
1. **TensorSpec 過度驗證**：236行代碼實際只做基本的形狀檢查
2. **類別名稱衝突**：兩個同名 `VDXLSTMManager` 導致匯入混亂
3. **單一實現抽象**：`interfaces/` 目錄包含大量只有一個實現的 ABC
4. **TrafficFeature 過度設計**：58行提供5個字段的簡單功能

---

## 具體分析結果

### 1. TensorSpec 分析 ❌ **可完全移除**

**問題**：236行代碼僅做基本形狀檢查，過度複雜化
**解決方案**：創建簡單的 `src/social_xlstm/utils/tensor_checks.py` 替代

### 2. VDXLSTMManager 衝突 ❌ **需要統一**

**問題**：存在兩個同名類別，造成匯入混亂
- `interfaces/vd_manager.py` - 使用 LSTM（錯誤）
- `models/vd_xlstm_manager.py` - 使用真正 xLSTM（正確）

**為什麼 VDXLSTMManager 是必要的**：
- Social-xLSTM 需要為每個 VD 維護獨立的 xLSTM 實例
- 需要 `nn.ModuleDict` 管理參數註冊和設備處理
- 支援動態 VD 集合的延遲初始化

**解決方案**：保留 `models/` 版本，移除 `interfaces/` 版本

### 3. TrafficFeature 分析 ⚠️ **需要簡化**

**問題**：58行代碼僅提供基本的字典轉換功能
**解決方案**：替換為簡單的常數定義和函數

### 4. interfaces/ 目錄分析 ⚠️ **大部分可移除**

**問題**：大量單一實現的抽象基類和重複代碼
**解決方案**：移除不必要的抽象層，保留必要的型別定義

---

## 實施計劃（5個 PR 分階段執行）

### PR-1: 移除 TensorSpec（最高 ROI）

**影響文件**：
- `src/social_xlstm/interfaces/tensor_spec.py`（刪除）
- `src/social_xlstm/models/vd_xlstm_manager.py`（更新匯入）
- `src/social_xlstm/data/distributed_datamodule.py`（更新匯入）

**具體步驟**：
1. 創建 `src/social_xlstm/utils/tensor_checks.py`（25行）
2. 替換所有 `TensorSpec` 使用為簡單 `assert_shape` 調用
3. 添加臨時兼容性 shim（可選）
4. 添加基本測試

**預期減少**：淨減少 ~200行

### PR-2: 統一 VDXLSTMManager

**影響文件**：
- `src/social_xlstm/interfaces/vd_manager.py`（刪除或添加別名）
- 所有匯入 `interfaces.vd_manager.VDXLSTMManager` 的文件

**具體步驟**：
1. 更新所有匯入路徑到 `models.vd_xlstm_manager`
2. 添加臨時別名（可選，下版本移除）
3. 更新包匯出

**預期減少**：~50行（或臨時保留，下個版本刪除）

### PR-3: PEMS-BAY 數據集整合實施方案 ✅ **已完成**

#### 📊 **數據品質分析結果**

**PEMS-BAY 數據品質優秀**：
- **完整性**: 52,116時間步 × 325傳感器，**零缺值**，數據品質極高
- **速度範圍**: 0-85.1 mph，平均62.6 mph，無異常值  
- **零值處理**: 僅521個零值（<0.01%），可能為交通堵塞，建議保留
- **時間連續**: 2017年1-6月，5分鐘間隔，無重複時間戳
- **空間覆蓋**: 舊金山灣區，緯度37.25-37.43，經度-122.08至-121.84

#### 💻 **已實現轉換腳本**

**位置**: `scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py`

**簡化設計**（避免過度複雜）：
- **單一模式**: 直接轉換所有 6 個特徵 (F=6)
- **缺值透明**: 轉換腳本不處理缺值，保留原始狀態交給後續流程  
- **參數極簡**: 僅需輸入輸出路徑，專注核心轉換功能

**使用方式**:
```bash
python scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py \
    --data-csv blob/dataset/raw/PEMS-BAY/PEMS-BAY.csv \
    --meta-csv blob/dataset/raw/PEMS-BAY/PEMS-BAY-META.csv \
    --output-h5 blob/dataset/processed/pems_bay.h5 \
    --validate
```

#### 📋 **輸出HDF5結構（與Taiwan VD完全兼容）**

```python
# 階層結構
/data/
  └── features: [52116, 325, 6] float32, gzip壓縮
/metadata/
  ├── vdids: [325] string, 傳感器ID  
  ├── timestamps: [52116] int64, Unix epoch
  ├── feature_names: [6] string, ['avg_speed', 'lanes', 'length', 'latitude', 'longitude', 'direction']
  ├── frequency: "5min"
  ├── units: JSON格式單位定義
  └── source: "PEMS-BAY 2017-01 to 2017-06"

# 根屬性
dataset_name: "pems_bay"
feature_set: "pems_bay_v1" 
feature_schema_version: "1.0"
creation_date: ISO8601時間戳
```

#### 🏷️ **6個特徵定義**

1. **avg_speed**: 速度數據 (mph → km/h, ×1.609344)
2. **lanes**: 車道數 (從META廣播到所有時間步)
3. **length**: 傳感器長度 miles (從META廣播)
4. **latitude**: 緯度座標 (從META廣播)  
5. **longitude**: 經度座標 (從META廣播)
6. **direction**: 交通方向 N/S/E/W → 0/180/90/270度 (從META廣播)

#### 🔍 **內建驗證機制**

腳本包含 `--validate` 選項，自動檢查：
- ✅ HDF5階層結構完整性
- ✅ 維度一致性 (T, N, F匹配)  
- ✅ 時間戳單調性
- ✅ 數據範圍合理性
- ✅ 特徵統計摘要

#### ⚠️ **缺值處理策略**

**當前PEMS-BAY**: 數據完整，無缺值問題

**未來如有缺值**: 
- **轉換腳本**: 保留NaN，不做處理 (單一職責)
- **後續處理**: 使用TrafficDataModule的 `fill_missing: "interpolate"`
- **或獨立腳本**: 可另寫專門的缺值處理腳本

#### 🎯 **已達成目標**
- **格式兼容**: 與Taiwan VD使用相同階層HDF5結構
- **特徵完整**: 6個特徵涵蓋速度和空間元數據
- **品質保證**: 內建驗證確保數據完整性
- **文檔齊全**: 包含README.md說明使用方式
- **專注轉換**: 避免過度設計，單純格式轉換功能

### PR-X: Profile 配置系統 ✅ **2025-08-26 新增**

**創建模塊化配置合併系統**：為 `train_multi_vd.py` 提供統一的 profile 配置接口。

#### 📁 **新增文件結構**
```
cfgs/profiles/
└── pems_bay_dev.yaml     # PEMS-BAY 開發 profile
```

#### 🔧 **增強 YAML 工具** (`src/social_xlstm/utils/yaml.py`)
- `deep_merge()`: 遞歸字典合併函數
- `load_profile_config()`: Profile 配置載入器
- 支援相對路徑解析和錯誤處理

#### 📋 **Profile 配置格式**
```yaml
configs:
  - cfgs/data/dev.yaml
  - cfgs/models/xlstm.yaml
  - cfgs/training/dev.yaml
  - cfgs/datasets/pems_bay.yaml

overrides:
  data:
    path: "blob/dataset/processed/pems_bay.h5"
    loader:
      batch_size: 16
```

#### 🎯 **責任分工**
- **datasets/pems_bay.yaml**: 數據集特定參數 (`input_size: 6`, `output_size: 6`)
- **profiles/pems_bay_dev.yaml**: 配置組合邏輯和實驗特定覆蓋

#### ⚡ **使用方式**
```bash
python scripts/train/with_social_pooling/train_multi_vd.py \
    --config cfgs/profiles/pems_bay_dev.yaml
```

**向後兼容**: `train_multi_vd.py` 無需修改，繼續使用 `load_yaml_file_to_dict()`


### PR-4: 清理 interfaces/ 抽象層

**影響文件**：
- `src/social_xlstm/interfaces/base_social_pooling.py`
- `src/social_xlstm/interfaces/distributed_model.py`
- `src/social_xlstm/interfaces/config.py`
- `src/social_xlstm/interfaces/types.py`

**具體步驟**：
1. 移除單一實現的 ABC
2. 將有用的型別定義移到實際使用的模組
3. 刪除重複的配置類別
4. 保留必要的型別別名

**預期減少**：~80-200行

### PR-5: 整理和文檔更新

**具體步驟**：
1. 移除所有臨時別名
2. 更新 README 和使用範例
3. 執行全域匯入檢查
4. 可選：添加 lint 規則防止新的單一實現抽象

**預期減少**：~50-100行（移除臨時代碼）

---

## 預期效益

### 量化指標
- **代碼減少**：365-485行淨減少（約40-50%）
- **文件減少**：3-5個不必要的抽象文件
- **維護負擔**：消除重複類別和混亂匯入

### 質化改進
- **開發效率**：消除類別名稱衝突和匯入混亂
- **認知負擔**：簡化數據流，明確責任分工
- **測試覆蓋**：減少抽象層使單元測試更直接

---

## 風險控制

### 低風險變更
- TensorSpec 移除（使用範圍極小）
- interfaces/ 抽象層清理（大多數未被外部使用）

### 中風險變更
- VDXLSTMManager 統一（需要仔細處理匯入）
- TrafficFeature 簡化（需確保轉換流程正常）

### 風險緩解措施
- 分階段實施，每個 PR 獨立測試
- 添加臨時兼容性別名
- 完整的 CI/CD 測試覆蓋
- 端到端訓練驗證（固定種子測試）

---

## 驗收標準

### 技術驗收
- [ ] 所有 CI 測試通過
- [ ] 無遺留的 `interfaces/tensor_spec.py` 和 `interfaces/vd_manager.py` 引用
- [ ] `feature.py` 使用常數定義，無類別
- [ ] 無單一實現的 ABC 存在
- [ ] 端到端訓練產生相同結果（固定種子）

### 代碼品質
- [ ] 無重複的類別名稱
- [ ] 所有匯入路徑正確
- [ ] 保持公開 API 兼容性（或有明確的遷移指南）

---

## 搜尋替換清單

### TensorSpec 替換
```bash
# 查找使用
rg -n "from.*tensor_spec import TensorSpec"
rg -n "TensorSpec\("

# 替換
from social_xlstm.interfaces.tensor_spec import TensorSpec
→ from social_xlstm.utils.tensor_checks import assert_shape
```

### VDXLSTMManager 替換
```bash
# 查找使用  
rg -n "from.*vd_manager import VDXLSTMManager"

# 替換
from social_xlstm.interfaces.vd_manager import VDXLSTMManager
→ from social_xlstm.models.vd_xlstm_manager import VDXLSTMManager
```

### TrafficFeature 替換
```bash
# 查找使用
rg -n "TrafficFeature\("
rg -n "\.to_dict\(\)"
rg -n "get_field_names\(\)"

# 替換
TrafficFeature(...).to_dict() → make_feature(...)
TrafficFeature.get_field_names() → get_feature_field_names()
```

---

## 下一步行動

1. **確認實施範圍**：用戶確認 PR 優先順序
2. **建立測試基準**：運行現有測試套件確保基準
3. **開始 PR-1**：TensorSpec 移除（最高投資回報率）

**目標**：回歸簡潔、可維護的架構，消除不必要的抽象和複雜性。

---

## 📝 **已完成重構總結**

### ✅ **PR-X: DataModule 統一重構**（已完成）

**完成目標**：整合 TrafficDataModule 和 DistributedTrafficDataModule，統一 API 並消除重複代碼。

**實現效果**：
- **統一 API**：`TrafficDataModule(config)` 自動根據 `config.batch_format` 選擇模式
- **向後兼容**：現有訓練腳本無需修改
- **代碼減少**：淨減少約 60 行，提升維護性

---

## 🔧 **PR-X: DistributedSocialXLSTMModel 配置重構計劃**

### 📋 **問題描述**

當前 `DistributedSocialXLSTMModel` 存在以下配置問題：

1. **參數混亂**：`hidden_dim` 既影響模型又影響 social pooling，語義不清
2. **參數過多**：constructor 接受 8 個分散參數，難以管理和擴展
3. **缺乏層次結構**：模型、social pooling、訓練配置混在一起
4. **已有資源未利用**：代碼庫中已有 `SocialPoolingConfig` 但未被 `DistributedSocialXLSTMModel` 使用

### 🎯 **設計原則** (基於使用者要求)

1. **明確性優於便利性**：所有參數必須明確指定，不提供預設值
2. **快速失敗**：配置不完整立即報錯，不允許模糊狀態
3. **無隱式行為**：不提供自動繼承、預設值或向後兼容性
4. **使用者責任**：使用者必須完全了解所有參數設定

### 🏗️ **重構方案**

#### **階段 1：創建嚴格配置系統**

```python
# 新增 src/social_xlstm/models/distributed_config.py
from dataclasses import dataclass
from typing import Tuple, Literal

# 基於實際 xlstm_pooling.py 實現的類型（不支援 "none"）
ALLOWED_POOL_TYPES: Tuple[str, ...] = ("mean", "max", "weighted_mean")

@dataclass
class SocialPoolingConfig:
    enabled: bool
    pool_type: str  # 只在 enabled=True 時使用，必須是 ALLOWED_POOL_TYPES 之一
    hidden_dim: int
    spatial_radius: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("config.social.enabled must be a boolean")
        
        # 所有參數都必須明確設定，無論是否啟用
        if self.hidden_dim <= 0:
            raise ValueError("config.social.hidden_dim must be > 0")
        if self.spatial_radius <= 0:
            raise ValueError("config.social.spatial_radius must be > 0")
            
        # 只在啟用時驗證 pool_type
        if self.enabled:
            if self.pool_type not in ALLOWED_POOL_TYPES:
                raise ValueError(f"config.social.pool_type must be one of {ALLOWED_POOL_TYPES}, got '{self.pool_type}'")
        # 停用時不驗證 pool_type（因為不會被使用）

@dataclass
class DistributedSocialXLSTMConfig:
    xlstm: TrafficXLSTMConfig
    num_features: int
    prediction_length: int
    learning_rate: float
    enable_gradient_checkpointing: bool
    social: SocialPoolingConfig

    def __post_init__(self) -> None:
        if self.num_features <= 0:
            raise ValueError("config.num_features must be > 0")
        if self.prediction_length <= 0:
            raise ValueError("config.prediction_length must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("config.learning_rate must be > 0")
        if not isinstance(self.enable_gradient_checkpointing, bool):
            raise ValueError("config.enable_gradient_checkpointing must be a boolean")
```

#### **階段 2：重構模型 Constructor**

```python
# 修改 src/social_xlstm/models/distributed_social_xlstm.py
class DistributedSocialXLSTMModel(pl.LightningModule):
    def __init__(self, config: DistributedSocialXLSTMConfig):
        super().__init__()
        
        # 快速失敗驗證
        if not isinstance(config, DistributedSocialXLSTMConfig):
            raise TypeError("config must be an instance of DistributedSocialXLSTMConfig")
        
        self.config = config
        
        # 構建 xLSTM 核心
        self.xlstm = self._build_xlstm(self.config.xlstm, self.config.num_features)
        
        # 條件性構建 social pooling
        self.social_pool = None
        if self.config.social.enabled:
            self.social_pool = self._build_social_pool(
                pool_type=self.config.social.pool_type,
                hidden_dim=self.config.social.hidden_dim,
                radius=self.config.social.spatial_radius
            )
    
    def _build_social_pool(self, pool_type: str, hidden_dim: int, radius: float) -> torch.nn.Module:
        # 使用實際的 XLSTMSocialPoolingLayer 實現
        from ..pooling.xlstm_pooling import XLSTMSocialPoolingLayer
        return XLSTMSocialPoolingLayer(
            hidden_dim=hidden_dim,
            radius=radius,
            pool_type=pool_type,  # 支援: mean, max, weighted_mean
            learnable_radius=False
        )
        
        # 輸出層
        self.output_head = nn.Linear(self._xlstm_out_dim(), self.config.prediction_length)
        
        # 儲存超參數供 Lightning 使用
        self.save_hyperparameters()
    
    # 移除所有舊的參數化 constructor
    # 無向後兼容性
```

#### **階段 3：更新配置文件結構**

```yaml
# cfgs/models/xlstm.yaml 更新
model:
  xlstm:
    input_size: 5  # Taiwan VD，由 datasets/ 覆蓋
    embedding_dim: 64
    hidden_size: 128
    num_blocks: 6
    output_size: 5
    sequence_length: 24
    prediction_length: 12
    # ... 其他 xlstm 參數，全部必須明確指定
    
  distributed_social:
    num_features: 5
    prediction_length: 12
    learning_rate: 0.001
    enable_gradient_checkpointing: true
    
    social:
      enabled: true
      pool_type: "weighted_mean"  # 基於實際實現：mean, max, weighted_mean
      hidden_dim: 64  # 與 xlstm.hidden_size 分離，必須明確指定
      spatial_radius: 2.0
```

```yaml
# cfgs/datasets/pems_bay.yaml 覆蓋範例
model:
  xlstm:
    input_size: 6  # PEMS-BAY 特定
    output_size: 6
  distributed_social:
    num_features: 6
```

#### **階段 4：配置載入器**

```python
# src/social_xlstm/models/config_loader.py
def load_distributed_config(config_dict: Dict[str, Any]) -> DistributedSocialXLSTMConfig:
    """從 YAML 載入配置，要求所有欄位明確指定"""
    
    model_config = config_dict.get("model", {})
    xlstm_config = model_config.get("xlstm", {})
    distributed_config = model_config.get("distributed_social", {})
    
    # 快速失敗檢查
    required_xlstm = {"input_size", "embedding_dim", "hidden_size", "num_blocks", "output_size", 
                      "sequence_length", "prediction_length"}
    missing_xlstm = required_xlstm - xlstm_config.keys()
    if missing_xlstm:
        raise ValueError(f"Missing xlstm config keys: {sorted(missing_xlstm)}")
    
    required_distributed = {"num_features", "prediction_length", "learning_rate", 
                           "enable_gradient_checkpointing", "social"}
    missing_distributed = required_distributed - distributed_config.keys()
    if missing_distributed:
        raise ValueError(f"Missing distributed_social config keys: {sorted(missing_distributed)}")
    
    # 構建配置對象
    xlstm = TrafficXLSTMConfig(**xlstm_config)
    social = SocialPoolingConfig(**distributed_config["social"])
    
    return DistributedSocialXLSTMConfig(
        xlstm=xlstm,
        num_features=distributed_config["num_features"],
        prediction_length=distributed_config["prediction_length"],
        learning_rate=distributed_config["learning_rate"],
        enable_gradient_checkpointing=distributed_config["enable_gradient_checkpointing"],
        social=social
    )
```

### 🔧 **實施步驟**

1. **創建新配置結構**：`distributed_config.py` + 驗證邏輯
2. **重構模型 Constructor**：移除散列參數，只接受 config
3. **更新 YAML 配置**：明確指定所有參數，no defaults
4. **移除舊 API**：無向後兼容，強制遷移
5. **更新訓練腳本**：使用新的配置載入器

### ✅ **預期效果**

- **徹底解決 hidden_dim 歧義**：`xlstm.hidden_size` vs `social.hidden_dim` 完全分離
- **強制配置意識**：使用者必須明確了解所有參數含義
- **簡化模型邏輯**：移除所有隱式行為和預設值推導
- **提高可維護性**：配置結構清晰，擴展容易

### ⚠️ **Breaking Changes**

- **API 不兼容**：所有現有的 `DistributedSocialXLSTMModel(...)` 實例化都需要更新
- **配置文件調整**：所有 YAML 配置需要補充完整參數
- **無遷移路徑**：不提供向後兼容性，強制完全遷移

---

## 🔧 **PR-Y: Social Pooling Spatial-Only 簡化重構計劃**

### 📋 **問題分析**

經 GPT-5 深度分析，發現 `DistributedSocialXLSTMModel` 存在**兩種 Social Pooling 實現**：

1. **XLSTMSocialPoolingLayer (spatial)**：位於 `xlstm_pooling.py`，支援距離權重
2. **SocialPoolingLayer (legacy)**：直接定義在 `distributed_social_xlstm.py:23-37`，簡單平均

**用戶決策**：**只使用 Spatial Mode**，完全移除 legacy 實現。

### 🎯 **激進簡化策略**

既然確定只用 Spatial Mode，可進行**大幅度簡化**：

#### **核心變更**

```python
# 當前複雜邏輯 (第81-92行)
if enable_spatial_pooling:
    self.social_pooling = XLSTMSocialPoolingLayer(...)  # 保留
else:
    self.social_pooling = SocialPoolingLayer(...)       # 完全移除

# 簡化後邏輯
if config.social.enabled:
    self.social_pooling = XLSTMSocialPoolingLayer(...)
else:
    self.social_pooling = None  # 不創建任何 pooling
```

#### **參數語義轉換**

- **舊**: `enable_spatial_pooling: bool` (選擇實現類型)
- **新**: `social.enabled: bool` (是否啟用社會池化)

### 🏗️ **實施步驟**

#### **階段 1：移除 Legacy 實現**
1. **完全刪除** `distributed_social_xlstm.py:23-37` 的 `SocialPoolingLayer` 類
2. **風險評估**：低風險 - 只在單一文件中定義，無外部依賴

#### **階段 2：簡化配置系統**
```python
@dataclass
class SocialPoolingConfig:
    enabled: bool                    # 是否啟用社會池化
    radius: float                    # 空間半徑 (meters)
    aggregation: Literal["mean", "max", "weighted_mean"]  # 聚合方式
    hidden_dim: int                  # 隱藏維度
    
    # 移除不需要的參數：
    # - mode: 不需要選擇，只有 spatial
    # - coordinate_system: 簡化為預設 euclidean
```

#### **階段 3：重構模型邏輯**
```python
def __init__(self, config: DistributedSocialXLSTMConfig):
    # ... 其他初始化
    
    # 簡化的社會池化邏輯
    if config.social.enabled:
        self.social_pooling = XLSTMSocialPoolingLayer(
            hidden_dim=config.social.hidden_dim,
            radius=config.social.radius,
            pool_type=config.social.aggregation,
            learnable_radius=False
        )
    else:
        self.social_pooling = None

def forward(self, vd_inputs, neighbor_map=None, positions=None):
    # ... VD 處理
    individual_hidden_states = self.vd_manager(vd_inputs)
    
    if self.social_pooling is None:
        # 完全跳過社會池化，使用零向量
        social_contexts = {
            vd_id: torch.zeros_like(hidden[:, -1, :])
            for vd_id, hidden in individual_hidden_states.items()
        }
    else:
        # 使用空間感知池化
        social_contexts = self.social_pooling(
            agent_hidden_states=individual_hidden_states,
            agent_positions=positions,
            target_agent_ids=list(vd_inputs.keys())
        )
```

### 🗂️ **影響文件清單**

#### **需要修改的文件**
1. **主模型文件**：`src/social_xlstm/models/distributed_social_xlstm.py`
   - 移除 `SocialPoolingLayer` 類定義 (第23-37行)
   - 簡化構造函數邏輯 (第81-92行)
   - 更新 forward 方法處理 `social_pooling = None`

2. **配置映射**：`src/social_xlstm/err_impl/config/parameter_mapper.py`
   - 更新 `map_social_config_to_training_args()` 邏輯
   - 移除 spatial/graph 分支選擇

3. **訓練腳本** (基於 grep 結果)：
   - `scripts/train/with_social_pooling/err_imp/train_*.py`
   - 適配新的配置參數名稱

#### **配置文件結構更新**
```yaml
# 簡化的配置結構
model:
  distributed_social:
    social:
      enabled: true  # 取代 enable_spatial_pooling
      radius: 2.0
      aggregation: "weighted_mean"
      hidden_dim: 64
```

### ⚡ **預期效益**

#### **代碼簡化**
- **移除整個 legacy 分支**：~15行 (SocialPoolingLayer 類)
- **簡化構造邏輯**：~10行 (移除分支選擇)
- **配置參數減少**：從雙重實現選擇變為單純啟用/禁用

#### **邏輯清晰化**
- **消除語義混亂**：`enable_spatial_pooling` → `social.enabled`
- **統一返回格式**：只需處理一種 pooling 接口
- **降低認知負擔**：只需理解一種實現

#### **維護成本降低**
- **單一實現路徑**：不需要維護兩套邏輯
- **測試簡化**：只需測試 enabled/disabled 兩種情況
- **文檔簡化**：不需要解釋兩種模式差異

### ⚠️ **風險評估**

#### **低風險因素**
- **Legacy 實現隔離**：只在單一文件中定義
- **Spatial 實現成熟**：`XLSTMSocialPoolingLayer` 已經穩定
- **影響範圍可控**：主要是參數名稱變更

#### **需要注意的點**
- **參數映射更新**：確保 `parameter_mapper.py` 正確處理新參數
- **enabled=False 行為**：確保 `social_pooling = None` 時 forward 正常工作
- **位置數據要求**：Spatial mode 需要 positions 參數，確保調用方提供

### 🔧 **實施順序**

1. **創建新配置結構** (`distributed_config.py`)
2. **移除 legacy 實現** (delete SocialPoolingLayer)
3. **簡化構造邏輯** (remove branching)
4. **更新 forward 邏輯** (handle social_pooling=None)
5. **適配配置映射** (parameter_mapper.py)
6. **測試驗證** (enabled/disabled 場景)

### ✅ **驗收標準**

- [ ] `SocialPoolingLayer` 完全移除
- [ ] `enable_spatial_pooling` 參數不再存在
- [ ] `social.enabled=False` 時模型正常運行
- [ ] `social.enabled=True` 時使用 spatial pooling
- [ ] 所有配置文件更新為新格式
- [ ] 訓練腳本適配新參數結構

---

**目標**：回歸簡潔、可維護的架構，消除不必要的抽象和複雜性。