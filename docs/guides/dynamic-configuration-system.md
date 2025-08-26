# Dynamic Configuration System Usage Guide

## 概述

新的動態配置系統將模型架構、社會聚合、VD模式和訓練參數分離到獨立的YAML文件中，提供清晰的關注點分離和高度可重用性。

**主要優勢:**
- ✅ **參數減少70-75%**: CLI參數從22-25個減少到4-5個
- ✅ **模組化設計**: 不同配置層可獨立組合
- ✅ **類型安全**: 每個模型有專用配置類驗證
- ✅ **向下兼容**: 保持對現有CLI參數的支援

## 配置文件結構

```
cfgs/
├── models/           # 純模型架構配置
│   ├── lstm.yaml
│   └── xlstm.yaml
├── social_pooling/   # 社會聚合配置
│   ├── off.yaml
│   ├── weighted_mean.yaml
│   ├── weighted_sum.yaml
│   └── attention.yaml
├── vd_modes/        # VD模式配置
│   ├── single.yaml
│   └── multi.yaml
└── training/        # 訓練超參數
    └── default.yaml
```

## 使用方法

### 推薦方法: 使用 snakemake_warp.py 合併配置

```bash
# Social xLSTM with attention pooling
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/attention.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd

# 切換聚合方法只需更改一個配置檔案
# Weighted mean pooling
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/weighted_mean.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd

# Baseline without spatial pooling
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/off.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd
```

### 方法2: 直接在訓練腳本中使用

```bash
# 使用多個配置文件
python scripts/train/with_social_pooling/train_distributed_social_xlstm.py \
  --model-config cfgs/models/xlstm.yaml \
  --social-config cfgs/social_pooling/attention.yaml \
  --vd-config cfgs/vd_modes/multi.yaml \
  --training-config cfgs/training/default.yaml \
  --data_path blob/dataset/pre-processed/h5/traffic_features_dev.h5

# 使用單一合併配置文件  
python scripts/train/with_social_pooling/train_distributed_social_xlstm.py \
  --config-file merged_config.yaml \
  --data_path blob/dataset/pre-processed/h5/traffic_features_dev.h5
```

## 配置範例

### 模型配置 (xlstm.yaml)

```yaml
model:
  name: "TrafficXLSTM"
  xlstm:
    input_size: 3
    embedding_dim: 64
    num_blocks: 4
    slstm_at: [1, 3]
    dropout: 0.5
    sequence_length: 5
    prediction_length: 1
```

### 社會聚合配置 (attention.yaml)

```yaml
social:
  enabled: true
  pooling_radius: 2500.0
  max_neighbors: 10
  distance_metric: "euclidean"
  weighting_function: "gaussian"
  aggregation_method: "attention"
  coordinate_system: "projected"
```

### VD模式配置 (multi.yaml)

```yaml
vd:
  mode: "multi"
  count: 3
  selected_vdids:
    - VD-28-0740-000-001
    - VD-11-0020-008-001  
    - VD-13-0660-000-002
```

### 訓練配置 (default.yaml)

```yaml
training:
  epochs: 10
  batch_size: 16
  learning_rate: 0.0005
  weight_decay: 0.01
  optimizer: "adam"
  scheduler_type: "reduce_on_plateau"
```

## 消融研究支援

新配置系統特別適合消融研究，只需更換特定配置文件：

```bash
# 比較不同聚合方法
for method in off weighted_mean weighted_sum attention; do
  python workflow/snakemake_warp.py \
    --configfile cfgs/models/xlstm.yaml \
    --configfile cfgs/social_pooling/${method}.yaml \
    --configfile cfgs/vd_modes/multi.yaml \
    --configfile cfgs/training/default.yaml \
    train_ablation_${method}
done

# 比較不同模型架構
for model in lstm xlstm; do
  python workflow/snakemake_warp.py \
    --configfile cfgs/models/${model}.yaml \
    --configfile cfgs/social_pooling/attention.yaml \
    --configfile cfgs/vd_modes/multi.yaml \
    --configfile cfgs/training/default.yaml \
    train_model_${model}
done
```

## Python API 使用

```python
from social_xlstm.config import load_config_from_paths

# 載入並合併配置
config_files = [
    'cfgs/models/xlstm.yaml',
    'cfgs/social_pooling/attention.yaml',
    'cfgs/vd_modes/multi.yaml',
    'cfgs/training/default.yaml'
]

merged_config = load_config_from_paths(config_files)

print(f"Model: {merged_config.model_name}")
print(f"Effective input size: {merged_config.effective_input_size}")
print(f"Social pooling: {merged_config.social_config.get('enabled', False)}")
```

## 配置合併規則

1. **優先順序**: 基礎模型配置 < 社會聚合配置 < VD模式配置 < 訓練配置
2. **合併語義**: 
   - 字典: 深度合併 (後者覆蓋前者)
   - 列表/純量: 完全替換
3. **參數自動推導**: 
   - `effective_input_size = model.input_size × vd.count`
4. **一致性驗證**: 自動檢查配置組合的有效性

## 遷移指南

從舊的CLI參數系統遷移到新配置系統:

**舊方式 (25個參數):**
```bash
python train.py --model_type xlstm --hidden_size 128 --num_blocks 4 \
  --slstm_at 1 3 --enable_spatial_pooling --spatial_radius 2.5 \
  --epochs 50 --batch_size 16 --learning_rate 0.001 # ... 更多參數
```

**新方式 (4個文件):**
```bash
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/attention.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/production.yaml \
  train_model
```

**參數減少效果**: CLI參數從22-25個減少至4-5個（**70-75%減少**）

## 故障排除

### 常見錯誤

1. **配置文件不匹配**
   ```
   Error: Model configuration must contain 'xlstm' section
   ```
   確保模型配置文件使用正確的嵌套結構。

2. **參數類型錯誤** 
   ```
   Error: TrafficXLSTMConfig.__init__() got unexpected keyword argument 'backend'
   ```
   檢查YAML文件中的參數名稱是否與配置類匹配。

3. **社會聚合和單VD衝突**
   ```
   Error: Social pooling requires multi-VD mode
   ```
   社會聚合需要多VD模式，使用 `cfgs/vd_modes/multi.yaml`。

### 配置驗證

```python
# 檢查配置一致性
from social_xlstm.config import DynamicModelConfigManager

try:
    config = DynamicModelConfigManager.from_yaml_files(config_files)
    print("✅ Configuration is valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

## 開發和擴展

### 添加新模型類型

1. 創建新的配置類 (繼承自dataclass)
2. 在 `registry.py` 中註冊模型
3. 創建對應的YAML配置模板

### 添加新聚合方法

1. 在 `social_pooling.py` 中實現聚合邏輯
2. 創建對應的YAML配置文件
3. 更新參數映射器 (如需要)

這個新的配置系統為Social-xLSTM項目提供了現代化的配置管理方案，大幅簡化了實驗流程並提高了研究效率。