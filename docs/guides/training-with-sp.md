# Social-xLSTM Training Guide

本指南說明如何使用分散式 Social-xLSTM 架構進行空間社會聚合訓練。

## 📁 系統架構

```
Social-xLSTM 分散式架構：
├── DistributedSocialXLSTMModel     # 分散式社會 xLSTM 模型
├── SpatialPooling                  # 空間聚合模組
├── DynamicConfigManager            # 動態配置管理
└── snakemake_warp.py              # 配置整合工具
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 激活 conda 環境
conda activate social_xlstm

# 確認在專案根目錄
cd /path/to/Social-xLSTM
```

### 2. 使用動態配置系統

新的配置系統支援四層 YAML 配置：

```bash
# 使用動態配置訓練
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/attention.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  --data_path blob/dataset/pre-processed/h5/traffic_features_default.h5 \
  --epochs 50 \
  --batch_size 16
```

### 3. 直接使用訓練腳本

```bash
# 分散式 Social-xLSTM 訓練
python scripts/train/with_social_pooling/train_distributed_social_xlstm.py \
  --data_path blob/dataset/pre-processed/h5/traffic_features_default.h5 \
  --enable_spatial_pooling \
  --aggregation_method attention \
  --spatial_radius 2.0 \
  --epochs 50 \
  --batch_size 16 \
  --experiment_name social_xlstm_attention
```

## 🌐 社會聚合配置

### 可用的聚合方法

1. **weighted_mean**: 加權平均聚合（行歸一化）
2. **weighted_sum**: 加權求和聚合（無歸一化）
3. **attention**: 注意力聚合機制（Softmax 歸一化）

### 配置範例

**注意力聚合配置** (`cfgs/social_pooling/attention.yaml`):
```yaml
social:
  enabled: true
  pooling_radius: 1000.0
  max_neighbors: 8
  distance_metric: "euclidean"
  weighting_function: "gaussian"
  aggregation_method: "attention"
  coordinate_system: "projected"
```

**關閉社會聚合** (`cfgs/social_pooling/off.yaml`):
```yaml
social:
  enabled: false
```

## 📊 實驗架構

### 1. 基礎模型比較
- **TrafficLSTM**: 傳統 LSTM 基準模型
- **TrafficXLSTM**: 擴展 LSTM（無社會聚合）
- **DistributedSocialXLSTM**: 完整的社會 xLSTM

### 2. 社會聚合比較
- 無聚合 vs 三種聚合方法的性能比較
- 不同空間半徑的影響分析
- 鄰居數量的最佳化研究

### 3. 關鍵創新點

1. **分散式架構**: 每個 VD 維持獨立 xLSTM 實例
2. **空間聚合**: 基於地理座標的社會特徵融合
3. **動態配置**: 四層 YAML 配置系統
4. **參數映射**: 舊新系統的向後兼容性

## 📊 輸出結果

訓練完成後，結果保存在 `blob/experiments/` 目錄：

```
blob/experiments/social_xlstm_attention/
├── config.json              # 完整配置
├── training_history.json    # 訓練歷史
├── best_model.pt           # 最佳模型權重
└── plots/                  # 訓練曲線圖
```

## 🔧 進階使用

### 使用動態配置系統 (推薦)

```bash
# 訓練 Social-xLSTM with attention pooling
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/attention.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd

# 切換不同的聚合方法 (只需更改一個配置檔案)
# Attention mechanism
python workflow/snakemake_warp.py ... --configfile cfgs/social_pooling/attention.yaml ...

# Weighted mean pooling  
python workflow/snakemake_warp.py ... --configfile cfgs/social_pooling/weighted_mean.yaml ...

# Weighted sum pooling
python workflow/snakemake_warp.py ... --configfile cfgs/social_pooling/weighted_sum.yaml ...

# No spatial pooling (baseline)
python workflow/snakemake_warp.py ... --configfile cfgs/social_pooling/off.yaml ...
```

### 傳統 Snakemake 方式 (向後兼容)

```bash
# 使用現有配置檔案
snakemake train_social_xlstm_multi_vd --configfile cfgs/snakemake/dev.yaml --cores 2
```

### 參數調整建議

- **spatial_radius**: 1.0-5.0 公里（城市環境）
- **max_neighbors**: 4-12 個鄰居
- **aggregation_method**: 從 weighted_mean 開始測試
- **batch_size**: 8-32（依 GPU 記憶體調整）

## 🚨 故障排除

### 常見問題

1. **記憶體不足**: 減少 `batch_size` 或 `max_neighbors`
2. **配置衝突**: 使用 `snakemake_warp.py` 進行配置驗證
3. **參數不匹配**: 檢查 `aggregation_method` vs 舊版 `pool_type`

### 除錯指令

```bash
# 驗證配置
python -c "from social_xlstm.config import load_config_from_paths; print('配置系統正常')"

# 測試參數映射
python -c "from social_xlstm.config import ParameterMapper; print(ParameterMapper().POOL_TYPE_TO_AGGREGATION_METHOD)"
```

## 📚 參考資料

- [數學規格](../concepts/mathematical-specifications.md)
- [API 參考](../reference/api-reference.md)
- [配置系統文檔](../concepts/configuration-system.md)