# Post-Fusion Social Pooling Training

本目錄包含 Post-Fusion 策略的 Social Pooling 訓練腳本和工具。

## 📁 文件結構

```
post_fusion/
├── common.py              # Post-Fusion 專用工具函數
├── train_single_vd.py     # 單 VD Post-Fusion 訓練腳本
├── test_integration.py    # 整合測試腳本
└── README.md             # 本說明文件
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 激活 conda 環境
conda activate social_xlstm

# 確認在專案根目錄
cd /path/to/Social-xLSTM
```

### 2. 數據準備

確保有以下文件：
- HDF5 數據文件（通過數據預處理生成）
- VD 座標文件（JSON 格式）

示例座標文件格式：
```json
{
  "VD-C1T0440-N": [121.5654, 25.0330],
  "VD-C1T0441-S": [121.5643, 25.0315],
  "VD-C1T0442-N": [121.5632, 25.0345]
}
```

### 3. 運行整合測試

```bash
cd scripts/train/with_social_pooling/post_fusion

python test_integration.py \
  --coordinate_data data/sample_vd_coordinates.json \
  --select_vd_id VD-C1T0440-N \
  --scenario urban
```

### 4. 訓練模型

#### Social-LSTM (Post-Fusion)
```bash
python train_single_vd.py \
  --model_type lstm \
  --select_vd_id VD-C1T0440-N \
  --coordinate_data data/sample_vd_coordinates.json \
  --scenario urban \
  --epochs 2 \
  --batch_size 16
```

#### Social-xLSTM (Post-Fusion)
```bash
python train_single_vd.py \
  --model_type xlstm \
  --select_vd_id VD-C1T0440-N \
  --coordinate_data data/sample_vd_coordinates.json \
  --scenario highway \
  --epochs 2 \
  --batch_size 16
```

## ⚙️ 配置選項

### 場景預設 (`--scenario`)

- **urban**: 城市環境
  - `pooling_radius`: 500m
  - `max_neighbors`: 12
  - `weighting_function`: gaussian
  
- **highway**: 高速公路環境
  - `pooling_radius`: 2000m
  - `max_neighbors`: 5
  - `weighting_function`: exponential
  
- **mixed**: 混合環境（預設）
  - `pooling_radius`: 1200m
  - `max_neighbors`: 8
  - `weighting_function`: linear

### 自定義參數

```bash
python train_single_vd.py \
  --model_type lstm \
  --select_vd_id VD-C1T0440-N \
  --coordinate_data data/coordinates.json \
  --pooling_radius 1500 \
  --max_neighbors 10 \
  --distance_metric euclidean \
  --weighting_function gaussian \
  --aggregation_method weighted_mean
```

## 🔧 Post-Fusion 架構

Post-Fusion 策略的數據流：

```
VD 輸入 → 基礎模型 (LSTM/xLSTM) → 個體特徵
                                      ↓
座標數據 → Social Pooling → 空間特徵 → Gated Fusion → 預測輸出
```

### 核心組件

1. **基礎模型**: TrafficLSTM 或 TrafficXLSTM
2. **Social Pooling**: 座標驅動的空間聚合
3. **Gated Fusion**: 智能特徵融合層
4. **SocialTrafficModel**: 統一包裝器

## 📊 輸出結果

訓練完成後，結果保存在 `blob/experiments/` 目錄：

```
blob/experiments/social_lstm_post_fusion_urban/
├── config.json              # 完整配置
├── social_config.json       # Social Pooling 配置
├── coordinate_info.json     # 座標信息
├── best_model.pt           # 最佳模型權重
├── training_history.json   # 訓練歷史
└── plots/                  # 訓練圖表
```

## 🚨 常見問題

### 1. 環境錯誤
```
ModuleNotFoundError: No module named 'torch'
```
**解決**: 確保激活了正確的 conda 環境
```bash
conda activate social_xlstm
```

### 2. 座標文件錯誤
```
FileNotFoundError: Coordinate data file not found
```
**解決**: 檢查座標文件路徑是否正確，相對於專案根目錄

### 3. VD ID 不匹配
```
Selected VD 'XXX' not found in coordinate data
```
**解決**: 確保選擇的 VD ID 在座標文件中存在

### 4. 記憶體不足
```
CUDA out of memory
```
**解決**: 
- 減少 `batch_size`
- 減少 `max_neighbors`
- 使用 `--mixed_precision`

## 🔗 相關文件

- **設計文檔**: `docs/explanation/social-pooling-design.md`
- **數學規格**: `docs/reference/mathematical-specifications.md`
- **基礎訓練**: `scripts/train/without_social_pooling/`
- **數據預處理**: `scripts/dataset/`

## 📈 性能預期

與基礎 LSTM 相比，Post-Fusion Social Pooling 預期：
- MAE/RMSE 改善 > 5%
- 記憶體增長 < 50%
- 訓練時間增長 < 30%

## 🤝 支援

如有問題，請：
1. 檢查 `logs/` 目錄中的詳細日誌
2. 運行 `test_integration.py` 進行診斷
3. 參考相關文檔和 ADR 決策記錄