# Social-xLSTM 完整模組文檔

## 概述

本文檔提供 `src/social_xlstm` 所有模組的完整分析，包括功能描述、設計評估、使用建議和改進方向。

## 模組架構總覽

```
src/social_xlstm/
├── dataset/          # 數據處理模組
├── models/           # 模型實現模組
├── training/         # 訓練系統模組
├── evaluation/       # 評估框架模組
├── utils/           # 工具函數模組
└── visualization/   # 可視化模組
```

## 1. Dataset 模組

### 1.1 TrafficFeature (`feature.py`)
- **功能**: 標準化交通特徵數據結構
- **狀態**: ✅ 完成且穩定
- **評估**: 設計清晰，類型安全
- **改進**: 可添加數據驗證

### 1.2 數據載入器 (`loader.py`)
- **功能**: PyTorch Lightning 數據載入解決方案
- **狀態**: ✅ 功能完整
- **問題**: 文件過長（430行），正規化器共享問題
- **改進**: 分拆文件，修復正規化器處理

### 1.3 HDF5 工具 (`h5_utils.py`)
- **功能**: 高效的 HDF5 數據存儲和讀取
- **狀態**: ✅ 功能完整
- **問題**: 文件過長（499行），功能過於集中
- **改進**: 分離 Converter 和 Reader

### 1.4 JSON 工具 (`json_utils.py`)
- **功能**: 交通數據 JSON 格式處理
- **狀態**: ✅ 功能完整
- **評估**: 完整的數據結構定義
- **改進**: 可考慮使用 Pydantic 增強驗證

### 1.5 輔助工具
- **XML 工具**: 基礎的 XML 到 JSON 轉換
- **ZIP 工具**: 壓縮檔案處理和時間解析

## 2. Models 模組

### 2.1 TrafficLSTM (`lstm.py`)
- **功能**: 統一的 LSTM 實現（ADR-0002）
- **狀態**: ✅ 架構清理完成
- **評估**: 統一設計，消除重複代碼
- **創新**: 支援單VD和多VD模式

### 2.2 SocialPoolingLayer (`social_pooling.py`)
- **功能**: 座標驅動的空間關係學習
- **狀態**: ✅ 核心創新實現
- **評估**: 無需拓撲結構的空間學習
- **改進**: 座標轉換效率優化

### 2.3 SocialXLSTM (`social_xlstm.py`)
- **功能**: 完整的 Social-xLSTM 模型
- **狀態**: ✅ 架構設計完成
- **評估**: 正確實現 Social LSTM 概念
- **待完成**: xLSTM 整合（ADR-0101）

## 3. Training 模組

### 3.1 Trainer (`trainer.py`)
- **功能**: 統一的訓練系統
- **狀態**: ✅ 完整的訓練管線
- **評估**: 專業的配置管理和檢查點系統
- **特色**: 支援混合精度、早停、調度器

## 4. Evaluation 模組

### 4.1 ModelEvaluator (`evaluator.py`)
- **功能**: 模型評估和指標計算
- **狀態**: ✅ 基本功能完成
- **評估**: 支援訓練集和驗證集評估
- **指標**: MAE, MSE, RMSE, MAPE, R²

## 5. Utils 模組

### 5.1 CoordinateSystem (`spatial_coords.py`)
- **功能**: 座標系統轉換和空間計算
- **狀態**: ✅ 完整實現
- **評估**: 支援墨卡托投影和雙向轉換
- **特色**: 豐富的工廠方法和距離計算

### 5.2 輔助工具
- **convert_coords.py**: 簡單的座標轉換工具
- **graph.py**: VD 座標可視化
- **pure_text.py**: 文本文件載入

## 6. Visualization 模組

### 6.1 TrafficResultsPlotter (`model.py`)
- **功能**: 交通預測結果可視化
- **狀態**: ✅ 完整的可視化系統
- **評估**: 支援多種圖表類型
- **特色**: 評估儀表板和時間序列對比

## 冗餘功能和複雜度問題分析

### 1. 代碼重複問題

#### 已解決的重複
- ✅ **LSTM 實現統一**: 5個重複實現已統一為1個
- ✅ **訓練腳本重構**: 減少48%代碼重複

#### 仍存在的重複
- ⚠️ **座標轉換**: `convert_coords.py` 與 `spatial_coords.py` 功能重複
- ⚠️ **可視化**: `graph.py` 與 `visualization/model.py` 部分功能重複

### 2. 複雜度問題

#### 文件過長問題
- ❌ **loader.py**: 430行，功能過於集中
- ❌ **h5_utils.py**: 499行，Converter 和 Reader 應分離
- ❌ **trainer.py**: 673行，但結構合理，可接受

#### 性能問題
- ⚠️ **座標轉換**: 每次前向傳播重複計算
- ⚠️ **批次處理**: 社交池化中的迴圈處理

### 3. 功能多餘問題

#### 低價值工具
- ❓ **pure_text.py**: 功能過於簡單，可能不必要
- ❓ **convert_coords.py**: 被 `spatial_coords.py` 完全取代

#### 過度工程化
- ⚠️ **Social Pooling**: 網格劃分複雜度可能過高
- ⚠️ **配置類**: 部分配置選項可能很少使用

## 改進建議

### 1. 架構重構 (P0)

#### 文件分拆
```
# 當前問題
loader.py (430行) → 功能過於集中

# 建議分拆
├── dataset/
│   ├── config.py       # TrafficDatasetConfig
│   ├── processor.py    # TrafficDataProcessor
│   ├── timeseries.py   # TrafficTimeSeries
│   └── datamodule.py   # TrafficDataModule
```

#### 功能整合
```
# 移除重複
- 刪除 convert_coords.py，統一使用 spatial_coords.py
- 整合 graph.py 功能到 visualization/model.py
- 清理 pure_text.py，移至共用工具
```

### 2. 性能優化 (P1)

#### 座標轉換優化
```python
# 問題：每次前向傳播重複計算
# 解決：預計算相對座標矩陣
class SocialPoolingLayer(nn.Module):
    def __init__(self, config, vd_coordinates):
        # 預計算所有VD對的相對座標
        self.relative_coords_cache = self._precompute_relative_coords(vd_coordinates)
```

#### 批次處理優化
```python
# 問題：迴圈處理效率低
# 解決：向量化操作
def aggregate_features(self, neighbor_features, grid_assignments, distance_weights):
    # 使用 torch.scatter_add 等向量化操作
```

### 3. 代碼品質 (P2)

#### 測試覆蓋率
- 目前：<30%
- 目標：>70%
- 重點：核心模型和數據處理

#### 文檔完善
- 添加更多使用示例
- 完善 API 文檔
- 增加最佳實踐指南

### 4. 功能完善 (P3)

#### xLSTM 整合
```python
# 當前：使用標準 LSTM 占位符
# 目標：整合真正的 xLSTM (sLSTM + mLSTM)
class VDxLSTM(nn.Module):
    def __init__(self, config):
        from xlstm import sLSTM, mLSTM  # 真正的 xLSTM 實現
```

## 使用指南和最佳實踐

### 1. 開發工作流程

#### 快速開始
```bash
# 1. 環境設置
conda activate social_xlstm
pip install -e .

# 2. 數據準備
python -c "from social_xlstm.dataset.h5_utils import create_traffic_hdf5; create_traffic_hdf5('data/json', 'data/traffic.h5')"

# 3. 模型訓練
python scripts/train/train_single_vd.py
```

#### 核心開發模式
```python
# 1. 數據載入
from social_xlstm.dataset.loader import TrafficDataModule, TrafficDatasetConfig

config = TrafficDatasetConfig(hdf5_path="data/traffic.h5")
data_module = TrafficDataModule(config)

# 2. 模型創建
from social_xlstm.models.lstm import TrafficLSTM

model = TrafficLSTM.create_single_vd_model(input_size=3, hidden_size=128)

# 3. 訓練
from social_xlstm.training.trainer import Trainer, TrainingConfig

trainer = Trainer(model, TrainingConfig(), data_module.train_dataloader())
trainer.train()
```

### 2. 性能優化建議

#### 數據載入優化
```python
# 使用合適的批次大小和工作進程
config = TrafficDatasetConfig(
    batch_size=64,      # 根據GPU記憶體調整
    num_workers=4,      # 根據CPU核心數調整
    pin_memory=True     # GPU加速
)
```

#### 模型優化
```python
# 使用混合精度訓練
training_config = TrainingConfig(
    mixed_precision=True,
    gradient_clip_value=1.0
)
```

### 3. 實驗管理

#### 配置管理
```python
# 使用配置驅動的實驗
experiment_config = {
    "model": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2
    },
    "training": {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32
    }
}
```

#### 結果追蹤
```python
# 使用內建的實驗追蹤
trainer = Trainer(model, training_config, train_loader)
history = trainer.train()  # 自動保存結果和模型
```

### 4. 調試和監控

#### 日誌配置
```python
import logging
logging.basicConfig(level=logging.INFO)

# 啟用詳細日誌
logger = logging.getLogger('social_xlstm')
logger.setLevel(logging.DEBUG)
```

#### 性能監控
```python
# 使用模型信息檢查
model_info = model.get_model_info()
print(f"Parameters: {model_info['total_parameters']}")
print(f"Memory: {model_info['model_size_mb']:.2f} MB")
```

## 總結

Social-xLSTM 專案具有完整的模組架構和清晰的設計理念。主要成就包括：

1. **架構清理成功**: 統一了 LSTM 實現，減少了代碼重複
2. **創新實現**: 座標驅動的 Social Pooling 是核心創新
3. **專業工具**: 完整的訓練、評估和可視化系統
4. **良好基礎**: 為 xLSTM 整合提供了穩固基礎

主要改進方向：
1. **文件重構**: 分拆過長的文件
2. **性能優化**: 座標轉換和批次處理
3. **xLSTM 整合**: 完成核心技術目標
4. **測試覆蓋**: 提升代碼品質

專案已具備進行核心研究的所有技術基礎，可以專注於 Social-xLSTM 的創新研究和實驗驗證。