# LSTM 使用指南

本指南說明如何使用 Social-xLSTM 專案中的統一 LSTM 模型 (`TrafficLSTM`)。

## 📋 目錄

1. [快速開始](#快速開始)
2. [核心概念](#核心概念)
3. [基本使用](#基本使用)
4. [常見問題](#常見問題)

## 🚀 快速開始

### 基本訓練

```bash
# 單VD LSTM 訓練
python scripts/train/without_social_pooling/train_single_vd.py

# 多VD LSTM 訓練（空間關係）
python scripts/train/without_social_pooling/train_multi_vd.py

# 獨立多VD LSTM 訓練（基準比較）
python scripts/train/without_social_pooling/train_independent_multi_vd.py

# 使用 Snakemake 執行完整流程
snakemake --configfile cfgs/snakemake/dev.yaml --cores 4
```

## 🧠 核心概念

### 什麼是 VD (Vehicle Detector)？
VD 是路邊的車流監測器，每個VD提供以下數據：
- **volume**: 車流量
- **speed**: 平均車速
- **occupancy**: 車道佔用率

### 數據格式
- **輸入**: 時間序列交通數據
  - 3D 張量 (batch, sequence, features) - 單VD模式
  - 4D 張量 (batch, sequence, vd_count, features) - 多VD模式
- **輸出**: 預測的交通狀況
- **特徵**: 流量、速度、佔用率

### 訓練器類型
- **SingleVDTrainer**: 單VD專用，處理3D數據
- **MultiVDTrainer**: 多VD空間關係，處理4D數據
- **IndependentMultiVDTrainer**: 獨立多VD，提取單VD進行訓練

## 📖 基本使用

### 1. 數據準備

```bash
# 完整數據處理流程
snakemake --cores 4

# 或手動處理
python scripts/dataset/pre-process/create_h5_file.py \
  --source_dir blob/dataset/unzip_to_json \
  --output_path blob/dataset/pre-processed/h5/traffic_features.h5
```

### 2. 模型訓練

#### 基本 LSTM 訓練

```python
# 統一 LSTM 實現
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.dataset import TrafficDatasetConfig, TrafficDataModule
from social_xlstm.training.without_social_pooling import SingleVDTrainer

# 創建數據模組
data_module = TrafficDataModule(
    data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
    batch_size=32,
    sequence_length=12
)

# 創建模型
model = TrafficLSTM(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3
)

# 使用專業化訓練器
trainer = SingleVDTrainer(
    model=model,
    data_module=data_module,
    learning_rate=0.001,
    experiment_name="lstm_single_vd"
)

# 訓練
recorder = trainer.train()
```

#### 多VD LSTM 訓練

```python
from social_xlstm.training.without_social_pooling import MultiVDTrainer

# 多VD模型（支援空間關係）
model = TrafficLSTM(
    input_size=3,
    hidden_size=256,  # 較大的隱藏層
    num_layers=2,
    output_size=3,
    multi_vd_mode=True  # 啟用多VD模式
)

# 使用多VD訓練器
trainer = MultiVDTrainer(
    model=model,
    data_module=data_module,
    learning_rate=0.001,
    experiment_name="lstm_multi_vd"
)

# 訓練
recorder = trainer.train()
```

#### 獨立多VD LSTM 訓練

```python
from social_xlstm.training.without_social_pooling import IndependentMultiVDTrainer

# 單VD模型（用於獨立訓練）
model = TrafficLSTM(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    multi_vd_mode=False  # 單VD模式
)

# 使用獨立多VD訓練器
trainer = IndependentMultiVDTrainer(
    model=model,
    data_module=data_module,
    learning_rate=0.001,
    experiment_name="lstm_independent_multi_vd"
)

# 訓練
recorder = trainer.train()
```

### 3. 模型評估

```python
from social_xlstm.evaluation.evaluator import ModelEvaluator

# 創建評估器
evaluator = ModelEvaluator()

# 評估模型
metrics = evaluator.evaluate(model, test_data)
print(f"MAE: {metrics['mae']:.4f}")
print(f"MSE: {metrics['mse']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAPE: {metrics['mape']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### 4. 訓練記錄與分析

```python
from social_xlstm.training.recorder import TrainingRecorder
from social_xlstm.visualization.training_visualizer import TrainingVisualizer

# 載入訓練記錄
recorder = TrainingRecorder.load("blob/experiments/lstm_single_vd/training_record.json")

# 獲取訓練總結
summary = recorder.get_training_summary()
print(f"總epochs: {summary['total_epochs']}")
print(f"最佳驗證損失: {summary['best_val_loss']:.4f}")
print(f"最佳epoch: {summary['best_epoch']}")

# 生成視覺化
visualizer = TrainingVisualizer()
visualizer.plot_training_dashboard(recorder, "training_dashboard.png")
```

## ❓ 常見問題

### Q1: 如何處理缺失數據？
使用 `TrafficDataProcessor` 自動處理缺失值：
```python
from social_xlstm.dataset.core import TrafficDataProcessor

processor = TrafficDataProcessor()
processed_data = processor.process(raw_data)
```

### Q2: 如何選擇模型參數？
- **單VD模式**:
  - `hidden_size`: 64-128
  - `num_layers`: 2-3
  - `batch_size`: 32-64
- **多VD模式**:
  - `hidden_size`: 128-256（較大）
  - `num_layers`: 2-3
  - `batch_size`: 16-32（較小）
- **通用參數**:
  - `sequence_length`: 12 個時間步（12 分鐘）
  - `learning_rate`: 0.001

### Q3: 訓練時間過長怎麼辦？
- 減少批次大小
- 使用 GPU 加速
- 啟用混合精度：`--mixed_precision`
- 減少數據量進行測試

### Q4: 如何選擇訓練器？
- **SingleVDTrainer**: 單點預測，快速訓練
- **MultiVDTrainer**: 空間關係學習，準備Social Pooling
- **IndependentMultiVDTrainer**: 基準比較，獨立VD性能

### Q5: 如何可視化結果？
```python
from social_xlstm.visualization.training_visualizer import TrainingVisualizer

visualizer = TrainingVisualizer()
visualizer.plot_basic_training_curves(recorder, "training_curves.png")
```

### Q6: 訓練失敗如何調試？
1. 檢查數據路徑：`blob/dataset/pre-processed/h5/traffic_features.h5`
2. 確認conda環境：`conda activate social_xlstm`
3. 查看日誌：`logs/training.log`
4. 降低學習率：`--learning_rate 0.0001`
5. 檢查梯度範數：使用 `TrainingRecorder` 監控

## 🔗 相關文檔

- [模組功能說明](../implementation/modules.md)
- [訓練腳本使用指南](training_scripts_guide.md)
- [統一訓練系統使用指南](trainer_usage_guide.md)
- [訓練記錄器使用指南](training_recorder_guide.md)
- [Social xLSTM 架構設計](../architecture/social_xlstm_design.md)
- [專案概述](../overview/project_overview.md)

## 📊 實際範例

### 完整訓練流程

```bash
# 1. 確保環境
conda activate social_xlstm

# 2. 準備數據
snakemake --cores 4

# 3. 訓練模型
python scripts/train/without_social_pooling/train_single_vd.py \
  --epochs 100 \
  --batch_size 32 \
  --hidden_size 128 \
  --experiment_name "my_lstm_experiment"

# 4. 查看結果
ls blob/experiments/my_lstm_experiment/
```

### 批量實驗

```bash
# 並行執行所有訓練
snakemake train_single_vd_without_social_pooling \
         train_multi_vd_without_social_pooling \
         train_independent_multi_vd_without_social_pooling \
         --cores 3
```

### 結果分析

```python
# 比較不同訓練方法
from social_xlstm.training.recorder import TrainingRecorder

# 載入記錄
single_vd = TrainingRecorder.load("blob/experiments/single_vd/training_record.json")
multi_vd = TrainingRecorder.load("blob/experiments/multi_vd/training_record.json")
independent = TrainingRecorder.load("blob/experiments/independent_multi_vd/training_record.json")

# 比較結果
comparison = single_vd.compare_with(multi_vd)
print(f"更佳方法: {comparison['better_performer']}")
print(f"性能差異: {comparison['loss_difference']:.4f}")
```

## 📞 支援

如有問題，請參考：
1. 📖 專案文檔 (`docs/` 目錄)
2. 💻 範例代碼 (`scripts/train/without_social_pooling/`)
3. 🐛 GitHub Issues
4. 🔍 快速初始化：`python scripts/utils/claude_init.py --quick`
5. 📊 測試腳本：`python scripts/train/test_training_scripts.py --quick`

### 故障排除

**常見錯誤**：
- `ModuleNotFoundError`: 確保使用 `pip install -e .` 安裝
- `CUDA out of memory`: 降低 batch_size 或啟用混合精度
- `數據文件不存在`: 運行 `snakemake --cores 4` 生成數據
- `conda環境問題`: 確認使用 `conda activate social_xlstm`

**性能調優**：
- 單VD：batch_size=32-64, hidden_size=128
- 多VD：batch_size=16-32, hidden_size=256
- 啟用混合精度：`--mixed_precision`
- 使用GPU：確保CUDA可用