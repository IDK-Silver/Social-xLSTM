# 如何訓練模型

本指南整合完整的模型訓練工作流程，從基本 LSTM 到進階 xLSTM，提供實用的操作步驟。

## 📋 快速導覽

- [環境準備](#環境準備)
- [基本 LSTM 訓練](#基本-lstm-訓練)
- [進階 xLSTM 訓練](#進階-xlstm-訓練)
- [訓練腳本使用](#訓練腳本使用)
- [參數調優](#參數調優)
- [故障排除](#故障排除)

## 🚀 環境準備

### 1. 環境檢查
```bash
# 激活 conda 環境（必須）
conda activate social_xlstm

# 驗證安裝
python -c "import social_xlstm; print('✓ Package installed')"

# 檢查 GPU 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 數據準備
```bash
# 完整數據處理流程
snakemake --configfile cfgs/snakemake/dev.yaml --cores 4

# 驗證數據
python -c "import h5py; print('Data shape:', h5py.File('blob/dataset/pre-processed/h5/traffic_features.h5')['data/features'].shape)"
```

## 🧠 基本 LSTM 訓練

### 單 VD 模型（推薦入門）

```python
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.dataset import TrafficDataModule
from social_xlstm.training.trainer import Trainer, TrainingConfig

# 1. 準備數據
data_module = TrafficDataModule(
    data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
    batch_size=32,
    sequence_length=12
)
data_module.setup()

# 2. 創建模型
model = TrafficLSTM.create_single_vd_model(
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# 3. 配置訓練
config = TrainingConfig(
    epochs=50,
    learning_rate=0.001,
    experiment_name="my_first_lstm"
)

# 4. 訓練
trainer = Trainer(
    model=model,
    training_config=config,
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    test_loader=data_module.test_dataloader()
)

history = trainer.train()
```

### 多 VD 模型（空間關係）

```python
# 創建多 VD 模型
model = TrafficLSTM.create_multi_vd_model(
    num_vds=5,
    hidden_size=256,  # 較大的隱藏層
    num_layers=3,
    dropout=0.3
)

# 調整配置
config = TrainingConfig(
    epochs=100,
    batch_size=16,    # 較小批次
    learning_rate=0.0008,
    experiment_name="multi_vd_lstm"
)
```

### 使用腳本訓練

```bash
# 單 VD 訓練
python scripts/train/without_social_pooling/train_single_vd.py \
  --epochs 50 \
  --batch_size 32 \
  --hidden_size 128 \
  --experiment_name "my_experiment"

# 多 VD 訓練
python scripts/train/without_social_pooling/train_multi_vd.py \
  --num_vds 5 \
  --batch_size 16 \
  --hidden_size 256 \
  --mixed_precision

# 使用 Snakemake（推薦）
snakemake --configfile cfgs/snakemake/dev.yaml train_single_vd_without_social_pooling
```

## 🔬 進階 xLSTM 訓練

### 基本 xLSTM 使用

```python
from social_xlstm.models import TrafficXLSTM, TrafficXLSTMConfig

# 1. 創建配置
config = TrafficXLSTMConfig(
    input_size=3,
    embedding_dim=128,
    num_blocks=6,          # xLSTM 區塊數
    slstm_at=[1, 3],      # sLSTM 位置
    dropout=0.1,
    context_length=256
)

# 2. 初始化模型
model = TrafficXLSTM(config)

# 3. 檢查模型資訊
info = model.get_model_info()
print(f"總參數: {info['total_parameters']:,}")
print(f"xLSTM 區塊數: {info['num_blocks']}")
```

### xLSTM 配置選擇

```python
# 小型模型 - 快速實驗
small_config = TrafficXLSTMConfig(
    embedding_dim=64,
    num_blocks=4,
    slstm_at=[1],
    dropout=0.2
)

# 大型模型 - 完整訓練  
large_config = TrafficXLSTMConfig(
    embedding_dim=256,
    num_blocks=8,
    slstm_at=[1, 3, 5],
    dropout=0.1,
    context_length=512
)
```

## ⚙️ 參數調優

### 推薦配置

| 場景 | batch_size | hidden_size | num_layers | learning_rate |
|------|------------|-------------|------------|---------------|
| 單VD快速測試 | 32-64 | 64-128 | 2 | 0.001 |
| 單VD完整訓練 | 32 | 128-256 | 2-3 | 0.001 |
| 多VD訓練 | 16-32 | 256-512 | 2-3 | 0.0008 |
| xLSTM 訓練 | 16-32 | 128-256 | 6-8 blocks | 0.0005 |

### 優化器選擇

```python
# Adam - 預設選擇
TrainingConfig(optimizer_type="adam", learning_rate=0.001)

# AdamW - 大模型推薦
TrainingConfig(optimizer_type="adamw", weight_decay=0.01)

# 學習率調度
TrainingConfig(
    scheduler_type="reduce_on_plateau",
    scheduler_patience=10,
    scheduler_factor=0.5
)
```

### 性能優化

```python
# GPU 加速配置
TrainingConfig(
    device="cuda",
    mixed_precision=True,      # 混合精度
    gradient_clip_value=1.0,   # 梯度裁剪
    num_workers=4              # 數據載入並行
)
```

## 📊 訓練監控

### 使用 TrainingRecorder

```python
from social_xlstm.training.recorder import TrainingRecorder

# 初始化記錄器
recorder = TrainingRecorder(
    experiment_name="my_experiment",
    model_config=model.config.__dict__,
    training_config=config.__dict__
)

# 在訓練循環中記錄
recorder.log_epoch(
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    learning_rate=lr,
    epoch_time=time
)

# 保存記錄
recorder.save("experiments/my_experiment/training_record.json")
```

### 視覺化結果

```python
from social_xlstm.visualization.training_visualizer import TrainingVisualizer

visualizer = TrainingVisualizer()
visualizer.plot_training_dashboard(recorder, "dashboard.png")
```

## 🚨 故障排除

### 常見錯誤與解決

#### 1. 數據文件不存在
```bash
# 錯誤：找不到數據文件
# 解決：重新生成數據
snakemake --configfile cfgs/snakemake/dev.yaml --cores 4
```

#### 2. 記憶體不足
```python
# 解決方案：減少批次大小
TrainingConfig(batch_size=16)  # 或更小

# 啟用混合精度
TrainingConfig(mixed_precision=True)

# 減少模型大小
model = TrafficLSTM.create_single_vd_model(
    hidden_size=64,
    num_layers=2
)
```

#### 3. 模型不收斂
```python
# 解決方案：降低學習率
TrainingConfig(learning_rate=0.0001)

# 啟用梯度裁剪
TrainingConfig(gradient_clip_value=1.0)

# 增加正則化
TrainingConfig(weight_decay=0.01)
```

#### 4. 過擬合問題
```python
# 解決方案：增加 Dropout
model = TrafficLSTM.create_single_vd_model(dropout=0.4)

# 早停機制
TrainingConfig(early_stopping_patience=15)

# 資料增強
TrainingConfig(use_data_augmentation=True)
```

### 調試工作流程

```bash
# 1. 快速測試
python scripts/validation/training_validation.py

# 2. 檢查數據品質
python scripts/analysis/data_quality_analysis.py

# 3. 驗證配置
snakemake --configfile cfgs/snakemake/dev.yaml --dry-run
```

## 💡 最佳實踐

### 實驗管理
- 使用描述性實驗名稱：`single_vd_baseline_v1`
- 保存完整配置和結果
- 記錄實驗目的和發現

### 開發工作流程
1. **小數據測試**：先用少量 epoch 驗證配置
2. **漸進優化**：逐步增加模型複雜度
3. **比較基準**：與之前最佳結果比較
4. **文檔記錄**：記錄重要發現和配置

### 效能考量
- 單 VD：適合快速原型和基準測試
- 多 VD：適合學習空間關係
- xLSTM：適合探索新架構，但計算成本較高

---

此指南整合了完整的模型訓練流程，從基礎到進階，提供實用的操作指導。遇到問題時，請先參考故障排除部分，或查看相關日誌文件。