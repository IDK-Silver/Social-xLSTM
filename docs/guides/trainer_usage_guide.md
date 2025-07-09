# 統一訓練系統使用指南

本指南說明如何使用 Social-xLSTM 專案的統一訓練系統（`src/social_xlstm/training/trainer.py`）。

## 📋 目錄

1. [快速開始](#快速開始)
2. [核心組件](#核心組件)
3. [詳細使用說明](#詳細使用說明)
4. [配置選項](#配置選項)
5. [實際範例](#實際範例)
6. [常見問題](#常見問題)

## 🚀 快速開始

### 最簡單的使用方式

```python
from social_xlstm.training.trainer import Trainer, TrainingConfig
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.dataset import TrafficDataModule

# 1. 準備數據
data_module = TrafficDataModule(
    data_path="path/to/traffic_features.h5",
    batch_size=32,
    sequence_length=12
)
data_module.setup()

# 2. 創建模型
model = TrafficLSTM.create_single_vd_model()

# 3. 配置訓練
config = TrainingConfig(
    epochs=50,
    experiment_name="my_first_experiment"
)

# 4. 開始訓練
trainer = Trainer(
    model=model,
    training_config=config,
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    test_loader=data_module.test_dataloader()
)

# 5. 執行訓練
history = trainer.train()
```

## 🧠 核心組件

### TrainingConfig - 訓練配置類
控制訓練過程的所有參數：

```python
@dataclass
class TrainingConfig:
    # 基礎訓練參數
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # 優化器選擇
    optimizer_type: str = "adam"  # adam, sgd, adamw
    
    # 學習率調度
    scheduler_type: str = "reduce_on_plateau"
    
    # 早停機制
    early_stopping_patience: int = 20
    
    # 實驗管理
    experiment_name: str = "traffic_lstm_experiment"
    save_dir: str = "experiments"
```

### Trainer - 核心訓練類
提供完整的訓練功能：

```python
class Trainer:
    def __init__(self, model, training_config, train_loader, val_loader, test_loader)
    def train()  # 主訓練循環
    def evaluate_test_set()  # 測試集評估
    def save_checkpoint()  # 保存檢查點
    def plot_training_curves()  # 繪製訓練曲線
```

## 📖 詳細使用說明

### 1. 數據準備

```python
from social_xlstm.dataset import TrafficDataModule

# 方式1: 使用 TrafficDataModule
data_module = TrafficDataModule(
    data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
    batch_size=32,
    sequence_length=12,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
data_module.setup()

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

### 2. 模型創建

```python
from social_xlstm.models.lstm import TrafficLSTM

# 方式1: 使用便利方法
model = TrafficLSTM.create_single_vd_model(
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# 方式2: 手動配置
from social_xlstm.models.lstm import TrafficLSTMConfig

config = TrafficLSTMConfig(
    input_size=3,
    hidden_size=64,
    num_layers=3,
    output_size=3,
    dropout=0.3
)
model = TrafficLSTM(config)
```

### 3. 訓練配置

```python
from social_xlstm.training.trainer import TrainingConfig

# 基本配置
config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    experiment_name="traffic_prediction_v1"
)

# 進階配置
advanced_config = TrainingConfig(
    # 訓練參數
    epochs=200,
    batch_size=64,
    learning_rate=0.0005,
    weight_decay=1e-5,
    
    # 優化器配置
    optimizer_type="adamw",
    betas=(0.9, 0.999),
    
    # 學習率調度
    use_scheduler=True,
    scheduler_type="cosine",
    
    # 早停和檢查點
    early_stopping_patience=15,
    save_checkpoints=True,
    checkpoint_interval=5,
    
    # 性能優化
    mixed_precision=True,
    gradient_clip_value=1.0,
    
    # 實驗管理
    experiment_name="advanced_traffic_model",
    save_dir="experiments",
    
    # 視覺化
    plot_training_curves=True,
    plot_predictions=True,
    plot_interval=10
)
```

### 4. 執行訓練

```python
# 創建訓練器
trainer = Trainer(
    model=model,
    training_config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# 開始訓練
print("開始訓練...")
history = trainer.train()

# 查看訓練結果
print("訓練完成！")
print(f"最佳驗證損失: {trainer.best_val_loss:.6f}")
print(f"訓練歷史: {history}")
```

## ⚙️ 配置選項

### 優化器選擇
```python
# Adam 優化器 (預設)
TrainingConfig(optimizer_type="adam", learning_rate=0.001)

# AdamW 優化器 (適合大模型)
TrainingConfig(optimizer_type="adamw", weight_decay=0.01)

# SGD 優化器 (經典選擇)
TrainingConfig(optimizer_type="sgd", momentum=0.9)
```

### 學習率調度
```python
# 平台降低 (預設)
TrainingConfig(
    scheduler_type="reduce_on_plateau",
    scheduler_patience=10,
    scheduler_factor=0.5
)

# 階梯降低
TrainingConfig(
    scheduler_type="step",
    scheduler_step_size=30,
    scheduler_factor=0.1
)

# 餘弦退火
TrainingConfig(scheduler_type="cosine")
```

### 性能優化
```python
# GPU 加速配置
TrainingConfig(
    device="cuda",
    mixed_precision=True,  # 混合精度訓練
    gradient_clip_value=1.0,  # 梯度裁剪
    num_workers=4  # 數據載入並行
)
```

## 💻 實際範例

### 範例1: 基本單VD模型訓練

```python
import torch
from social_xlstm.training.trainer import Trainer, TrainingConfig
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.dataset import TrafficDataModule

def train_single_vd_model():
    """訓練單VD LSTM模型"""
    
    # 1. 數據準備
    print("準備數據...")
    data_module = TrafficDataModule(
        data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
        batch_size=32,
        sequence_length=12
    )
    data_module.setup()
    
    # 2. 模型創建
    print("創建模型...")
    model = TrafficLSTM.create_single_vd_model(
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"模型參數: {model.get_model_info()}")
    
    # 3. 訓練配置
    config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=10,
        experiment_name="single_vd_baseline",
        plot_training_curves=True
    )
    
    # 4. 創建訓練器
    trainer = Trainer(
        model=model,
        training_config=config,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    # 5. 開始訓練
    print("開始訓練...")
    history = trainer.train()
    
    # 6. 結果分析
    print("\n訓練完成！")
    print(f"最終訓練損失: {history['train_loss'][-1]:.6f}")
    print(f"最終驗證損失: {history['val_loss'][-1]:.6f}")
    
    return trainer, history

if __name__ == "__main__":
    trainer, history = train_single_vd_model()
```

### 範例2: 多VD模型訓練

```python
def train_multi_vd_model():
    """訓練多VD LSTM模型"""
    
    # 數據準備 (多VD數據)
    data_module = TrafficDataModule(
        data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
        batch_size=16,  # 較小批次，因為多VD數據較大
        sequence_length=12,
        selected_vd_ids=['VD001', 'VD002', 'VD003', 'VD004', 'VD005']
    )
    data_module.setup()
    
    # 創建多VD模型
    model = TrafficLSTM.create_multi_vd_model(
        num_vds=5,
        hidden_size=256,  # 更大的隱藏層
        num_layers=3,     # 更多層數
        dropout=0.3
    )
    
    # 進階訓練配置
    config = TrainingConfig(
        epochs=100,
        batch_size=16,
        learning_rate=0.0008,
        optimizer_type="adamw",
        weight_decay=0.01,
        
        # 學習率調度
        use_scheduler=True,
        scheduler_type="cosine",
        
        # 早停
        early_stopping_patience=15,
        
        # 性能優化
        mixed_precision=True,
        gradient_clip_value=1.0,
        
        # 實驗管理
        experiment_name="multi_vd_advanced",
        save_dir="experiments",
        
        # 視覺化
        plot_training_curves=True,
        plot_predictions=True,
        plot_interval=10
    )
    
    # 訓練
    trainer = Trainer(model, config, 
                     data_module.train_dataloader(),
                     data_module.val_dataloader(),
                     data_module.test_dataloader())
    
    history = trainer.train()
    
    return trainer, history
```

### 範例3: 從檢查點恢復訓練

```python
def resume_training():
    """從檢查點恢復訓練"""
    
    # 準備數據載入器
    data_module = TrafficDataModule(
        data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
        batch_size=32
    )
    data_module.setup()
    
    # 從檢查點載入
    trainer = Trainer.load_checkpoint(
        checkpoint_path="experiments/my_experiment/best_model.pt",
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader()
    )
    
    print(f"從 epoch {trainer.epoch} 恢復訓練")
    print(f"最佳驗證損失: {trainer.best_val_loss:.6f}")
    
    # 繼續訓練
    history = trainer.train()
    
    return trainer, history
```

### 範例4: 超參數調優

```python
def hyperparameter_tuning():
    """超參數調優範例"""
    
    # 超參數組合
    hyperparams = [
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001},
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.0008},
        {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.4, 'lr': 0.0005},
    ]
    
    best_val_loss = float('inf')
    best_config = None
    results = []
    
    for i, params in enumerate(hyperparams):
        print(f"\n實驗 {i+1}/{len(hyperparams)}: {params}")
        
        # 數據準備
        data_module = TrafficDataModule(
            data_path="blob/dataset/pre-processed/h5/traffic_features.h5",
            batch_size=32
        )
        data_module.setup()
        
        # 創建模型
        model = TrafficLSTM.create_single_vd_model(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
        
        # 配置訓練
        config = TrainingConfig(
            epochs=30,  # 較少epoch用於快速評估
            learning_rate=params['lr'],
            early_stopping_patience=5,
            experiment_name=f"hyperparam_exp_{i+1}",
            plot_training_curves=False  # 關閉視覺化以加快速度
        )
        
        # 訓練
        trainer = Trainer(model, config,
                         data_module.train_dataloader(),
                         data_module.val_dataloader(),
                         data_module.test_dataloader())
        
        history = trainer.train()
        
        # 記錄結果
        final_val_loss = min(history['val_loss'])
        results.append({
            'params': params,
            'val_loss': final_val_loss,
            'experiment': f"hyperparam_exp_{i+1}"
        })
        
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_config = params
        
        print(f"驗證損失: {final_val_loss:.6f}")
    
    # 輸出最佳結果
    print(f"\n最佳配置: {best_config}")
    print(f"最佳驗證損失: {best_val_loss:.6f}")
    
    return results, best_config
```

## ❓ 常見問題

### Q1: 記憶體不足怎麼辦？

**解決方案**:
```python
# 減少批次大小
TrainingConfig(batch_size=16)  # 或更小

# 啟用混合精度
TrainingConfig(mixed_precision=True)

# 減少模型大小
model = TrafficLSTM.create_single_vd_model(
    hidden_size=64,  # 減少隱藏層大小
    num_layers=2     # 減少層數
)
```

### Q2: 訓練速度太慢怎麼辦？

**解決方案**:
```python
# 增加批次大小
TrainingConfig(batch_size=64)

# 啟用混合精度
TrainingConfig(mixed_precision=True)

# 增加數據載入工作進程
TrainingConfig(num_workers=8)

# 使用更快的優化器
TrainingConfig(optimizer_type="adamw")
```

### Q3: 模型不收斂怎麼辦？

**解決方案**:
```python
# 降低學習率
TrainingConfig(learning_rate=0.0001)

# 啟用梯度裁剪
TrainingConfig(gradient_clip_value=1.0)

# 使用學習率調度
TrainingConfig(
    use_scheduler=True,
    scheduler_type="reduce_on_plateau"
)

# 增加正則化
TrainingConfig(weight_decay=0.01)
```

### Q4: 如何監控訓練過程？

**解決方案**:
```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.INFO)

# 啟用視覺化
TrainingConfig(
    plot_training_curves=True,
    plot_interval=10,
    log_interval=5
)

# 查看實驗目錄
# experiments/your_experiment_name/
# ├── config.json          # 配置文件
# ├── training_curves.png  # 訓練曲線
# ├── predictions.png      # 預測結果
# ├── test_evaluation.json # 測試評估
# └── best_model.pt        # 最佳模型
```

### Q5: 如何比較不同模型？

**解決方案**:
```python
# 使用不同的實驗名稱
configs = [
    TrainingConfig(experiment_name="lstm_small", hidden_size=64),
    TrainingConfig(experiment_name="lstm_large", hidden_size=256),
    TrainingConfig(experiment_name="lstm_deep", num_layers=4)
]

# 訓練多個模型並比較結果
results = []
for config in configs:
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    history = trainer.train()
    results.append({
        'name': config.experiment_name,
        'best_val_loss': min(history['val_loss'])
    })

# 比較結果
for result in sorted(results, key=lambda x: x['best_val_loss']):
    print(f"{result['name']}: {result['best_val_loss']:.6f}")
```

## 🔗 相關文檔

- [LSTM 使用指南](lstm_usage_guide.md)
- [模組功能說明](../implementation/modules.md)
- [Social xLSTM 架構設計](../architecture/social_xlstm_design.md)
- [專案概述](../overview/project_overview.md)

## 📞 支援

如有問題，請參考：
1. 📖 這份使用指南
2. 💻 範例代碼 (`examples/`)
3. 🐛 GitHub Issues
4. 📧 專案團隊聯繫方式

---

**提醒**: 訓練系統會自動創建實驗目錄並保存所有結果，請確保有足夠的磁碟空間。