# Training Recorder 使用指南

## 📋 概述

`TrainingRecorder` 是 Social-xLSTM 專案的核心訓練記錄系統，專注於完整記錄訓練過程中的所有資訊。它能夠將訓練資料保存為多種格式，支援事後分析和新指標計算。視覺化功能由獨立的 `TrainingVisualizer` 類別提供。

## 🎯 主要功能

### 1. 完整的訓練記錄
- **每個 epoch 的詳細資訊**：損失、指標、學習率、時間、記憶體使用等
- **系統元數據**：Git commit、Python 版本、PyTorch 版本、CUDA 資訊
- **實驗配置**：模型配置、訓練配置的完整記錄

### 2. 智能分析
- **最佳 epoch 自動追蹤**
- **訓練穩定性分析**
- **過擬合檢測**
- **收斂狀態評估**

### 3. 多格式輸出支援
- **JSON 格式**：完整的結構化資料
- **CSV 格式**：方便用 Excel 或其他工具分析
- **TensorBoard**：支援即時監控和視覺化
- **PKL 格式**：原始 Python 物件，方便程式處理

### 4. 完整的資料保存
- **每個 epoch 的完整記錄**：損失、指標、系統狀態
- **原始資料保存**：支援事後計算新指標
- **系統元數據**：確保實驗可重現
- **靈活的載入機制**：支援續訓練或分析

## 🚀 快速開始

### 基本使用

```python
from social_xlstm.training.recorder import TrainingRecorder

# 1. 初始化記錄器
recorder = TrainingRecorder(
    experiment_name="baseline_lstm_v1",
    model_config=model.config.__dict__,
    training_config=training_config.__dict__
)

# 2. 在訓練循環中記錄
for epoch in range(epochs):
    # 訓練邏輯...
    start_time = time.time()
    
    train_metrics = train_one_epoch()
    val_metrics = validate_one_epoch()
    
    epoch_time = time.time() - start_time
    
    # 記錄這個 epoch
    recorder.log_epoch(
        epoch=epoch,
        train_loss=train_metrics['loss'],
        val_loss=val_metrics['loss'],
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        learning_rate=optimizer.param_groups[0]['lr'],
        epoch_time=epoch_time
    )

# 3. 保存記錄（多種格式）
recorder.save("experiments/baseline_lstm_v1/training_record.json")  # JSON格式
recorder.export_to_csv("experiments/baseline_lstm_v1/training_history.csv")  # CSV格式
recorder.export_to_tensorboard("experiments/baseline_lstm_v1/tensorboard")  # TensorBoard

# 4. 使用獨立的視覺化工具
from social_xlstm.visualization.training_visualizer import TrainingVisualizer

visualizer = TrainingVisualizer()
visualizer.plot_training_dashboard(recorder, "experiments/baseline_lstm_v1/dashboard.png")
```

### 與現有 Trainer 整合

```python
# 修改 trainer.py 中的 __init__ 方法
class Trainer:
    def __init__(self, model, training_config, ...):
        # 原有初始化...
        
        # 替換原有的 training_history
        self.recorder = TrainingRecorder(
            experiment_name=training_config.experiment_name,
            model_config=model.config.__dict__,
            training_config=training_config.__dict__
        )
    
    def train(self):
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # 訓練和驗證
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            epoch_time = time.time() - start_time
            
            # 記錄到 recorder
            self.recorder.log_epoch(
                epoch=epoch,
                train_loss=train_metrics['train_loss'],
                val_loss=val_metrics.get('val_loss'),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time,
                gradient_norm=self._calculate_gradient_norm()
            )
            
            # 其他訓練邏輯...
        
        # 保存完整記錄
        self.recorder.save(self.experiment_dir / "training_record.json")
        
        return self.recorder
```

## 📊 詳細功能說明

### 1. EpochRecord 資料結構

每個 epoch 的記錄包含：

```python
@dataclass
class EpochRecord:
    epoch: int                              # Epoch 編號
    timestamp: datetime                     # 時間戳
    train_loss: float                       # 訓練損失
    val_loss: Optional[float]               # 驗證損失
    train_metrics: Dict[str, float]         # 訓練指標 (MAE, MSE, etc.)
    val_metrics: Dict[str, float]           # 驗證指標
    learning_rate: float                    # 學習率
    epoch_time: float                       # 訓練時間
    memory_usage: Optional[float]           # 記憶體使用
    gradient_norm: Optional[float]          # 梯度範數
    is_best: bool                          # 是否最佳 epoch
    sample_predictions: Optional[Dict]      # 樣本預測（可選）
```

### 2. 記錄方法

#### log_epoch() - 記錄單個 epoch

```python
recorder.log_epoch(
    epoch=0,
    train_loss=0.1234,
    val_loss=0.1456,
    train_metrics={
        'mae': 0.0876,
        'mse': 0.0123,
        'rmse': 0.1110,
        'r2': 0.8765
    },
    val_metrics={
        'mae': 0.0923,
        'mse': 0.0134,
        'rmse': 0.1158,
        'r2': 0.8654
    },
    learning_rate=0.001,
    epoch_time=45.6,
    gradient_norm=0.234
)
```

### 3. 查詢方法

#### get_metric_history() - 獲取指標歷史

```python
# 獲取訓練 MAE 歷史
train_mae_history = recorder.get_metric_history('mae', 'train')

# 獲取驗證 MSE 歷史
val_mse_history = recorder.get_metric_history('mse', 'val')
```

#### get_loss_history() - 獲取損失歷史

```python
train_losses, val_losses = recorder.get_loss_history()
```

#### get_best_epoch() - 獲取最佳 epoch

```python
best_epoch = recorder.get_best_epoch()
print(f"最佳 epoch: {best_epoch.epoch}")
print(f"最佳驗證損失: {best_epoch.val_loss}")
```

#### get_training_summary() - 獲取訓練總結

```python
summary = recorder.get_training_summary()
print(f"總 epochs: {summary['total_epochs']}")
print(f"總時間: {summary['total_time']:.2f} 秒")
print(f"平均 epoch 時間: {summary['avg_epoch_time']:.2f} 秒")
print(f"最佳 epoch: {summary['best_epoch']}")
print(f"最佳驗證損失: {summary['best_val_loss']:.4f}")
```

### 4. 分析方法

#### analyze_training_stability() - 訓練穩定性分析

```python
stability = recorder.analyze_training_stability()
print(f"訓練損失趨勢: {stability['train_trend']:.6f}")
print(f"驗證損失趨勢: {stability['val_trend']:.6f}")
print(f"過擬合分數: {stability['overfitting_score']:.4f}")
print(f"收斂狀態: {stability['convergence_status']}")
```

### 5. 資料輸出方法

#### save() - 保存為 JSON 格式

```python
# 保存完整記錄
recorder.save("training_record.json")
```

JSON 檔案包含：
- 實驗配置（模型、訓練參數）
- 每個 epoch 的詳細記錄
- 系統元數據（Git commit、版本資訊）
- 訓練總結和分析結果

#### export_to_csv() - 輸出為 CSV 格式

```python
# 匯出為 CSV，方便用 Excel 分析
recorder.export_to_csv("training_history.csv")
```

CSV 檔案包含：
- 每個 epoch 一行
- 所有損失、指標、系統狀態
- 方便匯入其他分析工具

#### export_to_tensorboard() - TensorBoard 格式

```python
# 匯出 TensorBoard 日誌
recorder.export_to_tensorboard("runs/experiment_1")

# 啟動 TensorBoard 查看
# tensorboard --logdir=runs
```

### 6. 持久化方法

#### save() - 保存記錄

```python
# 保存為 JSON 文件
recorder.save("experiments/my_experiment/training_record.json")
```

保存的 JSON 包含：
- 實驗配置
- 所有 epoch 記錄
- 系統元數據
- 訓練總結
- 分析結果

#### load() - 載入記錄

```python
# 載入之前保存的記錄
recorder = TrainingRecorder.load("experiments/my_experiment/training_record.json")

# 繼續分析
summary = recorder.get_training_summary()
recorder.plot_training_curves()
```

### 7. 比較方法

#### compare_with() - 實驗比較

```python
# 載入兩個實驗記錄
recorder1 = TrainingRecorder.load("experiments/baseline/training_record.json")
recorder2 = TrainingRecorder.load("experiments/improved/training_record.json")

# 比較實驗
comparison = recorder1.compare_with(recorder2)
print(f"較佳模型: {comparison['better_performer']}")
print(f"損失差異: {comparison['loss_difference']:.4f}")
print(f"收斂 epoch 差異: {comparison['epoch_difference']}")
```

## 📊 視覺化功能（使用 TrainingVisualizer）

`TrainingVisualizer` 是獨立的視覺化類別，提供豐富的圖表功能：

### 基本使用

```python
from social_xlstm.visualization.training_visualizer import TrainingVisualizer

# 載入記錄
recorder = TrainingRecorder.load("training_record.json")

# 創建視覺化器
visualizer = TrainingVisualizer()

# 生成各種圖表
visualizer.plot_basic_training_curves(recorder, "basic_curves.png")
visualizer.plot_training_dashboard(recorder, "dashboard.png")
visualizer.plot_advanced_metrics(recorder, "advanced_metrics.png")
```

### 可用的視覺化功能

1. **基本訓練曲線** (`plot_basic_training_curves`)
   - 損失曲線、主要指標、學習率、訓練時間

2. **訓練儀表板** (`plot_training_dashboard`)
   - 綜合視圖，包含多個關鍵圖表和摘要

3. **進階指標** (`plot_advanced_metrics`)
   - 梯度範數、記憶體使用、收斂分析、過擬合檢測

4. **實驗比較** (`plot_experiment_comparison`)
   - 多個實驗的並排比較

5. **指標演化** (`plot_metric_evolution`)
   - 特定指標的詳細變化過程

6. **完整報告** (`create_training_report`)
   - 自動生成包含所有圖表的完整報告

## 🔧 實際使用場景

### 場景 1: Baseline 實驗記錄

```python
# 單 VD baseline 實驗
recorder = TrainingRecorder(
    experiment_name="single_vd_baseline",
    model_config=model.config.__dict__,
    training_config=training_config.__dict__
)

# 訓練並記錄
for epoch in range(100):
    # 訓練邏輯...
    recorder.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, lr, time)

# 保存多種格式
recorder.save("experiments/single_vd_baseline/training_record.json")
recorder.export_to_csv("experiments/single_vd_baseline/history.csv")

# 視覺化分析
visualizer = TrainingVisualizer()
visualizer.create_training_report(recorder, "experiments/single_vd_baseline/report")

# 獲取 baseline 結果
baseline_summary = recorder.get_training_summary()
```

### 場景 2: 實驗比較和分析

```python
# 載入多個實驗記錄
single_vd = TrainingRecorder.load("experiments/single_vd_baseline/training_record.json")
multi_vd = TrainingRecorder.load("experiments/multi_vd_baseline/training_record.json")

# 比較結果
comparison = single_vd.compare_with(multi_vd)
print(f"單 VD 最佳損失: {single_vd.get_best_epoch().val_loss:.4f}")
print(f"多 VD 最佳損失: {multi_vd.get_best_epoch().val_loss:.4f}")
print(f"更佳方法: {comparison['better_performer']}")

# 分析穩定性
single_stability = single_vd.analyze_training_stability()
multi_stability = multi_vd.analyze_training_stability()
```

### 場景 3: 離線分析和新指標計算

```python
# 載入已保存的記錄
recorder = TrainingRecorder.load("experiments/baseline/training_record.json")

# 計算自定義指標
def custom_metric(predictions, targets):
    # 自定義指標計算邏輯
    return np.mean(np.abs(predictions - targets) / targets)

# 獲取歷史數據進行分析
train_mae_history = recorder.get_metric_history('mae', 'train')
val_mae_history = recorder.get_metric_history('mae', 'val')

# 分析收斂行為
convergence_analysis = recorder._analyze_convergence()
```

## 📈 進階功能

### 1. 自定義指標記錄

```python
# 在訓練循環中記錄自定義指標
custom_metrics = {
    'mae': calculate_mae(predictions, targets),
    'custom_score': calculate_custom_score(predictions, targets),
    'feature_importance': calculate_feature_importance(model)
}

recorder.log_epoch(
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    train_metrics=custom_metrics,
    val_metrics=custom_metrics,
    # 其他參數...
)
```

### 2. 梯度監控

```python
# 計算梯度範數
def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

# 記錄梯度資訊
recorder.log_epoch(
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    gradient_norm=calculate_gradient_norm(model),
    # 其他參數...
)
```

### 3. 樣本預測追蹤

```python
# 記錄代表性樣本的預測
sample_predictions = {
    'sample_inputs': sample_inputs.cpu().numpy(),
    'sample_targets': sample_targets.cpu().numpy(),
    'sample_predictions': sample_outputs.cpu().numpy()
}

recorder.log_epoch(
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    sample_predictions=sample_predictions,
    # 其他參數...
)
```

## 🛠️ 整合到現有系統

### 1. 修改 Trainer 類

```python
# 在 src/social_xlstm/training/trainer.py 中：

from .recorder import TrainingRecorder

class Trainer:
    def __init__(self, model, training_config, ...):
        # 現有初始化...
        
        # 替換 training_history
        self.recorder = TrainingRecorder(
            experiment_name=training_config.experiment_name,
            model_config=model.config.__dict__,
            training_config=training_config.__dict__
        )
    
    def train(self):
        for epoch in range(self.config.epochs):
            # 訓練邏輯...
            
            # 使用 recorder 記錄
            self.recorder.log_epoch(
                epoch=epoch,
                train_loss=train_metrics['train_loss'],
                val_loss=val_metrics.get('val_loss'),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time
            )
        
        # 保存記錄
        self.recorder.save(self.experiment_dir / "training_record.json")
        
        return self.recorder
```

### 2. 訓練腳本更新

```python
# 在 scripts/train/train_single_vd.py 中：

def main():
    # 現有邏輯...
    
    # 訓練
    trainer = Trainer(model, training_config, ...)
    recorder = trainer.train()
    
    # 生成分析報告
    recorder.plot_training_curves(trainer.experiment_dir / "training_curves.png")
    
    # 獲取訓練總結
    summary = recorder.get_training_summary()
    logger.info(f"訓練完成！最佳驗證損失: {summary['best_val_loss']:.4f}")
```

## 📋 最佳實踐

### 1. 命名規範
- 使用描述性的實驗名稱：`single_vd_baseline_v1`
- 包含關鍵參數：`multi_vd_h256_l3_lr0001`

### 2. 記錄內容
- 始終記錄訓練和驗證損失
- 包含關鍵指標（MAE, MSE, RMSE）
- 記錄學習率和訓練時間
- 適當記錄梯度範數

### 3. 文件管理
- 每個實驗獨立目錄
- 統一的文件命名規範
- 定期清理舊實驗記錄

### 4. 分析流程
- 訓練後立即生成視覺化
- 比較關鍵實驗的結果
- 記錄實驗發現和改進方向

## 🚨 注意事項

1. **記憶體使用**：大量 epoch 記錄可能佔用較多記憶體
2. **文件大小**：詳細記錄會產生較大的 JSON 文件
3. **版本相容性**：確保載入記錄時的 Python 版本相容
4. **Git 依賴**：Git commit 記錄需要在 Git 倉庫中執行

## 🔧 故障排除

### 常見問題

1. **無法載入記錄**
   ```python
   # 檢查文件是否存在
   if not Path("training_record.json").exists():
       print("記錄文件不存在")
   ```

2. **繪圖失敗**
   ```python
   # 確保有 matplotlib 後端
   import matplotlib
   matplotlib.use('Agg')  # 無顯示器環境
   ```

3. **記憶體不足**
   ```python
   # 減少詳細記錄
   recorder.log_epoch(
       epoch=epoch,
       train_loss=train_loss,
       val_loss=val_loss,
       # 不記錄 sample_predictions
   )
   ```

這個 Training Recorder 系統為你的實驗提供了完整的記錄和分析能力，確保每個實驗都能得到充分的追蹤和理解。