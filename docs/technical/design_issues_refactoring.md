# 設計問題與重構計劃

## 📋 概述

本文檔記錄了當前代碼庫中發現的設計問題，以及相應的重構計劃。這些問題是在2025年7月8日多VD訓練除錯過程中發現的。

## 🔴 P0 - 關鍵問題 (立即重寫)

### 問題1：單/多VD處理邏輯混亂
**位置**: `src/social_xlstm/training/trainer.py:254-278`

**問題描述**:
- 在 `train_epoch()`, `validate_epoch()`, `evaluate_test_set()` 中重複相同的模式檢測邏輯
- 運行時形狀轉換應該在模型或數據集層面處理
- 硬編碼假設單步預測 `targets[:, 0:1, :]`

**現有問題代碼**:
```python
# 重複出現3次的邏輯
if not getattr(self.model.config, 'multi_vd_mode', False):
    inputs = inputs[:, :, 0, :]  # 單VD
    targets = targets[:, :, 0, :]
    targets = targets[:, 0:1, :]  # 硬編碼單步預測
else:
    batch_size, seq_len, num_vds, num_features = inputs.shape
    targets = targets.view(batch_size, targets.shape[1], num_vds * num_features)
    targets = targets[:, 0:1, :]  # 硬編碼單步預測
```

**建議重寫**:
```python
# 策略模式
class TrainingStrategy:
    def process_batch(self, batch): pass

class SingleVDStrategy(TrainingStrategy):
    def process_batch(self, batch):
        inputs = batch['input_seq'][:, :, 0, :]
        targets = batch['target_seq'][:, :, 0, :]
        return inputs, targets

class MultiVDStrategy(TrainingStrategy):
    def process_batch(self, batch):
        inputs = batch['input_seq']
        targets = batch['target_seq'].view(batch_size, -1, num_vds * num_features)
        return inputs, targets
```

### 問題2：訓練腳本90%重複代碼
**位置**: `scripts/train/train_single_vd.py` 和 `scripts/train/train_multi_vd.py`

**問題描述**:
- 兩個腳本共464行代碼，只有10-20行不同
- 違反DRY原則，維護困難
- 錯誤處理和日誌信息不一致

**現有問題代碼**:
```python
# train_single_vd.py 和 train_multi_vd.py 幾乎完全相同
# 唯一差異：
# single: model = TrafficLSTM.create_single_vd_model()
# multi:  model = TrafficLSTM.create_multi_vd_model()
```

**建議重寫**:
```python
# 統一腳本：scripts/train/train_lstm.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single_vd', 'multi_vd'], required=True)
    parser.add_argument('--num_vds', type=int, help='Required for multi_vd mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single_vd':
        trainer = create_single_vd_trainer(args)
    elif args.mode == 'multi_vd':
        trainer = create_multi_vd_trainer(args)
    
    trainer.train()
```

## 🟡 P1 - 高優先級問題

### 問題3：模型架構設計缺陷
**位置**: `src/social_xlstm/models/lstm.py:144-157`

**問題描述**:
- `multi_vd_mode` 標誌導致運行時檢查
- 輸入形狀驗證邏輯混亂
- 工廠方法設計不當

**現有問題代碼**:
```python
if self.config.multi_vd_mode:
    if x.dim() != 4:
        raise ValueError(f"Multi-VD mode expects 4D input, got {x.dim()}D")
    # 運行時處理邏輯
```

**建議重寫**:
```python
class SingleVDLSTM(TrafficLSTM):
    def forward(self, x):
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"
        return super().forward(x)

class MultiVDLSTM(TrafficLSTM):
    def __init__(self, config, num_vds):
        config.input_size = config.input_size * num_vds
        config.output_size = config.output_size * num_vds
        super().__init__(config)
        self.num_vds = num_vds
    
    def forward(self, x):
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        return super().forward(x)
```

### 問題4：數據載入器設計問題
**位置**: `src/social_xlstm/dataset/loader.py:255-259`

**問題描述**:
- 驗證/測試集各自擬合縮放器，違反ML基本原則
- 記憶體使用效率低
- 缺乏快取機制

**現有問題代碼**:
```python
if self.config.normalize:
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.fit_transform(val_data)    # 錯誤！應該用train的scaler
    test_data = scaler.fit_transform(test_data)  # 錯誤！應該用train的scaler
```

**建議重寫**:
```python
class DataScaler:
    def __init__(self):
        self._scaler = None
    
    def fit(self, train_data):
        self._scaler = StandardScaler()
        self._scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
    
    def transform(self, data):
        if self._scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self._scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
```

### 問題5：硬編碼值過多
**位置**: 散布在多個文件中

**問題描述**:
- 魔數到處都是
- 配置不靈活
- 參數含義不清

**現有問題代碼**:
```python
# 散布在各處的硬編碼
input_size: int = 3  # 為什麼是3？
train_ratio=0.7, val_ratio=0.15, test_ratio=0.15  # 為什麼這個比例？
batch_size=32  # 為什麼是32？
early_stopping_patience=15  # 為什麼是15？
```

**建議重寫**:
```python
# constants.py
class TrafficConstants:
    # 交通特徵定義
    BASIC_TRAFFIC_FEATURES = 3  # volume, speed, occupancy
    EXTENDED_TRAFFIC_FEATURES = 5  # + density, flow
    
    # 數據分割比例
    DEFAULT_SPLIT_RATIOS = {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }
    
    # 推薦批次大小
    RECOMMENDED_BATCH_SIZES = {
        'single_vd': 32,
        'multi_vd': 16  # 多VD需要更多記憶體
    }
    
    # 訓練參數
    DEFAULT_EARLY_STOPPING_PATIENCE = 15
    DEFAULT_SEQUENCE_LENGTH = 12
    DEFAULT_PREDICTION_LENGTH = 1
```

## 🟡 P2 - 中優先級問題

### 問題6：錯誤處理不一致
**位置**: 多個文件

**問題描述**:
- 異常類型不統一
- 錯誤信息不具體
- 缺乏錯誤恢復機制

**現有問題代碼**:
```python
# 不同地方用不同的異常類型
raise ValueError("num_vds must be specified")
raise Exception("Training failed")
logger.warning("Test evaluation failed")  # 只是警告，不是異常
```

**建議重寫**:
```python
# exceptions.py
class TrafficModelError(Exception):
    """基礎異常類"""
    pass

class InvalidConfigError(TrafficModelError):
    """配置錯誤"""
    pass

class DataLoadingError(TrafficModelError):
    """數據載入錯誤"""
    pass

class ModelTrainingError(TrafficModelError):
    """模型訓練錯誤"""
    pass
```

### 問題7：性能和記憶體問題
**位置**: `src/social_xlstm/evaluation/evaluator.py`

**問題描述**:
- 所有預測結果載入記憶體
- 沒有批次處理
- 記憶體洩漏風險

**現有問題代碼**:
```python
# 一次性載入所有數據到記憶體
all_predictions = []
all_targets = []
for batch in data_loader:
    # 累積所有數據
    all_predictions.append(output.cpu().numpy())
    all_targets.append(target.cpu().numpy())
```

**建議重寫**:
```python
class EfficientModelEvaluator:
    def __init__(self, model, batch_size=1000):
        self.model = model
        self.batch_size = batch_size
    
    def evaluate_in_batches(self, data_loader):
        metrics = []
        for batch in self.batch_iterator(data_loader):
            batch_metrics = self.compute_batch_metrics(batch)
            metrics.append(batch_metrics)
            # 立即釋放記憶體
        return self.aggregate_metrics(metrics)
```

## 🟢 P3 - 低優先級問題

### 問題8：代碼組織和風格
**位置**: 多個文件

**問題描述**:
- 中英文混用
- 函數過長
- 缺乏類型註解

**建議改進**:
- 統一使用英文或中文
- 函數拆分到50行以內
- 添加完整的類型註解

## 📋 重寫執行計劃

### 階段1：P0關鍵問題 (1-2天)
1. 統一訓練腳本
2. 重構 Trainer 類的單/多VD處理

### 階段2：P1高優先級 (3-4天)
3. 分離模型類
4. 修復數據載入器
5. 提取常數配置

### 階段3：P2中優先級 (2-3天)
6. 統一錯誤處理
7. 性能優化

### 階段4：P3低優先級 (1-2天)
8. 代碼風格統一
9. 添加類型註解

## 🔧 重寫原則

1. **向後兼容**：保持現有API不變
2. **增量重寫**：一次只重寫一個模組
3. **測試驅動**：每次重寫都要有對應測試
4. **文檔同步**：更新CLAUDE.md和相關文檔

## 📊 已知錯誤記錄

### 多VD訓練潛在錯誤
參考 `docs/technical/known_errors.md` 中的詳細錯誤記錄。

## 🔄 更新記錄

- **2025-07-08**: 初始版本 - 多VD訓練除錯過程中發現的設計問題
- **未來更新**: 每次重寫完成後更新此文檔

## 📚 相關文檔

- [CLAUDE.md](../../CLAUDE.md) - 專案總覽
- [ADR記錄](../adr/) - 架構決策記錄
- [數學公式](./mathematical_formulation.tex) - 模型數學定義
- [待辦事項](../todo.md) - 當前任務追蹤