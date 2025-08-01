# 如何驗證和調試

本指南整合了分析、驗證和故障排除的完整工作流程，幫助你診斷和解決訓練中的問題。

## 📋 快速導覽

- [診斷工作流程](#診斷工作流程)
- [數據分析工具](#數據分析工具)
- [模型驗證方法](#模型驗證方法)
- [常見問題解決](#常見問題解決)
- [性能分析](#性能分析)

## 🔍 診斷工作流程

### 標準診斷步驟

```bash
# 1. 檢查系統狀態
conda activate social_xlstm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 驗證數據完整性
python scripts/analysis/data_quality_analysis.py

# 3. 快速訓練測試
python scripts/validation/training_validation.py

# 4. 檢查配置檔案
cat cfgs/snakemake/dev.yaml

# 5. 查看錯誤日誌
tail -n 50 logs/training.log
```

### 問題分類架構

```
🔧 配置錯誤 → 檢查參數和路徑
💾 數據錯誤 → 檢查資料格式和內容  
🏗️ 架構錯誤 → 檢查模型和訓練器配置
📈 性能問題 → 檢查指標和收斂性
🖥️ 系統問題 → 檢查環境和資源
```

## 📊 數據分析工具

### 1. 數據品質全面檢查

```bash
# 綜合數據健康檢查
python scripts/analysis/data_quality_analysis.py

# 輸出包含：
# - 時間一致性檢查
# - 數值合理性驗證  
# - 缺失數據分析
# - 異常值檢測
```

### 2. HDF5 數據深度分析

```bash
# 針對 HDF5 格式的詳細分析
python scripts/analysis/h5_data_analysis.py

# 分析內容：
# - 數據結構檢查：(4267, 3, 5) 時間樣本 x VD數量 x 特徵數
# - 每個 VD 的數據品質統計
# - 訓練/驗證集分佈比較
# - 數據洩漏檢測
```

### 3. 時間模式分析  

```bash
# 深度時間分析（當發現時間相關問題時）
python scripts/analysis/temporal_pattern_analysis.py

# 檢測：
# - 數據隨時間的完整性
# - 數值分佈的時間漂移
# - 系統性模式問題
# - 數據生成異常
```

### 4. 快速結構檢查

```bash
# 檢查 HDF5 文件結構
python scripts/utils/h5_structure_inspector.py

# 輸出示例：
# 📁 Root level keys: ['data', 'metadata']
# 📄 Dataset: data/features, shape: (4267, 3, 5), dtype: float64
```

## 🧪 模型驗證方法

### 1. 快速訓練驗證

```bash
# 最小化訓練測試（8-10 epochs）
python scripts/validation/training_validation.py

# 評估標準：
# - 優秀：過擬合比例 < 3
# - 良好：過擬合比例 < 8  
# - 中等：過擬合比例 < 20
# - 差：過擬合比例 > 20
```

### 2. 過擬合修復驗證

```bash
# 綜合過擬合測試
python scripts/validation/overfitting_validation.py

# 比較基準：
# - 原始 LSTM：訓練/驗證比例 113.55
# - 原始 xLSTM：訓練/驗證比例 98.98
# - 目標：比例 < 5（優秀）或 < 10（可接受）
```

### 3. 時間切分驗證

```bash
# 驗證數據切分策略
python scripts/validation/temporal_split_validation.py

# 比較：
# - 隨機切分 vs 時間切分
# - 分佈差異改善度量
# - 數據洩漏檢測
```

### 程式化驗證

```python
from social_xlstm.evaluation.evaluator import ModelEvaluator

# 創建評估器
evaluator = ModelEvaluator()

# 評估模型
metrics = evaluator.evaluate(model, test_data)
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")

# 檢查指標健康度
if metrics['r2'] < 0:
    print("⚠️ 警告：R² < 0，模型效果比隨機猜測還差")
elif metrics['r2'] < 0.5:
    print("⚠️ 注意：R² < 0.5，模型效果不佳")
```

## 🚨 常見問題解決

### 過擬合問題（最常見）

**識別症狀**：
- 訓練 R² = 0.9，驗證 R² = -6
- 訓練 MAE = 0.3，驗證 MAE = 1.9

**解決步驟**：
```python
# 1. 增加正則化
TrainingConfig(
    dropout=0.4,           # 增加 Dropout
    weight_decay=0.01,     # 添加權重衰減
    early_stopping_patience=15
)

# 2. 減少模型複雜度
model = TrafficLSTM.create_single_vd_model(
    hidden_size=64,        # 減少隱藏層
    num_layers=2          # 減少層數
)

# 3. 增加數據
# 使用數據增強或收集更多訓練數據
```

### 數據載入錯誤

**錯誤範例**：`FileNotFoundError: traffic_features.h5`

**解決步驟**：
```bash
# 1. 檢查文件存在
ls -la blob/dataset/pre-processed/h5/

# 2. 重新生成數據
snakemake --configfile cfgs/snakemake/dev.yaml create_h5_file

# 3. 檢查權限
chmod 644 blob/dataset/pre-processed/h5/traffic_features.h5
```

### 記憶體不足錯誤

**錯誤範例**：`CUDA out of memory`

**解決步驟**：
```python
# 1. 減少批次大小
TrainingConfig(batch_size=8)  # 從 32 降到 8

# 2. 啟用混合精度
TrainingConfig(mixed_precision=True)

# 3. 減少序列長度
TrafficDataModule(sequence_length=6)  # 從 12 降到 6

# 4. 檢查記憶體使用
import torch
print(f"GPU 記憶體使用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
```

### 訓練不收斂

**識別症狀**：
- 損失不下降或劇烈震盪
- 梯度範數異常大或小

**解決步驟**：
```python
# 1. 調整學習率
TrainingConfig(learning_rate=0.0001)  # 降低學習率

# 2. 梯度裁剪
TrainingConfig(gradient_clip_value=1.0)

# 3. 批次標準化
# 在模型中添加 BatchNorm 層

# 4. 檢查數據標準化
from social_xlstm.dataset.core import TrafficDataProcessor
processor = TrafficDataProcessor()
processed_data = processor.process(raw_data)
```

### 配置錯誤

**錯誤範例**：路徑不存在、參數不匹配

**解決模式**：
```bash
# 1. 檢查配置文件
vim cfgs/snakemake/dev.yaml

# 2. 驗證路徑計算  
snakemake --dry-run target_rule

# 3. 比較配置差異
diff cfgs/snakemake/dev.yaml cfgs/snakemake/default.yaml

# 4. 測試配置修正
snakemake --configfile cfgs/snakemake/dev.yaml target_rule
```

## 📈 性能分析

### 訓練指標解讀

**理想指標表現**：
```
損失：     ↓ 穩定下降，最終穩定在低值
MAE/RMSE： ↓ 數值合理，訓練/驗證接近  
MAPE：     ↓ 低於 20%（依應用而定）
R²：       ↑ 訓練 > 0.8，驗證 > 0.6，差距 < 0.2
```

**警告信號**：
```
過擬合：   訓練指標優秀，驗證指標很差
欠擬合：   所有指標都不理想
不穩定：   指標劇烈震盪
數據問題： 異常值（如 MAPE = Infinity）
```

### 性能監控腳本

```python
def analyze_training_health(recorder):
    """分析訓練健康狀況"""
    
    # 獲取最新指標
    latest_epoch = recorder.get_best_epoch()
    train_r2 = latest_epoch.train_metrics.get('r2', 0)
    val_r2 = latest_epoch.val_metrics.get('r2', 0)
    
    # 過擬合檢測
    overfitting_ratio = latest_epoch.train_loss / latest_epoch.val_loss
    
    # 健康評估
    if val_r2 < 0:
        return "🚨 嚴重問題：驗證 R² < 0，比隨機猜測還差"
    elif overfitting_ratio > 10:
        return "⚠️ 嚴重過擬合：需要增加正則化"
    elif val_r2 > 0.7 and overfitting_ratio < 3:
        return "✅ 健康：模型表現良好"
    else:
        return "🔧 需要調優：模型效果有改善空間"
```

### 基準比較

```python
# 載入多個實驗進行比較
single_vd = TrainingRecorder.load("experiments/single_vd/training_record.json")
multi_vd = TrainingRecorder.load("experiments/multi_vd/training_record.json")

# 比較結果
comparison = single_vd.compare_with(multi_vd)
print(f"更佳方法: {comparison['better_performer']}")
print(f"性能差異: {comparison['loss_difference']:.4f}")
```

## 🛠️ 調試最佳實踐

### 調試工作流程

1. **問題重現**：確保能穩定重現問題
2. **最小化測試**：使用小數據和少量 epoch
3. **逐步排除**：一次只修改一個變數
4. **記錄發現**：文檔化問題和解決方案

### 日誌管理

```python
import logging

# 設置詳細日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 在關鍵點添加日誌
logger.info(f"模型輸入形狀: {input_tensor.shape}")
logger.info(f"當前學習率: {optimizer.param_groups[0]['lr']}")
```

### 實驗記錄

```bash
# 創建調試實驗目錄
mkdir -p blob/debug/experiment_$(date +%Y%m%d_%H%M%S)

# 保存完整環境資訊
pip freeze > blob/debug/requirements.txt
nvidia-smi > blob/debug/gpu_info.txt
```

## 📞 尋求幫助

### 支援資源

- **錯誤記錄**：`docs/technical/known_errors.md`
- **設計問題**：`docs/technical/design_issues_refactoring.md` 
- **專案狀態**：`docs_rebuild/PROJECT_STATUS.md`
- **技術決策**：`docs_rebuild/explanation/key-decisions.md`

### 報告問題格式

```markdown
## 問題描述
[詳細描述遇到的問題]

## 重現步驟  
1. 執行命令：`python script.py`
2. 使用配置：`dev.yaml`
3. 錯誤出現在：第X步

## 環境資訊
- Python版本：3.11
- PyTorch版本：2.0.1
- CUDA版本：12.4

## 錯誤訊息
[完整的錯誤堆疊追蹤]

## 已嘗試解決方案
[列出已經嘗試的方法]
```

---

這個整合指南提供了完整的驗證和調試工作流程。遇到問題時，請按照診斷流程逐步排查，大多數問題都有標準的解決方案。