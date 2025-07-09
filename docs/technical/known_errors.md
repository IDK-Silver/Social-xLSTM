# 已知錯誤記錄

## 🚨 多VD訓練潛在錯誤記錄

本文檔記錄了多VD訓練中可能發生的各種錯誤，以及相應的解決方案。

## 1. 維度相關錯誤

### 錯誤A: 輸入維度不匹配
```
ValueError: Multi-VD mode expects 4D input, got 3D
```
**原因**: 
- 訓練器錯誤檢測 `self.model.multi_vd_mode` 而非 `self.model.config.multi_vd_mode`
- 數據加載器返回錯誤的張量形狀

**解決方案**:
```python
# 正確的檢測方式
if not getattr(self.model.config, 'multi_vd_mode', False):
```

### 錯誤B: 輸出維度不匹配
```
RuntimeError: The size of tensor a (3) must match the size of tensor b (10) at non-singleton dimension 2
```
**原因**: 模型輸出維度與目標張量維度不匹配
- 模型輸出: `[batch, 1, 3]` (默認 output_size=3)
- 目標張量: `[batch, 1, 10]` (2VD × 5特徵)

**解決方案**:
```python
# 多VD模型工廠方法需要正確設置輸出維度
config = TrafficLSTMConfig(
    input_size=input_size * num_vds,
    output_size=input_size * num_vds,  # 關鍵修復
    multi_vd_mode=True,
    num_vds=num_vds
)
```

## 2. 記憶體相關錯誤

### 錯誤C: GPU記憶體不足 (OOM)
```
RuntimeError: CUDA out of memory
```
**原因**: 多VD模型參數量大，需要更多GPU記憶體
- 單VD模型: ~226K 參數
- 多VD模型: ~1.4M 參數 (6倍增長)

**解決方案**:
```bash
# 降低批次大小
python scripts/train/train_multi_vd.py --batch_size 2

# 降低模型複雜度
python scripts/train/train_multi_vd.py --hidden_size 128 --num_layers 2

# 啟用混合精度
python scripts/train/train_multi_vd.py --mixed_precision
```

## 3. 數據相關錯誤

### 錯誤D: 時間戳解析錯誤
```
ValueError: time data '' does not match format '%Y-%m-%dT%H:%M:%S'
```
**原因**: 數據文件中存在空的時間戳字段

**解決方案**:
```python
# 使用經過測試的數據文件
--data_path /tmp/tmpdhc_pz_1.h5
```

### 錯誤E: VD數量不匹配
```
IndexError: index 2 is out of bounds for axis 2 with size 2
```
**原因**: 指定的VD數量超過數據文件中實際的VD數量

**解決方案**:
```bash
# 檢查數據文件中實際的VD數量
# 或降低指定的VD數量
python scripts/train/train_multi_vd.py --num_vds 2
```

## 4. 配置相關錯誤

### 錯誤F: 模型配置不一致
```
AttributeError: 'TrafficLSTMConfig' object has no attribute 'multi_vd_mode'
```
**原因**: 配置類缺少必要屬性

**解決方案**:
```python
# 確保配置類包含所有必要屬性
@dataclass
class TrafficLSTMConfig:
    multi_vd_mode: bool = False
    num_vds: Optional[int] = None
```

### 錯誤G: 設備不匹配
```
RuntimeError: Expected all tensors to be on the same device
```
**原因**: 模型和數據在不同設備上

**解決方案**:
```python
# 確保數據和模型都在相同設備
inputs = inputs.to(self.config.device)
targets = targets.to(self.config.device)
```

## 5. 訓練過程錯誤

### 錯誤H: 驗證集為空導致的除零錯誤
```
ZeroDivisionError: division by zero
```
**原因**: 小數據集導致驗證集為空

**解決方案**:
```python
# 在 validate_epoch 中添加檢查
if num_batches == 0:
    return {}
avg_loss = total_loss / num_batches
```

### 錯誤I: 學習率調度器錯誤
```
ValueError: step must be a positive integer
```
**原因**: 調度器配置不當

**解決方案**:
```python
# 檢查並修正調度器配置
if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
    if 'val_loss' in val_metrics:
        self.scheduler.step(val_metrics['val_loss'])
```

## 6. 故障排除檢查清單

### 多VD訓練前檢查:
1. ✅ 確認模型配置 `multi_vd_mode=True`
2. ✅ 確認輸出維度 `output_size = input_size * num_vds`
3. ✅ 確認數據文件包含足夠的VD
4. ✅ 確認GPU記憶體足夠 (建議 ≥8GB)
5. ✅ 確認批次大小合理 (建議 ≤16)

### 發生錯誤時的調試步驟:
1. 啟用調試日誌: `logging.basicConfig(level=logging.DEBUG)`
2. 檢查張量形狀: 觀察 `inputs.shape` 和 `targets.shape`
3. 檢查模型配置: 確認 `model.config.multi_vd_mode`
4. 檢查設備配置: 確認所有張量在同一設備
5. 降低複雜度: 減少批次大小、隱藏層大小或層數

## 7. 常見解決方案速查

```bash
# 基本測試配置 (最不容易出錯)
python scripts/train/train_multi_vd.py \
    --epochs 1 \
    --batch_size 4 \
    --num_vds 2 \
    --hidden_size 128 \
    --num_layers 2

# 生產環境配置
python scripts/train/train_multi_vd.py \
    --epochs 100 \
    --batch_size 8 \
    --num_vds 5 \
    --hidden_size 256 \
    --num_layers 3 \
    --mixed_precision \
    --early_stopping_patience 20
```

## 8. 已修復的錯誤

### 2025-07-08 修復記錄:
1. ✅ **多VD輸入維度問題**: 修正訓練器中的 `multi_vd_mode` 檢測
2. ✅ **輸出維度不匹配**: 修正多VD模型工廠方法的 `output_size` 設置
3. ✅ **數據路徑問題**: 更新訓練腳本使用有效的測試數據文件
4. ✅ **ModelEvaluator方法缺失**: 添加 `evaluate()` 方法

## 9. 更新記錄

- **2025-07-08**: 初始版本 - 多VD訓練除錯過程中發現的錯誤
- **未來更新**: 每次發現新錯誤或修復錯誤後更新此文檔

## 📚 相關文檔

- [設計問題記錄](./design_issues_refactoring.md) - 需要重構的設計問題
- [CLAUDE.md](../../CLAUDE.md) - 專案總覽
- [ADR記錄](../adr/) - 架構決策記錄