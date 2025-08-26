# PEMS-BAY 驗證方法分析：論文間的計算細節差異

> 分析三篇關鍵論文在 PEMS-BAY 數據集上的性能驗證方法，揭示計算細節的不足與實現差異

---

## 📚 背景概述

PEMS-BAY 是交通預測領域的標準基準數據集，但不同論文在使用該數據集進行性能驗證時，計算細節往往不夠明確，導致結果難以直接比較。本文分析三篇代表性論文的驗證方法，識別關鍵問題並提出解決方案。

## 🔍 三篇論文的驗證方法對比

### 1. LVSTformer (2024) - Multi-level Multi-view Framework

#### 實驗設置
- **數據分割**: 7:1:2 (訓練:驗證:測試) 
- **序列長度**: 輸入12步 → 預測12步 (1小時到1小時)
- **批次大小**: 16
- **優化器**: Adam (lr=0.001)
- **訓練輪數**: 200 epochs
- **硬體**: NVIDIA RTX 3090

#### 評估指標公式
```
MAE = (1/Ω) × Σ|yi - xi|
RMSE = √[(1/Ω) × Σ|yi - xi|²]
MAPE = (1/Ω) × Σ|yi - xi|/xi × 100%
```
其中 `Ω` 為樣本總數，`yi` 為預測值，`xi` 為真實值。

#### 性能結果 (PEMS-BAY)
| 模型 | MAE | RMSE | MAPE |
|------|-----|------|------|
| DCRNN | 1.75 | 3.92 | 3.93% |
| STGCN | 1.87 | 4.58 | 4.30% |
| GMAN | 1.63 | 3.77 | 3.69% |
| PDFormer | 1.62 | 3.52 | 3.67% |
| **LVSTformer** | **1.55** | **3.46** | **3.56%** |

### 2. SUSTeR (2023) - Sparse Unstructured Reconstruction

#### 實驗設置（稀疏性研究）
- **數據分割**: 70% 訓練 / 10% 驗證 / 20% 測試
- **序列長度**: 輸入12步 → 預測1步 (單步預測)
- **批次大小**: 32
- **優化器**: Adam (lr=5e-4, L2=1e-5)
- **訓練輪數**: 50 epochs
- **硬體**: Tesla V100

#### 稀疏性實驗方法
```python
# 人工引入缺失值
dropout_rates = [10%, 80%, 90%, 99%, 99.9%]

# 隨機採樣策略
for each_timestep:
    for each_sensor:
        keep_prob = uniform_random(0, 1)
        if keep_prob > dropout_rate:
            preserve_sensor_value()
        else:
            set_to_zero()
```

#### 性能結果 (PEMS-BAY, 不同缺失率)
| 缺失率 | STGCN | D2STGNN | SUSTeR |
|--------|--------|---------|--------|
| 10% | 2.293 | 2.359 | 4.711 |
| 80% | 2.340 | 2.353 | 3.183 |
| 90% | 2.356 | 2.348 | **2.369** |
| 99% | 3.377 | 2.491 | **2.457** |
| 99.9% | 4.325 | 3.814 | **2.572** |

### 3. Social LSTM (2016) - 原創架構論文

#### 實驗特點
- **主要用於**: 行人軌跡預測（ETH/UCY數據集）
- **不直接使用**: PEMS-BAY 交通數據
- **評估指標**: ADE (Average Displacement Error), FDE (Final Displacement Error)

```
ADE = (1/Tpred) × Σ||pos_pred - pos_true||₂
FDE = ||pos_final_pred - pos_final_true||₂
```

---

## ❓ 計算細節的關鍵缺失

### 1. 聚合維度不明確

#### 問題描述
PEMS-BAY 數據形狀為 `[Batch, Time, Sensors, Features]`，但論文未說明如何聚合多個維度：

```python
# 方法A: 全局扁平化
predictions.shape = [B, T, N, F]  # [批次, 時間, 感測器, 特徵]
mae_global = mean(abs(pred.flatten() - true.flatten()))

# 方法B: 先按感測器聚合
mae_per_sensor = []
for sensor in range(N):
    mae_s = mean(abs(pred[:,:,sensor,:] - true[:,:,sensor,:]))
    mae_per_sensor.append(mae_s)
mae_sensor_avg = mean(mae_per_sensor)

# 方法C: 先按時間步聚合
mae_per_timestep = []
for t in range(T):
    mae_t = mean(abs(pred[:,t,:,:] - true[:,t,:,:]))
    mae_per_timestep.append(mae_t)
mae_temporal_avg = mean(mae_per_timestep)
```

#### 實際影響
不同聚合方式可能導致 **20-30%** 的指標差異。

### 2. 缺失值處理策略未知

#### 問題描述
雖然 PEMS-BAY 聲稱無缺失值，但實際存在：
- **零速度值**: 521個（可能為交通堵塞）
- **異常高速值**: >100 mph（可能為感測器故障）
- **邊界情況**: 預測時可能產生負值

#### 可能處理方式
```python
def calculate_mae_robust(pred, true):
    # 策略1: 忽略異常值
    valid_mask = (true >= 0) & (true <= 100) & (~torch.isnan(true))
    
    # 策略2: 包含所有非NaN值
    valid_mask = ~torch.isnan(true)
    
    # 策略3: 原始數據不處理
    return torch.mean(torch.abs(pred - true))
```

### 3. MAPE 分母為零問題

#### 問題描述
當真實速度為 0 km/h 時，MAPE 計算會發散：

```python
# 原始公式會出錯
mape = mean(abs(pred - true) / true)  # 當 true=0 時除以零

# 可能的解決方案：
def calculate_mape_safe(pred, true, epsilon=1e-8):
    # 方案1: 添加小常數
    return mean(abs(pred - true) / (true + epsilon))
    
    # 方案2: 跳過零值
    nonzero_mask = abs(true) > epsilon
    return mean(abs(pred[nonzero_mask] - true[nonzero_mask]) / true[nonzero_mask])
    
    # 方案3: 使用對稱MAPE
    return mean(abs(pred - true) / (abs(pred) + abs(true) + epsilon))
```

### 4. 數據預處理不一致

#### 標準化影響
```python
# LVSTformer: 可能使用原始速度值
speed_raw = data  # km/h

# 其他模型: 可能使用標準化
speed_normalized = (data - mean(data)) / std(data)

# 標準化會影響MAE絕對值，但不影響相對排序
mae_raw = 1.55      # 原始單位 km/h
mae_norm = 0.23     # 標準化後的無單位值
```

#### 單位轉換
```python
# 原始PEMS-BAY: mph
speed_mph = raw_data

# 轉換為km/h (標準做法)
speed_kmh = raw_data * 1.609344

# 不同單位導致指標值完全不同
mae_mph = 0.96      # LVSTformer結果可能是mph單位？
mae_kmh = 1.55      # 還是已經轉換為km/h？
```

---

## 📊 結果差異分析

### 異常的性能差異

#### 同數據集，不同結果
- **LVSTformer**: MAE = 1.55 (完整數據，多步預測)
- **SUSTeR**: MAE = 4.71 (10%缺失) vs 2.57 (99.9%缺失)

#### 可能原因
1. **預測任務不同**: 多步 vs 單步
2. **數據預處理**: 標準化 vs 原始值  
3. **計算方式**: 不同聚合策略
4. **模型能力**: 架構本身的差異

### 不合理的趨勢
SUSTeR 論文中，99.9% 缺失率比 10% 缺失率性能更好，這在直覺上不合理，可能說明：
- **基準模型實現有問題**
- **評估方式不一致**
- **數據處理存在bug**

---

## 🎯 對我們專案的啟示

### 1. 建立標準化評估框架

#### 核心要求
```python
class PEMSBAYEvaluator:
    def __init__(self, 
                 aggregation_mode="global",      # global/sensor/temporal
                 handle_zeros="preserve",        # preserve/skip/epsilon
                 mape_epsilon=1e-8,
                 normalize_input=False):
        pass
    
    def calculate_metrics(self, predictions, targets):
        # 確保一致的計算方式
        pass
```

### 2. 可重現的實驗設置

#### 配置標準
```yaml
dataset:
  name: pems_bay
  split_ratio: [0.7, 0.1, 0.2]
  normalize: false              # 使用原始km/h單位
  
model:
  sequence_length: 12           # 1小時輸入
  prediction_length: 12         # 1小時預測
  
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 200
  
evaluation:
  metrics: [mae, rmse, mape]
  aggregation: "global"         # 明確指定聚合方式
  handle_missing: "preserve"    # 明確處理策略
```

### 3. 多重驗證策略

#### 對標主流論文
1. **LVSTformer設置**: 多步預測，完整數據
2. **SUSTeR設置**: 單步預測，稀疏測試  
3. **自定義設置**: 針對Social-xLSTM優化

#### 結果報告
```
# 標準格式報告
PEMS-BAY Results (LVSTformer Setup):
- Data Split: 7:1:2, Input: 12 steps, Output: 12 steps
- Aggregation: Global, Units: km/h, Missing: N/A
- MAE: X.XX ± Y.YY
- RMSE: X.XX ± Y.YY  
- MAPE: X.XX% ± Y.YY%
```

---

## 📋 實施建議

### 階段 1: 建立評估基礎設施 (優先級最高)
1. **統一的指標計算類**
   - 支持多種聚合方式
   - 一致的缺失值處理
   - 可配置的MAPE計算

2. **標準化實驗配置**
   - YAML驅動的設置
   - 可重現的隨機種子
   - 完整的超參數記錄

### 階段 2: 基準實驗重現 
1. **實現經典基準模型**
   - DCRNN, STGCN (用於對標)
   - 確保與論文結果一致

2. **多重驗證測試**
   - 不同計算方式的影響分析
   - 數據預處理的敏感性測試

### 階段 3: Social-xLSTM 專用驗證
1. **全面性能評估**
   - 標準PEMS-BAY基準
   - 稀疏性魯棒性測試
   - 多時間步驟預測

2. **可發表級別的結果**
   - 完整的實驗設置說明
   - 統計顯著性測試
   - 與SOTA方法的公平比較

---

## 🔚 總結

當前交通預測論文在 PEMS-BAY 驗證方面存在顯著的**計算細節不透明**問題，導致結果難以直接比較。我們的專案需要：

1. **建立明確的評估標準** - 解決聚合、缺失值、單位等問題
2. **確保結果可重現** - 完整記錄所有實驗細節  
3. **提供公平的比較** - 與主流方法使用相同的評估條件

只有這樣，Social-xLSTM 的性能評估才能具有學術可信度和實際應用價值。

---

**文件建立時間**: 2025-01-27  
**最後更新**: 2025-01-27  
**相關檔案**: `scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py`