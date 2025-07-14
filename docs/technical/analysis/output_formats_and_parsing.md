# 模型輸出格式與解析技術文檔

## 概述

本文檔說明 Social-xLSTM 專案中各種模型的輸出格式，以及如何解析這些輸出以獲得有意義的預測結果。

## TrafficLSTM 輸出格式

### Single VD Mode (單VD模式)

#### 輸入輸出格式
```python
# 輸入格式
input_shape = [batch_size, sequence_length, num_features]
# 例子: [4, 12, 5] - 4個樣本，12個時間步，5個特徵

# 輸出格式
output_shape = [batch_size, prediction_length, num_features]
# 例子: [4, 1, 5] - 4個樣本，1個預測步，5個特徵
```

#### 特徵含義
```python
# 預設5個交通特徵 (按順序)
features = [
    'volume',      # 交通流量
    'speed',       # 平均速度
    'occupancy',   # 佔有率
    'density',     # 交通密度
    'flow'         # 交通流
]
```

#### 解析方法
Single VD 模式的輸出不需要特殊解析，可以直接使用：
```python
# 直接訪問預測結果
predictions = model(inputs)  # [4, 1, 5]

# 提取特定特徵 (例如：速度預測)
speed_predictions = predictions[:, :, 1]  # [4, 1] - 索引1對應速度
```

### Multi-VD Mode (多VD模式)

#### 輸入輸出格式
```python
# 輸入格式 (扁平化)
input_shape = [batch_size, sequence_length, num_vds * num_features]
# 例子: [4, 12, 15] - 4個樣本，12個時間步，15個扁平化特徵 (3VD × 5特徵)

# 輸出格式 (扁平化)
output_shape = [batch_size, prediction_length, num_vds * num_features]
# 例子: [4, 1, 15] - 4個樣本，1個預測步，15個扁平化特徵
```

#### 扁平化順序
Multi-VD 輸出按以下順序組織：
```python
# 假設有3個VD，每個VD有5個特徵
# 輸出索引對應關係：
[
    # VD_000 的特徵
    VD_000_volume,     # 索引 0
    VD_000_speed,      # 索引 1
    VD_000_occupancy,  # 索引 2
    VD_000_density,    # 索引 3
    VD_000_flow,       # 索引 4
    
    # VD_001 的特徵
    VD_001_volume,     # 索引 5
    VD_001_speed,      # 索引 6
    VD_001_occupancy,  # 索引 7
    VD_001_density,    # 索引 8
    VD_001_flow,       # 索引 9
    
    # VD_002 的特徵
    VD_002_volume,     # 索引 10
    VD_002_speed,      # 索引 11
    VD_002_occupancy,  # 索引 12
    VD_002_density,    # 索引 13
    VD_002_flow,       # 索引 14
]
```

#### 解析方法
Multi-VD 模式需要將扁平化輸出重構為結構化格式：

##### 基本重構
```python
# 將扁平化輸出重構為4D張量
flat_output = model(inputs)  # [4, 1, 15]
structured = TrafficLSTM.parse_multi_vd_output(
    flat_output, 
    num_vds=3, 
    num_features=5
)
print(structured.shape)  # [4, 1, 3, 5]
```

##### 提取特定VD
```python
# 提取特定VD的預測 (例如：VD_001)
vd_001_prediction = TrafficLSTM.extract_vd_prediction(structured, vd_index=1)
print(vd_001_prediction.shape)  # [4, 1, 5]

# 提取VD_001的速度預測
vd_001_speed = vd_001_prediction[:, :, 1]  # [4, 1]
```

##### 批次處理範例
```python
# 完整的預測和解析流程
def predict_and_parse_multi_vd(model, inputs, num_vds, num_features):
    """
    完整的多VD預測和解析流程
    
    Args:
        model: TrafficLSTM 模型 (multi_vd_mode=True)
        inputs: 輸入張量 [batch, seq, num_vds * num_features]
        num_vds: VD數量
        num_features: 每VD特徵數
    
    Returns:
        Dict[str, torch.Tensor]: 每個VD的預測結果
    """
    model.eval()
    with torch.no_grad():
        # 模型預測
        flat_output = model(inputs)
        
        # 重構為結構化格式
        structured = TrafficLSTM.parse_multi_vd_output(
            flat_output, num_vds, num_features
        )
        
        # 分離每個VD的預測
        vd_predictions = {}
        for vd_idx in range(num_vds):
            vd_pred = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
            vd_predictions[f'VD_{vd_idx:03d}'] = vd_pred
    
    return vd_predictions

# 使用範例
vd_results = predict_and_parse_multi_vd(model, inputs, num_vds=3, num_features=5)
print(f"VD_000 預測形狀: {vd_results['VD_000'].shape}")  # [4, 1, 5]
print(f"VD_001 速度預測: {vd_results['VD_001'][:, :, 1]}")  # [4, 1]
```

## 為什麼使用扁平化輸出？

### 技術原因

1. **統一訓練邏輯**：不同聚合模式 (flatten, attention, pooling) 都能產生一致的輸出格式
2. **簡化損失計算**：扁平化的預測和目標可以直接計算MSE等損失
3. **LSTM架構契合**：LSTM自然輸出2D張量，扁平化避免額外的reshape操作

### 設計權衡

#### 優點
- ✅ 訓練過程簡單統一
- ✅ 不同聚合策略易於實現
- ✅ 損失計算直接明了
- ✅ 與現有LSTM架構自然契合

#### 缺點
- ❌ 需要額外的解析步驟
- ❌ 輸出語義不夠直觀
- ❌ 增加使用複雜度

## 未來擴展

### xLSTM 模型輸出
未來的 xLSTM 模型預期會採用相同的輸出格式：
```python
# 單VD xLSTM
output_shape = [batch_size, prediction_length, num_features]

# 多VD xLSTM  
output_shape = [batch_size, prediction_length, num_vds * num_features]
# 使用相同的解析方法
```

### Social-xLSTM 模型輸出
Social-xLSTM 可能需要更複雜的輸出格式：
```python
# 可能的輸出格式
{
    'predictions': [batch_size, prediction_length, num_vds * num_features],
    'attention_weights': [batch_size, num_vds, num_vds],
    'spatial_features': [batch_size, num_vds, spatial_dim]
}
# 將需要專門的解析邏輯
```

## 最佳實踐

### 訓練時
```python
# 保持扁平化格式，直接計算損失
outputs = model(inputs)  # [batch, 1, num_vds * num_features]
targets = flat_targets   # [batch, 1, num_vds * num_features]
loss = criterion(outputs, targets)
```

### 評估時
```python
# 重構為結構化格式，按VD分析
structured_pred = TrafficLSTM.parse_multi_vd_output(outputs, num_vds, num_features)
structured_true = TrafficLSTM.parse_multi_vd_output(targets, num_vds, num_features)

# 計算每VD的指標
for vd_idx in range(num_vds):
    vd_pred = TrafficLSTM.extract_vd_prediction(structured_pred, vd_idx)
    vd_true = TrafficLSTM.extract_vd_prediction(structured_true, vd_idx)
    vd_mae = torch.mean(torch.abs(vd_pred - vd_true))
    print(f"VD_{vd_idx:03d} MAE: {vd_mae.item():.4f}")
```

### 預測時
```python
# 根據需求選擇格式
if need_individual_vd_analysis:
    structured = TrafficLSTM.parse_multi_vd_output(outputs, num_vds, num_features)
    # 進行VD級別分析
else:
    # 直接使用扁平化輸出進行整體分析
    pass
```

## 錯誤排除

### 常見錯誤

1. **維度不匹配**
```python
# 錯誤：忘記考慮VD數量
TrafficLSTM.parse_multi_vd_output(output, num_vds=2, num_features=5)  
# 但實際輸出是3VD × 5特徵 = 15維

# 正確：確保參數匹配實際模型配置
TrafficLSTM.parse_multi_vd_output(output, num_vds=3, num_features=5)
```

2. **VD索引越界**
```python
# 錯誤：VD索引超出範圍
TrafficLSTM.extract_vd_prediction(structured, vd_index=3)  # 但只有3個VD (索引0-2)

# 正確：檢查VD數量
num_vds = structured.shape[2]
assert vd_index < num_vds, f"VD index {vd_index} out of range [0, {num_vds})"
```

## 相關文檔

- [LSTM 使用指南](../guides/lstm_usage_guide.md)
- [Multi-VD 訓練指南](../guides/training_scripts_guide.md)
- [已知錯誤記錄](./known_errors.md)

---

**更新記錄**：
- 2025-07-13: 初始版本，涵蓋 TrafficLSTM 的輸出格式和解析方法
- 未來更新: 將添加 xLSTM 和 Social-xLSTM 的輸出格式說明