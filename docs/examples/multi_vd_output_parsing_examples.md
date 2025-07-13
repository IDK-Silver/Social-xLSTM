# Multi-VD 輸出解析使用範例

本文檔提供 Multi-VD TrafficLSTM 模型輸出解析的實際使用範例。

## 基本使用

### 簡單的預測和解析

```python
import torch
from social_xlstm.models.lstm import TrafficLSTM

# 創建 Multi-VD 模型
model = TrafficLSTM.create_multi_vd_model(
    num_vds=3,
    input_size=5,
    hidden_size=128,
    num_layers=2
)

# 準備輸入數據 (扁平化格式)
batch_size, seq_len = 4, 12
num_vds, num_features = 3, 5
inputs = torch.randn(batch_size, seq_len, num_vds * num_features)

# 模型預測
model.eval()
with torch.no_grad():
    flat_output = model(inputs)
    print(f"模型輸出形狀: {flat_output.shape}")  # [4, 1, 15]

# 解析輸出
structured_output = TrafficLSTM.parse_multi_vd_output(
    flat_output, 
    num_vds=3, 
    num_features=5
)
print(f"結構化輸出: {structured_output.shape}")  # [4, 1, 3, 5]
```

### 提取特定VD的預測

```python
# 提取 VD_001 的預測
vd_001_prediction = TrafficLSTM.extract_vd_prediction(structured_output, vd_index=1)
print(f"VD_001 預測: {vd_001_prediction.shape}")  # [4, 1, 5]

# 提取 VD_001 的速度預測 (假設速度是第1個特徵)
vd_001_speed = vd_001_prediction[:, :, 1]
print(f"VD_001 速度預測: {vd_001_speed.shape}")  # [4, 1]
```

## 實際應用場景

### 場景1: 訓練後的模型評估

```python
def evaluate_multi_vd_model(model, test_loader, num_vds, num_features):
    """
    評估 Multi-VD 模型，計算每個VD的指標
    """
    model.eval()
    vd_maes = {f'VD_{i:03d}': [] for i in range(num_vds)}
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_seq']
            targets = batch['target_seq']
            
            # 模型預測
            flat_predictions = model(inputs)
            
            # 解析預測和目標
            pred_structured = TrafficLSTM.parse_multi_vd_output(
                flat_predictions, num_vds, num_features
            )
            true_structured = TrafficLSTM.parse_multi_vd_output(
                targets, num_vds, num_features
            )
            
            # 計算每個VD的MAE
            for vd_idx in range(num_vds):
                vd_pred = TrafficLSTM.extract_vd_prediction(pred_structured, vd_idx)
                vd_true = TrafficLSTM.extract_vd_prediction(true_structured, vd_idx)
                
                mae = torch.mean(torch.abs(vd_pred - vd_true))
                vd_maes[f'VD_{vd_idx:03d}'].append(mae.item())
    
    # 計算平均MAE
    avg_maes = {}
    for vd_name, mae_list in vd_maes.items():
        avg_maes[vd_name] = sum(mae_list) / len(mae_list)
        print(f"{vd_name} 平均 MAE: {avg_maes[vd_name]:.4f}")
    
    return avg_maes

# 使用範例
# avg_maes = evaluate_multi_vd_model(model, test_loader, num_vds=3, num_features=5)
```

### 場景2: 實時預測與結果展示

```python
def predict_and_display(model, current_data, vd_names=None, feature_names=None):
    """
    進行預測並友善地顯示結果
    """
    if feature_names is None:
        feature_names = ['volume', 'speed', 'occupancy', 'density', 'flow']
    
    if vd_names is None:
        vd_names = [f'VD_{i:03d}' for i in range(current_data.shape[-2])]
    
    # 準備輸入數據
    batch_size, seq_len, num_vds, num_features = current_data.shape
    flat_inputs = current_data.view(batch_size, seq_len, num_vds * num_features)
    
    # 模型預測
    model.eval()
    with torch.no_grad():
        flat_predictions = model(flat_inputs)
        structured_predictions = TrafficLSTM.parse_multi_vd_output(
            flat_predictions, num_vds, num_features
        )
    
    # 顯示結果
    print("🚦 交通預測結果")
    print("=" * 60)
    
    for vd_idx in range(num_vds):
        vd_prediction = TrafficLSTM.extract_vd_prediction(structured_predictions, vd_idx)
        vd_name = vd_names[vd_idx] if vd_idx < len(vd_names) else f'VD_{vd_idx:03d}'
        
        print(f"\n📍 {vd_name}:")
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_idx < num_features:
                value = vd_prediction[0, 0, feat_idx].item()  # 第一個樣本
                print(f"  {feat_name:12}: {value:8.3f}")

# 使用範例
# current_traffic_data = torch.randn(1, 12, 3, 5)  # 1個樣本，12時間步，3VD，5特徵
# vd_names = ['台南_VD_001', '台南_VD_002', '台南_VD_003']
# predict_and_display(model, current_traffic_data, vd_names)
```

### 場景3: 批次預測處理

```python
def batch_predict_all_vds(model, data_loader, num_vds, num_features):
    """
    批次預測所有VD的結果
    """
    model.eval()
    all_vd_predictions = {f'VD_{i:03d}': [] for i in range(num_vds)}
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_seq']
            
            # 模型預測
            flat_predictions = model(inputs)
            
            # 解析並分離各VD
            structured_predictions = TrafficLSTM.parse_multi_vd_output(
                flat_predictions, num_vds, num_features
            )
            
            for vd_idx in range(num_vds):
                vd_prediction = TrafficLSTM.extract_vd_prediction(
                    structured_predictions, vd_idx
                )
                all_vd_predictions[f'VD_{vd_idx:03d}'].append(vd_prediction)
    
    # 合併所有批次的結果
    final_predictions = {}
    for vd_name in all_vd_predictions:
        final_predictions[vd_name] = torch.cat(
            all_vd_predictions[vd_name], dim=0
        )
        print(f"{vd_name} 總預測形狀: {final_predictions[vd_name].shape}")
    
    return final_predictions

# 使用範例
# all_predictions = batch_predict_all_vds(model, test_loader, num_vds=3, num_features=5)
```

## 進階應用

### 特徵級別分析

```python
def analyze_feature_predictions(structured_predictions, feature_names=None):
    """
    分析各個特徵的預測結果
    """
    if feature_names is None:
        feature_names = ['volume', 'speed', 'occupancy', 'density', 'flow']
    
    batch_size, seq_len, num_vds, num_features = structured_predictions.shape
    
    print("📊 特徵預測統計")
    print("=" * 50)
    
    for feat_idx, feat_name in enumerate(feature_names[:num_features]):
        # 提取所有VD的此特徵
        feature_values = structured_predictions[:, :, :, feat_idx]  # [batch, seq, vds]
        
        print(f"\n🔸 {feat_name}:")
        print(f"  全體平均: {feature_values.mean().item():.4f}")
        print(f"  標準差: {feature_values.std().item():.4f}")
        print(f"  最小值: {feature_values.min().item():.4f}")
        print(f"  最大值: {feature_values.max().item():.4f}")
        
        # 各VD的統計
        for vd_idx in range(num_vds):
            vd_feature = feature_values[:, :, vd_idx]
            print(f"    VD_{vd_idx:03d} 平均: {vd_feature.mean().item():.4f}")
```

### 空間相關性分析

```python
def analyze_spatial_correlation(structured_predictions, vd_names=None):
    """
    分析VD之間的空間相關性
    """
    batch_size, seq_len, num_vds, num_features = structured_predictions.shape
    
    if vd_names is None:
        vd_names = [f'VD_{i:03d}' for i in range(num_vds)]
    
    print("🌐 VD空間相關性分析")
    print("=" * 50)
    
    # 計算VD之間的相關係數
    correlations = torch.zeros(num_vds, num_vds)
    
    for i in range(num_vds):
        for j in range(num_vds):
            vd_i_data = structured_predictions[:, :, i, :].flatten()
            vd_j_data = structured_predictions[:, :, j, :].flatten()
            
            # 計算皮爾森相關係數
            corr = torch.corrcoef(torch.stack([vd_i_data, vd_j_data]))[0, 1]
            correlations[i, j] = corr
    
    # 顯示相關性矩陣
    print("\n相關性矩陣:")
    print("    ", end="")
    for name in vd_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name_i in enumerate(vd_names):
        print(f"{name_i:<8}", end="")
        for j in range(num_vds):
            print(f"{correlations[i, j].item():8.3f}", end="")
        print()
    
    return correlations
```

## 錯誤處理範例

### 常見錯誤及解決方法

```python
def safe_parse_multi_vd_output(flat_output, num_vds, num_features):
    """
    安全的Multi-VD輸出解析，包含錯誤處理
    """
    try:
        # 驗證輸入
        if not isinstance(flat_output, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(flat_output)}")
        
        if flat_output.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {flat_output.dim()}D")
        
        # 解析輸出
        structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
        return structured, None
        
    except ValueError as e:
        error_msg = f"維度錯誤: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg
        
    except Exception as e:
        error_msg = f"未知錯誤: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg

def safe_extract_vd_prediction(structured_output, vd_index):
    """
    安全的VD預測提取，包含錯誤處理
    """
    try:
        vd_prediction = TrafficLSTM.extract_vd_prediction(structured_output, vd_index)
        return vd_prediction, None
        
    except IndexError as e:
        error_msg = f"VD索引錯誤: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg
        
    except Exception as e:
        error_msg = f"未知錯誤: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg

# 使用範例
flat_output = torch.randn(4, 1, 15)
structured, error = safe_parse_multi_vd_output(flat_output, num_vds=3, num_features=5)

if structured is not None:
    vd_001, error = safe_extract_vd_prediction(structured, vd_index=1)
    if vd_001 is not None:
        print(f"✅ 成功提取 VD_001: {vd_001.shape}")
```

## 性能優化建議

### 避免重複解析

```python
# ❌ 低效：重複解析
for vd_idx in range(num_vds):
    structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
    vd_pred = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
    process_vd_prediction(vd_pred)

# ✅ 高效：一次解析，多次使用
structured = TrafficLSTM.parse_multi_vd_output(flat_output, num_vds, num_features)
for vd_idx in range(num_vds):
    vd_pred = TrafficLSTM.extract_vd_prediction(structured, vd_idx)
    process_vd_prediction(vd_pred)
```

### 批次處理優化

```python
# ✅ 對於大批次數據，考慮分批處理
def process_large_batch(flat_outputs, num_vds, num_features, chunk_size=100):
    """
    分批處理大型數據集
    """
    total_samples = flat_outputs.shape[0]
    results = []
    
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk = flat_outputs[start_idx:end_idx]
        
        structured_chunk = TrafficLSTM.parse_multi_vd_output(
            chunk, num_vds, num_features
        )
        results.append(structured_chunk)
    
    return torch.cat(results, dim=0)
```

## 相關文檔

- [模型輸出格式技術文檔](../technical/output_formats_and_parsing.md)
- [LSTM 使用指南](../guides/lstm_usage_guide.md)
- [Multi-VD 訓練指南](../guides/training_scripts_guide.md)

---

**最後更新**: 2025-07-13  
**版本**: 1.0