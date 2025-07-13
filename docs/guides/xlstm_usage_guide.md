# TrafficXLSTM 使用指南

## 📖 概述

TrafficXLSTM 是基於 extended LSTM (xLSTM) 架構的交通預測模型，實現了 2024 年最新的 xLSTM 技術，結合 sLSTM（標量 LSTM）和 mLSTM（矩陣 LSTM）的混合架構。

### 🎯 設計目標

- **創新性**: 探索 xLSTM 在交通預測中的應用
- **性能提升**: 改善傳統 LSTM 的過擬合問題
- **長期記憶**: 利用指數門控機制建模長期時間依賴
- **空間建模**: 支援未來與 Social Pooling 的整合

## 🏗️ 架構特點

### xLSTM 混合架構
```
總共 6 個區塊:
- sLSTM 位置: [1, 3] - 處理時間序列特徵
- mLSTM 位置: [0, 2, 4, 5] - 處理空間特徵
```

### 核心組件
1. **輸入嵌入層**: 將交通特徵投影到高維空間
2. **xLSTM 區塊堆疊**: 混合 sLSTM 和 mLSTM 的處理
3. **輸出投影層**: 生成最終預測結果
4. **正則化**: Dropout 防止過擬合

## 🚀 快速開始

### 基本使用

```python
from social_xlstm.models import TrafficXLSTM, TrafficXLSTMConfig
import torch

# 1. 創建配置
config = TrafficXLSTMConfig(
    input_size=3,          # [volume, speed, occupancy]
    embedding_dim=128,     # 嵌入維度
    num_blocks=6,          # xLSTM 區塊數
    slstm_at=[1, 3],      # sLSTM 位置
    dropout=0.1,          # Dropout 率
    context_length=256    # 上下文長度
)

# 2. 初始化模型
model = TrafficXLSTM(config)

# 3. 準備數據
batch_size, seq_len, input_size = 4, 12, 3
x = torch.randn(batch_size, seq_len, input_size)

# 4. 前向傳播
output = model(x)  # shape: (4, 1, 3)

print(f"輸入形狀: {x.shape}")
print(f"輸出形狀: {output.shape}")
```

### 模型資訊

```python
# 獲取模型資訊
info = model.get_model_info()
print(f"模型類型: {info['model_type']}")
print(f"總參數: {info['total_parameters']:,}")
print(f"可訓練參數: {info['trainable_parameters']:,}")
print(f"xLSTM 區塊數: {info['num_blocks']}")
print(f"sLSTM 位置: {info['slstm_positions']}")
```

## ⚙️ 配置選項

### TrafficXLSTMConfig 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `input_size` | 3 | 輸入特徵數（音量、速度、佔有率） |
| `embedding_dim` | 128 | 嵌入層維度 |
| `hidden_size` | 128 | 隱藏層大小 |
| `num_blocks` | 6 | xLSTM 區塊總數 |
| `output_size` | 3 | 輸出特徵數 |
| `sequence_length` | 12 | 輸入序列長度 |
| `prediction_length` | 1 | 預測時間步數 |
| `slstm_at` | [1, 3] | sLSTM 區塊位置 |
| `slstm_backend` | "vanilla" | sLSTM 計算後端 |
| `mlstm_backend` | "vanilla" | mLSTM 計算後端 |
| `context_length` | 256 | xLSTM 上下文長度 |
| `dropout` | 0.1 | Dropout 率 |
| `multi_vd_mode` | False | 是否啟用多 VD 模式 |

### 自定義配置範例

```python
# 小型模型 - 適合快速實驗
small_config = TrafficXLSTMConfig(
    embedding_dim=64,
    num_blocks=4,
    slstm_at=[1],
    dropout=0.2
)

# 大型模型 - 適合完整訓練
large_config = TrafficXLSTMConfig(
    embedding_dim=256,
    num_blocks=8,
    slstm_at=[1, 3, 5],
    dropout=0.1,
    context_length=512
)

# 多 VD 模式
multi_vd_config = TrafficXLSTMConfig(
    multi_vd_mode=True,
    num_vds=5,
    embedding_dim=128
)
```

## 🎮 進階使用

### 設備管理

```python
# 自動檢測設備
model = TrafficXLSTM(config)  # 自動使用 CUDA 或 CPU

# 手動指定設備
model.to_device("cpu")
model.to_device("cuda:0")

# 檢查當前設備
print(f"模型設備: {config.device}")
```

### 訓練模式切換

```python
# 訓練模式（啟用 Dropout）
model.train()
output = model(x)

# 評估模式（關閉 Dropout）
model.eval()
with torch.no_grad():
    output = model(x)
```

### 梯度計算

```python
# 啟用梯度計算
x = torch.randn(2, 12, 3, requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()

# 檢查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: 梯度範數 = {param.grad.norm().item():.6f}")
```

## 🔍 與 LSTM 的比較

| 特點 | TrafficLSTM | TrafficXLSTM |
|------|-------------|--------------|
| 架構 | 傳統 LSTM | 混合 sLSTM + mLSTM |
| 門控機制 | Sigmoid | 指數門控 |
| 記憶容量 | 有限 | 更大容量 |
| 長期依賴 | 較弱 | 更強 |
| 參數量 | ~226K | ~655K |
| 計算複雜度 | 較低 | 較高 |
| 過擬合風險 | 較高 | 預期較低 |

## 📊 性能期望

### 目標改善
- **減少過擬合**: 改善 LSTM 的負 R² 問題
- **更好收斂**: 訓練/驗證指標差距 < 2 倍
- **長期預測**: 更好的長時間序列建模能力

### 記憶體使用
```python
# 檢查模型大小
total_params = sum(p.numel() for p in model.parameters())
model_size_mb = total_params * 4 / (1024**2)  # 假設 float32
print(f"模型大小: {model_size_mb:.2f} MB")
```

## 🧪 測試與驗證

### 運行單元測試
```bash
# 完整測試
pytest tests/test_social_xlstm/models/test_xlstm.py -v

# 特定測試類別
pytest tests/test_social_xlstm/models/test_xlstm.py::TestTrafficXLSTM -v

# 配置相關測試
pytest tests/test_social_xlstm/models/test_xlstm.py -k "config" -v
```

### 功能驗證
```python
# 基本功能測試
def test_basic_functionality():
    config = TrafficXLSTMConfig()
    model = TrafficXLSTM(config)
    
    # 測試輸入輸出
    x = torch.randn(2, 12, 3)
    output = model(x)
    
    assert output.shape == (2, 1, 3)
    assert not torch.isnan(output).any()
    print("✅ 基本功能正常")

test_basic_functionality()
```

## 🔮 未來發展

### 計劃整合
1. **Social Pooling**: 與空間聚合機制結合
2. **多 VD 支援**: 完整的多車輛檢測器模式
3. **性能優化**: 考慮使用 TFLA 優化內核
4. **超參數調優**: 自動化的參數搜索

### 研究方向
- xLSTM 在交通預測中的有效性分析
- 與傳統 LSTM 的系統性比較
- 不同 sLSTM/mLSTM 配置的影響
- 長期預測能力評估

## 📚 相關文檔

- [ADR-0501: xLSTM 整合策略](../adr/0501-xlstm-integration-strategy.md)
- [ADR-0101: xLSTM vs Traditional LSTM](../adr/0101-xlstm-vs-traditional-lstm.md)
- [LSTM 使用指南](lstm_usage_guide.md)
- [專案待辦事項](../todo.md)

## 🆘 常見問題

### Q: 為什麼參數量比 LSTM 多這麼多？
A: xLSTM 使用更複雜的區塊結構和更大的嵌入維度，但這帶來了更強的建模能力。

### Q: 如何選擇 sLSTM 位置？
A: 預設 [1, 3] 是基於時間序列處理的經驗配置，可根據具體數據特性調整。

### Q: 訓練速度比 LSTM 慢嗎？
A: 是的，但我們優先考慮模型效果。後續可考慮性能優化。

### Q: 如何與現有訓練流程整合？
A: TrafficXLSTM 與 TrafficLSTM 具有相同的接口，可以直接替換使用。

---

**更新日期**: 2025-07-13  
**版本**: 1.0  
**作者**: Social-xLSTM Project Team