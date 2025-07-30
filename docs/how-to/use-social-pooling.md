# 如何使用 Social Pooling

本指南提供 Social Pooling 的完整使用說明，從快速入門到進階整合。

## 📋 快速導覽

- [概念理解](#概念理解)  
- [快速開始](#快速開始)
- [配置選項](#配置選項)
- [進階整合](#進階整合)
- [故障排除](#故障排除)

## 🧠 概念理解

### 什麼是 Social Pooling？

Social Pooling 是一種**無拓撲依賴**的空間聚合方法，通過地理座標而非預定義的網路結構來學習交通節點間的空間關係。

### 核心原理

```python
def social_pooling(node_features, coordinates, radius):
    # 1. 計算距離矩陣（基於座標）
    distances = compute_distance_matrix(coordinates)
    
    # 2. 生成空間權重（高斯核函數）  
    spatial_weights = gaussian_kernel(distances, radius)
    
    # 3. 加權聚合鄰居特徵
    pooled_features = weighted_aggregation(node_features, spatial_weights)
    
    return pooled_features
```

### 優勢特點

- **無拓撲依賴**：純粹基於座標的空間關係學習
- **適應不規則分佈**：感測器位置不規則時仍能有效工作  
- **動態擴展**：易於添加新的空間節點
- **可解釋性**：基於物理距離的直觀理解

## 🚀 快速開始

### 5 分鐘體驗

```python
import torch
import numpy as np
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig

# 1. 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)

# 2. 準備測試數據
batch_size, seq_len, feature_dim, num_vds = 8, 12, 3, 20
features = torch.randn(batch_size, seq_len, feature_dim)
coordinates = torch.randn(num_vds, 2) * 2000  # ±2km 範圍
vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]

# 3. 創建 Social Pooling 模組
config = SocialPoolingConfig(
    pooling_radius=1000.0,      # 1 公里半徑
    max_neighbors=8,            # 最多 8 個鄰居
    weighting_function="gaussian"
)
social_pooling = SocialPooling(config)

# 4. 執行空間聚合
with torch.no_grad():
    pooled_features = social_pooling(features, coordinates, vd_ids)

print(f"✓ 輸入形狀: {features.shape}")
print(f"✓ 輸出形狀: {pooled_features.shape}")
print(f"✓ 特徵變化: {torch.norm(pooled_features - features):.4f}")
```

### 完整範例腳本

```python
#!/usr/bin/env python3
"""Social Pooling 最小工作範例"""

import torch
import numpy as np
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig

def main():
    # 設定
    torch.manual_seed(42)
    
    # 數據準備
    batch_size, seq_len, feature_dim, num_vds = 8, 12, 3, 20
    features = torch.randn(batch_size, seq_len, feature_dim)
    coordinates = torch.randn(num_vds, 2) * 2000
    vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]
    
    # Social Pooling 配置
    config = SocialPoolingConfig(
        pooling_radius=1000.0,
        max_neighbors=8,
        weighting_function="gaussian"
    )
    
    # 創建模組並執行
    social_pooling = SocialPooling(config)
    pooled_features = social_pooling(features, coordinates, vd_ids)
    
    # 輸出結果
    print("🎉 Social Pooling 測試成功!")
    print(f"輸入形狀: {features.shape}")
    print(f"輸出形狀: {pooled_features.shape}")
    print(f"特徵變化程度: {torch.norm(pooled_features - features):.4f}")

if __name__ == "__main__":
    main()
```

將此腳本保存為 `test_social_pooling.py` 並執行：
```bash
python test_social_pooling.py
```

## ⚙️ 配置選項

### SocialPoolingConfig 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `pooling_radius` | 1000.0 | 池化半徑（公尺） |
| `max_neighbors` | 8 | 最大鄰居數量 |
| `weighting_function` | "gaussian" | 權重函數類型 |
| `distance_metric` | "euclidean" | 距離計算方式 |
| `aggregation_method` | "weighted_mean" | 聚合方法 |
| `enable_caching` | True | 啟用距離快取 |
| `normalize_weights` | True | 權重正規化 |

### 場景化配置

#### 城市環境（高密度）
```python
urban_config = SocialPoolingConfig(
    pooling_radius=500.0,       # 較小半徑
    max_neighbors=12,           # 更多鄰居
    weighting_function="gaussian",
    distance_metric="euclidean"
)
```

#### 高速公路環境（稀疏）
```python
highway_config = SocialPoolingConfig(
    pooling_radius=2000.0,      # 較大半徑
    max_neighbors=5,            # 較少鄰居
    weighting_function="exponential",
    enable_caching=True
)
```

#### 地理座標數據
```python
geo_config = SocialPoolingConfig(
    pooling_radius=1000.0,
    max_neighbors=8,
    distance_metric="haversine", # 球面距離
    weighting_function="gaussian"
)
```

#### 開發除錯
```python
debug_config = SocialPoolingConfig(
    pooling_radius=800.0,
    max_neighbors=3,            # 少量鄰居便於檢查
    weighting_function="linear", # 簡單權重函數
    enable_caching=False,       # 關閉快取便於除錯
    include_self=True           # 包含自身節點
)
```

## 🔧 進階整合

### 與 LSTM 整合

#### 手動組合（Post-Fusion 策略）
```python
from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig

# 1. 創建基礎 LSTM 模型
lstm_config = TrafficLSTMConfig(
    hidden_size=64,
    num_layers=2,
    dropout=0.1
)
base_model = TrafficLSTM(lstm_config)

# 2. 創建 Social Pooling
social_config = SocialPoolingConfig(pooling_radius=1000.0)
social_pooling = SocialPooling(social_config)

# 3. 手動組合前向傳播
def forward_with_social_pooling(features, coordinates, vd_ids):
    # 先進行空間聚合
    social_features = social_pooling(features, coordinates, vd_ids)
    
    # 再通過 LSTM 處理
    lstm_output = base_model(social_features)
    
    return lstm_output

# 測試整合
output = forward_with_social_pooling(features, coordinates, vd_ids)
print(f"整合模型輸出形狀: {output.shape}")
```

#### 使用工廠函數（推薦）
```python
from social_xlstm.models.social_pooling import create_social_traffic_model

# 使用工廠函數創建整合模型
integrated_model = create_social_traffic_model(
    base_model_type="lstm",
    strategy="post_fusion",
    base_config=lstm_config,
    social_config=social_config
)

# 直接使用
output = integrated_model(features, coordinates, vd_ids)
print(f"工廠模型輸出形狀: {output.shape}")
```

### 與 xLSTM 整合

```python
from social_xlstm.models import TrafficXLSTM, TrafficXLSTMConfig

# xLSTM 配置
xlstm_config = TrafficXLSTMConfig(
    embedding_dim=128,
    num_blocks=6,
    slstm_at=[1, 3]
)

# 創建整合模型
xlstm_social_model = create_social_traffic_model(
    base_model_type="xlstm",
    strategy="post_fusion", 
    base_config=xlstm_config,
    social_config=social_config
)
```

### 訓練整合模型

```python
from social_xlstm.training.trainer import Trainer, TrainingConfig

# 訓練配置（針對 Social Pooling 優化）
training_config = TrainingConfig(
    epochs=100,
    batch_size=16,              # 較小批次，因為空間計算開銷
    learning_rate=0.0008,       # 稍微降低學習率
    optimizer_type="adamw",
    weight_decay=0.01,
    early_stopping_patience=20
)

# 創建訓練器
trainer = Trainer(
    model=integrated_model,
    training_config=training_config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# 開始訓練
history = trainer.train()
```

## 🔍 效能檢查與優化

### 基本效能測試

```python
import time

def benchmark_social_pooling():
    """簡單的效能測試"""
    config = SocialPoolingConfig()
    social_pooling = SocialPooling(config)
    
    # 準備數據
    features = torch.randn(32, 12, 3)
    coordinates = torch.randn(50, 2) * 1000
    vd_ids = [f"VD_{i}" for i in range(50)]
    
    # 暖身運行
    for _ in range(5):
        _ = social_pooling(features, coordinates, vd_ids)
    
    # 計時測試
    start_time = time.time()
    for _ in range(20):
        _ = social_pooling(features, coordinates, vd_ids)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 20
    print(f"平均執行時間: {avg_time*1000:.2f} ms")
    
    return avg_time

# 執行效能測試
benchmark_social_pooling()
```

### 優化建議

#### 記憶體優化
```python
# 減少批次大小或鄰居數量
config = SocialPoolingConfig(
    max_neighbors=5,        # 減少鄰居數
    use_sparse_computation=True  # 使用稀疏計算
)
```

#### 速度優化
```python
# 啟用快取和優化設定
config = SocialPoolingConfig(
    enable_caching=True,    # 啟用距離快取
    pooling_radius=1000.0,  # 適中的半徑
    max_neighbors=8         # 適中的鄰居數
)
```

## 🚨 故障排除

### 常見問題與解決方案

#### 1. 記憶體不足
```python
# 問題：RuntimeError: CUDA out of memory
# 解決：減少計算複雜度
config = SocialPoolingConfig(
    max_neighbors=5,
    use_sparse_computation=True,
    batch_processing=True
)
```

#### 2. 計算速度慢
```python
# 問題：執行時間過長
# 解決：優化配置
config = SocialPoolingConfig(
    enable_caching=True,
    pooling_radius=1000.0,  # 不要設太大
    max_neighbors=8         # 適中數量
)
```

#### 3. 座標數據格式錯誤
```python
# 問題：tensor 格式或數值異常
# 解決：確保數據格式正確
coordinates = coordinates.float()  # 確保是 float 類型
assert coordinates.shape[1] == 2, "座標必須是 (N, 2) 形狀"
assert not torch.isnan(coordinates).any(), "座標不能包含 NaN"
```

#### 4. VD 識別碼不匹配
```python
# 問題：ID 數量與座標不匹配
# 解決：確保數量一致
assert len(vd_ids) == coordinates.shape[0], "VD 識別碼數量必須與座標數量匹配"
```

### 調試技巧

#### 啟用詳細日誌
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Social Pooling 模組會輸出詳細的計算資訊
```

#### 檢查中間結果
```python
# 檢查距離計算
distances = social_pooling.calculate_distances(coordinates)
print(f"距離矩陣形狀: {distances.shape}")
print(f"最大距離: {distances.max():.2f}m")

# 檢查權重分佈
weights = social_pooling.compute_weights(distances)
print(f"權重範圍: {weights.min():.4f} - {weights.max():.4f}")
```

## 📊 結果分析

### 空間聚合效果評估

```python
# 分析聚合效果
original_std = features.std(dim=0).mean()
pooled_std = pooled_features.std(dim=0).mean()

print(f"原始特徵標準差: {original_std:.4f}")
print(f"池化特徵標準差: {pooled_std:.4f}")
print(f"平滑程度: {(original_std - pooled_std) / original_std * 100:.1f}%")
```

### 權重矩陣分析

```python
# 檢查權重計算合理性
distances = social_pooling.calculate_distances(coordinates)
weights = social_pooling.compute_weights(distances)

print(f"距離矩陣形狀: {distances.shape}")
print(f"權重矩陣形狀: {weights.shape}")
print(f"最大權重: {weights.max():.4f}")
print(f"平均權重: {weights.mean():.4f}")

# 視覺化權重分佈（可選）
import matplotlib.pyplot as plt
plt.hist(weights.flatten().numpy(), bins=50)
plt.title("Weight Distribution")
plt.savefig("weight_distribution.png")
```

## 📚 下一步發展

### 學習路徑
1. **深入理解**：閱讀 `docs_rebuild/explanation/social-pooling-design.md`
2. **完整整合**：查看訓練整合範例
3. **性能調優**：參考效能最佳化建議
4. **進階功能**：探索多種聚合策略

### 相關資源
- [核心技術決策](../explanation/key-decisions.md)
- [如何訓練模型](train-models.md)
- [驗證和調試指南](validate-and-debug.md)
- [專案狀態](../PROJECT_STATUS.md)

---

Social Pooling 提供了一種創新的空間建模方法，適合交通預測中的無拓撲場景。通過這個指南，你應該能夠成功整合和使用 Social Pooling 功能。