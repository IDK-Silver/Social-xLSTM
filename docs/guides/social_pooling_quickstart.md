# Social Pooling 快速開始指南

## 概述

這份指南將幫助您在 5 分鐘內開始使用 Social Pooling 功能。我們將涵蓋基本設定、簡單範例和常見配置模式。

## 先決條件

確保您已經完成以下設定：

```bash
# 1. 啟用 conda 環境
conda activate social_xlstm

# 2. 確認套件已安裝
python -c "import social_xlstm; print('✓ Package installed')"

# 3. 檢查 GPU 可用性（可選）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 5 分鐘快速體驗

### 步驟 1: 基本導入

```python
import torch
import numpy as np
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig
from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig

# 設定隨機種子以確保結果可重現
torch.manual_seed(42)
np.random.seed(42)
```

### 步驟 2: 準備測試數據

```python
# 模擬交通數據參數
batch_size = 8      # 批次大小
seq_len = 12        # 序列長度（12 個時間點）
feature_dim = 3     # 特徵維度 [volume, speed, occupancy]
num_vds = 20        # VD 數量

# 生成模擬的交通特徵數據
features = torch.randn(batch_size, seq_len, feature_dim)

# 生成模擬的 VD 座標（範圍 ±2000 公尺）
coordinates = torch.randn(num_vds, 2) * 2000

# 生成 VD 識別碼
vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]

print(f"✓ 數據準備完成:")
print(f"  - 特徵形狀: {features.shape}")
print(f"  - 座標形狀: {coordinates.shape}")
print(f"  - VD 數量: {len(vd_ids)}")
```

### 步驟 3: 創建 Social Pooling 模組

```python
# 基本配置
config = SocialPoolingConfig(
    pooling_radius=1000.0,      # 1 公里半徑
    max_neighbors=8,            # 最多 8 個鄰居
    weighting_function="gaussian",  # 高斯權重函數
    distance_metric="euclidean"     # 歐幾里得距離
)

# 創建 Social Pooling 模組
social_pooling = SocialPooling(config)

print(f"✓ Social Pooling 模組已創建")
print(f"  - 池化半徑: {config.pooling_radius} 公尺")
print(f"  - 最大鄰居數: {config.max_neighbors}")
```

### 步驟 4: 執行空間聚合

```python
# 執行前向傳播
with torch.no_grad():  # 推論模式，不計算梯度
    pooled_features = social_pooling(features, coordinates, vd_ids)

print(f"✓ 空間聚合完成:")
print(f"  - 原始特徵: {features.shape}")
print(f"  - 池化特徵: {pooled_features.shape}")
print(f"  - 特徵變化: {torch.norm(pooled_features - features):.4f}")
```

### 步驟 5: 基本結果分析

```python
# 分析空間聚合效果
original_std = features.std(dim=0).mean()
pooled_std = pooled_features.std(dim=0).mean()

print(f"✓ 聚合效果分析:")
print(f"  - 原始特徵標準差: {original_std:.4f}")
print(f"  - 池化特徵標準差: {pooled_std:.4f}")
print(f"  - 平滑程度: {(original_std - pooled_std) / original_std * 100:.1f}%")

# 檢查權重計算
distances = social_pooling.calculate_distances(coordinates)
weights = social_pooling.compute_weights(distances)

print(f"✓ 權重矩陣分析:")
print(f"  - 距離矩陣形狀: {distances.shape}")
print(f"  - 權重矩陣形狀: {weights.shape}")
print(f"  - 最大權重: {weights.max():.4f}")
print(f"  - 平均權重: {weights.mean():.4f}")
```

## 完整的最小範例

將以上所有步驟組合成一個完整的腳本：

```python
#!/usr/bin/env python3
"""
Social Pooling 最小工作範例
展示基本的空間聚合功能
"""

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

## 常見配置模式

### 1. 高精度配置（城市環境）

```python
# 適用於密集的城市交通網絡
urban_config = SocialPoolingConfig(
    pooling_radius=500.0,       # 較小半徑
    max_neighbors=12,           # 更多鄰居
    weighting_function="gaussian",
    distance_metric="euclidean",
    aggregation_method="weighted_mean"
)
```

### 2. 高效配置（高速公路環境）

```python
# 適用於高速公路或稀疏網絡
highway_config = SocialPoolingConfig(
    pooling_radius=2000.0,      # 較大半徑
    max_neighbors=5,            # 較少鄰居
    weighting_function="exponential",
    enable_caching=True,        # 啟用快取
    use_sparse_computation=False
)
```

### 3. 地理座標配置

```python
# 適用於真實地理座標數據
geo_config = SocialPoolingConfig(
    pooling_radius=1000.0,
    max_neighbors=8,
    distance_metric="haversine", # 球面距離
    weighting_function="gaussian",
    normalize_weights=True
)
```

### 4. 開發除錯配置

```python
# 適用於開發和除錯
debug_config = SocialPoolingConfig(
    pooling_radius=800.0,
    max_neighbors=3,            # 少量鄰居便於檢查
    weighting_function="linear", # 簡單權重函數
    enable_caching=False,       # 關閉快取便於除錯
    include_self=True           # 包含自身節點
)
```

## 與現有模型整合

### 基本整合範例

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

# 3. 手動組合（Post-Fusion 策略）
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

### 使用工廠函數（推薦）

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

## 常見問題排除

### 問題 1: 記憶體不足

```python
# 解決方案：減少批次大小或鄰居數量
config = SocialPoolingConfig(
    max_neighbors=5,        # 減少鄰居數
    use_sparse_computation=True  # 使用稀疏計算
)
```

### 問題 2: 計算速度慢

```python
# 解決方案：啟用快取和優化設定
config = SocialPoolingConfig(
    enable_caching=True,    # 啟用距離快取
    pooling_radius=1000.0,  # 適中的半徑
    max_neighbors=8         # 適中的鄰居數
)
```

### 問題 3: 座標數據格式錯誤

```python
# 確保座標格式正確
coordinates = coordinates.float()  # 確保是 float 類型
assert coordinates.shape[1] == 2, "座標必須是 (N, 2) 形狀"
assert not torch.isnan(coordinates).any(), "座標不能包含 NaN"
```

### 問題 4: VD 識別碼不匹配

```python
# 確保 VD 識別碼列表長度正確
assert len(vd_ids) == coordinates.shape[0], "VD 識別碼數量必須與座標數量匹配"
```

## 效能檢查

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

## 下一步

現在您已經掌握了 Social Pooling 的基本用法，建議您：

1. **深入學習**: 閱讀 [API 參考文檔](../implementation/social_pooling_api.md)
2. **進階整合**: 查看 [整合指南](social_pooling_integration_guide.md)
3. **效能調優**: 參考 [效能最佳化指南](../technical/social_pooling_optimization.md)
4. **問題排除**: 查閱 [故障排除指南](social_pooling_troubleshooting.md)

## 相關資源

- [Social Pooling API 參考](../implementation/social_pooling_api.md)
- [配置選項詳解](../implementation/social_pooling_config.md)
- [整合策略比較](../technical/social_pooling_strategies.md)
- [開發工作流程](social_pooling_development.md)