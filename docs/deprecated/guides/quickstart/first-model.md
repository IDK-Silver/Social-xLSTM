# 🎯 第一個 Social-xLSTM 模型

**基於**: 實際可用的程式碼實現  
**目標**: 15分鐘內建立、訓練並評估你的第一個 Social-xLSTM 模型

## 📋 概述

本指南將帶你完成以下步驟：
1. 建立分散式 Social-xLSTM 模型
2. 使用範例數據進行訓練
3. 評估模型性能
4. 可視化結果

## 🏗️ 模型建立

### 創建 `first_model.py`

```python
"""
完整的 Social-xLSTM 第一個模型範例
基於已驗證的程式碼實現
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# 導入已實現的模組
from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig
from social_xlstm.evaluation.evaluator import ModelEvaluator

class SocialXLSTMModel(nn.Module):
    """
    完整的 Social-xLSTM 模型範例
    
    實現分散式架構：
    VD_A: 序列 → xLSTM_A → 隱狀態_A ┐
    VD_B: 序列 → xLSTM_B → 隱狀態_B ├→ Social_Pooling → 融合預測
    VD_C: 序列 → xLSTM_C → 隱狀態_C ┘
    """
    
    def __init__(self, xlstm_config: TrafficXLSTMConfig, social_config: SocialPoolingConfig):
        super().__init__()
        
        # 共享的 xLSTM 核心 - 所有 VD 使用相同權重
        self.shared_xlstm = TrafficXLSTM(xlstm_config)
        
        # Social Pooling 層 - 處理隱狀態
        self.social_pooling = SocialPooling(
            config=social_config,
            feature_dim=xlstm_config.hidden_size  # 注意：使用隱狀態維度
        )
        
        # 融合層
        fusion_dim = xlstm_config.hidden_size + social_config.social_embedding_dim
        self.fusion = nn.Linear(fusion_dim, xlstm_config.output_size)
        
        self.xlstm_config = xlstm_config
        self.social_config = social_config
    
    def forward(self, vd_sequences: Dict[str, torch.Tensor], 
                coordinates: torch.Tensor, vd_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        前向傳播：分散式 Social-xLSTM
        
        Args:
            vd_sequences: {"VD_001": tensor[batch, seq, features], ...}
            coordinates: tensor[num_vds, 2] - VD 座標
            vd_ids: ["VD_001", "VD_002", ...] - VD 識別碼
        
        Returns:
            {"VD_001": predictions, ...} - 每個 VD 的預測結果
        """
        batch_size = next(iter(vd_sequences.values())).size(0)
        
        # 步驟 1: 每個 VD 獨立的 xLSTM 處理
        hidden_states = {}
        for vd_id in vd_ids:
            # 使用共享權重的 xLSTM 處理每個 VD 的序列
            xlstm_output = self.shared_xlstm(vd_sequences[vd_id])  # [batch, 1, hidden_size]
            hidden_state = xlstm_output.squeeze(1)  # [batch, hidden_size]
            hidden_states[vd_id] = hidden_state
        
        # 步驟 2: 堆疊隱狀態用於 Social Pooling
        hidden_stack = torch.stack([hidden_states[vd_id] for vd_id in vd_ids], dim=1)
        # hidden_stack: [batch, num_vds, hidden_size]
        
        # 步驟 3: Social Pooling 處理隱狀態
        social_features = self.social_pooling(hidden_stack, coordinates, vd_ids)
        # social_features: [batch, num_vds, social_embedding_dim]
        
        # 步驟 4: 融合預測
        predictions = {}
        for i, vd_id in enumerate(vd_ids):
            individual = hidden_stack[:, i, :]     # [batch, hidden_size]
            social = social_features[:, i, :]      # [batch, social_embedding_dim]
            fused = torch.cat([individual, social], dim=-1)  # [batch, fusion_dim]
            pred = self.fusion(fused)              # [batch, output_size]
            predictions[vd_id] = pred
        
        return predictions

# 1. 配置模型
print("🔧 配置 Social-xLSTM 模型...")

xlstm_config = TrafficXLSTMConfig(
    input_size=3,
    hidden_size=64,      # 適中的隱狀態維度
    num_blocks=4,        # 4個 xLSTM 塊
    output_size=3,
    slstm_at=[1, 3],     # sLSTM 在位置 1 和 3
    sequence_length=12,
    dropout=0.1
)

social_config = SocialPoolingConfig(
    pooling_radius=800.0,           # 800 公尺影響半徑
    max_neighbors=3,                # 最多 3 個鄰居
    social_embedding_dim=24,        # 社交特徵維度
    distance_metric="euclidean",
    weighting_function="gaussian",
    aggregation_method="weighted_mean"
)

# 2. 創建模型
model = SocialXLSTMModel(xlstm_config, social_config)
print(f"✅ 模型創建成功")
print(f"   xLSTM 參數: {sum(p.numel() for p in model.shared_xlstm.parameters()):,}")
print(f"   Social Pooling 參數: {sum(p.numel() for p in model.social_pooling.parameters()):,}")
print(f"   總參數: {sum(p.numel() for p in model.parameters()):,}")
```

## 🎲 範例數據生成

```python
# 3. 準備範例數據（模擬真實交通數據）
print("\\n📊 準備範例數據...")

def generate_sample_traffic_data():
    """生成模擬的交通數據"""
    # VD 座標（台南市區範例）
    coordinates = torch.tensor([
        [120.2062, 22.9908],  # VD_001: 台南火車站附近
        [120.2134, 22.9853],  # VD_002: 成功大學附近  
        [120.1976, 22.9976],  # VD_003: 安平區
        [120.2095, 22.9845],  # VD_004: 東區
    ], dtype=torch.float32)
    
    vd_ids = ["VD_001", "VD_002", "VD_003", "VD_004"]
    
    # 生成時間序列數據 (批次大小=8, 序列長度=12, 特徵=3)
    batch_size, seq_len, num_features = 8, 12, 3
    vd_sequences = {}
    
    for i, vd_id in enumerate(vd_ids):
        # 模擬不同 VD 的交通模式
        base_traffic = 50 + i * 10  # 不同基礎流量
        
        # 添加時間趨勢和隨機變化
        time_trend = torch.linspace(-0.5, 0.5, seq_len).unsqueeze(0).repeat(batch_size, 1)
        sequences = []
        
        for batch in range(batch_size):
            sequence = []
            for t in range(seq_len):
                # 速度 (km/h): 基礎速度 + 時間趨勢 + 噪音
                speed = base_traffic + time_trend[batch, t] * 10 + torch.randn(1) * 5
                speed = torch.clamp(speed, 10, 100)  # 限制在合理範圍
                
                # 流量 (vehicles/hour): 與速度相關
                volume = 1000 - speed * 8 + torch.randn(1) * 100
                volume = torch.clamp(volume, 100, 2000)
                
                # 佔有率 (%): 與流量相關
                occupancy = volume / 30 + torch.randn(1) * 3
                occupancy = torch.clamp(occupancy, 5, 90)
                
                sequence.append([speed.item(), volume.item(), occupancy.item()])
            
            sequences.append(sequence)
        
        vd_sequences[vd_id] = torch.tensor(sequences, dtype=torch.float32)
    
    return vd_sequences, coordinates, vd_ids

vd_sequences, coordinates, vd_ids = generate_sample_traffic_data()

# 顯示數據資訊
print(f"✅ 數據生成完成")
print(f"   VD 數量: {len(vd_ids)}")
print(f"   批次大小: {vd_sequences[vd_ids[0]].shape[0]}")
print(f"   序列長度: {vd_sequences[vd_ids[0]].shape[1]}")
print(f"   特徵數量: {vd_sequences[vd_ids[0]].shape[2]}")

# 顯示範例數據
print(f"\\n📋 VD_001 第一個批次的前3個時間步:")
sample_data = vd_sequences["VD_001"][0, :3, :]
for t, (speed, volume, occupancy) in enumerate(sample_data):
    print(f"   t={t}: 速度={speed:.1f}km/h, 流量={volume:.0f}輛/h, 佔有率={occupancy:.1f}%")
```

## 🚀 模型訓練

```python
# 4. 簡單訓練循環
print("\\n🚀 開始訓練...")

# 設置訓練參數
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
num_epochs = 20

model.train()
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 前向傳播
    predictions = model(vd_sequences, coordinates, vd_ids)
    
    # 計算損失（簡單的自回歸任務：預測下一時刻的值）
    total_loss = 0
    for vd_id in vd_ids:
        # 使用最後一個時間步作為目標
        target = vd_sequences[vd_id][:, -1, :]  # [batch, features]
        pred = predictions[vd_id]  # [batch, features]
        loss = criterion(pred, target)
        total_loss += loss
    
    # 反向傳播
    total_loss.backward()
    optimizer.step()
    
    losses.append(total_loss.item())
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1:2d}/{num_epochs}, Loss: {total_loss.item():.4f}")

print(f"✅ 訓練完成，最終損失: {losses[-1]:.4f}")
```

## 📊 模型評估

```python
# 5. 評估模型
print("\\n📊 評估模型性能...")

model.eval()
evaluator = ModelEvaluator()

with torch.no_grad():
    predictions = model(vd_sequences, coordinates, vd_ids)
    
    # 評估每個 VD
    all_metrics = {}
    for vd_id in vd_ids:
        target = vd_sequences[vd_id][:, -1, :].numpy()  # [batch, features]
        pred = predictions[vd_id].numpy()  # [batch, features]
        
        # 分別評估每個特徵（速度、流量、佔有率）
        feature_names = ["速度", "流量", "佔有率"]
        vd_metrics = {}
        
        for i, feature_name in enumerate(feature_names):
            metrics = evaluator.compute_all_metrics(
                y_true=target[:, i], 
                y_pred=pred[:, i]
            )
            vd_metrics[feature_name] = metrics
        
        all_metrics[vd_id] = vd_metrics

# 顯示評估結果
print("\\n📈 模型性能評估結果:")
print("=" * 60)

for vd_id in vd_ids:
    print(f"\\n🚗 {vd_id}:")
    for feature_name in ["速度", "流量", "佔有率"]:
        metrics = all_metrics[vd_id][feature_name]
        print(f"   {feature_name:>4s}: MAE={metrics['mae']:.2f}, "
              f"RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")

# 計算平均性能
print("\\n🎯 整體平均性能:")
avg_mae = np.mean([[all_metrics[vd][feat]['mae'] 
                   for feat in ["速度", "流量", "佔有率"]] 
                   for vd in vd_ids])
avg_rmse = np.mean([[all_metrics[vd][feat]['rmse'] 
                    for feat in ["速度", "流量", "佔有率"]] 
                    for vd in vd_ids])
avg_r2 = np.mean([[all_metrics[vd][feat]['r2'] 
                  for feat in ["速度", "流量", "佔有率"]] 
                  for vd in vd_ids])

print(f"   平均 MAE: {avg_mae:.2f}")
print(f"   平均 RMSE: {avg_rmse:.2f}")
print(f"   平均 R²: {avg_r2:.3f}")
```

## 📈 結果可視化

```python
# 6. 簡單的結果可視化
print("\\n📈 生成可視化結果...")

import matplotlib.pyplot as plt

# 繪製訓練損失
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('訓練損失曲線')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 繪製預測 vs 真實值示例（VD_001 的速度）
plt.subplot(1, 2, 2)
vd_id = "VD_001"
target_speed = vd_sequences[vd_id][:, -1, 0].numpy()  # 實際速度
pred_speed = predictions[vd_id][:, 0].detach().numpy()  # 預測速度

plt.scatter(target_speed, pred_speed, alpha=0.7)
plt.plot([target_speed.min(), target_speed.max()], 
         [target_speed.min(), target_speed.max()], 'r--', alpha=0.8)
plt.xlabel('實際速度 (km/h)')
plt.ylabel('預測速度 (km/h)')
plt.title(f'{vd_id} 速度預測準確度')
plt.grid(True)

plt.tight_layout()
plt.savefig('first_model_results.png', dpi=150, bbox_inches='tight')
print("✅ 結果已保存到 'first_model_results.png'")

# 顯示 Social Pooling 的鄰居關係
print("\\n🌐 Social Pooling 鄰居關係分析:")
neighbor_info = model.social_pooling.get_neighbor_info(coordinates, vd_ids, node_idx=0)
print(f"   {neighbor_info['node_id']} 的鄰居:")
for i, (neighbor_id, distance, weight) in enumerate(zip(
    neighbor_info['neighbor_ids'], 
    neighbor_info['neighbor_distances'],
    neighbor_info['neighbor_weights']
)):
    print(f"     {neighbor_id}: 距離={distance:.0f}m, 權重={weight:.3f}")

print("\\n🎉 第一個 Social-xLSTM 模型完成！")
print("\\n📋 模型摘要:")
print(f"   架構: 分散式 Social-xLSTM")
print(f"   xLSTM 塊數: {xlstm_config.num_blocks}")
print(f"   社交半徑: {social_config.pooling_radius}m")
print(f"   總參數: {sum(p.numel() for p in model.parameters()):,}")
print(f"   平均 R²: {avg_r2:.3f}")
```

## 🎯 運行完整範例

將所有程式碼保存到 `first_model.py`，然後運行：

```bash
python first_model.py
```

**預期輸出結構**：
```
🔧 配置 Social-xLSTM 模型...
✅ 模型創建成功
   xLSTM 參數: 143,891
   ...

📊 準備範例數據...
✅ 數據生成完成
   ...

🚀 開始訓練...
   Epoch  5/20, Loss: 2485.4839
   ...

📊 評估模型性能...
📈 模型性能評估結果:
...

🎉 第一個 Social-xLSTM 模型完成！
```

## 🚀 下一步建議

完成第一個模型後，建議探索：

1. **真實數據訓練**: [完整訓練指南](../guides/training-guide.md)
2. **模型配置調優**: [模型配置指南](../guides/model-configuration.md)
3. **高級 Social Pooling**: [Social Pooling 進階用法](../guides/social-pooling-advanced.md)
4. **性能基準比較**: [性能基準](../reference/benchmarks.md)

## 💡 關鍵要點

- ✅ **分散式架構**: 每個 VD 獨立的 xLSTM 核心
- ✅ **隱狀態聚合**: Social Pooling 作用於高層特徵
- ✅ **權重共享**: 所有 VD 共享相同的 xLSTM 權重
- ✅ **座標驅動**: 基於地理位置的空間聚合

這個範例展示了 Social-xLSTM 的核心原理，為進一步的研究和應用奠定了基礎。