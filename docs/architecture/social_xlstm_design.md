# Social xLSTM 架構設計

## 📚 前置閱讀
建議先閱讀 [Social LSTM 正確理解](social_lstm_correct_understanding.md) 了解核心概念。

## 🎯 核心概念

Social xLSTM 的創新在於**將 Social Pooling 機制與 xLSTM 結合**，用於多VD交通預測。

**關鍵原則**：每個VD有自己的xLSTM模型，通過Social Pooling在隱藏狀態層面進行交互。

## 🏗️ 正確的架構設計

### 個別VD模型 + Social Pooling
```
VD1: xLSTM₁ → h₁ → Social Pool → h₁' → prediction₁
VD2: xLSTM₂ → h₂ → Social Pool → h₂' → prediction₂
VD3: xLSTM₃ → h₃ → Social Pool → h₃' → prediction₃
```

### 與錯誤設計的對比

#### ❌ 錯誤設計（聚合方式）
```
多個VD → 聚合特徵 → 單一xLSTM → 單一預測
```

#### ✅ 正確設計（Social LSTM範式）
```
每個VD → 個別xLSTM → 隱藏狀態交互 → 各自預測
```

## 🔧 技術實現架構

### 正確的 Social xLSTM 模型

```python
class SocialTrafficXLSTM(nn.Module):
    def __init__(self, config, vd_ids):
        super().__init__()
        self.vd_ids = vd_ids
        
        # 每個VD有自己的xLSTM模型（權重共享）
        self.shared_xlstm_config = config
        self.xlstm_models = nn.ModuleDict()
        
        for vd_id in vd_ids:
            self.xlstm_models[vd_id] = xLSTMBlockStack(config)
        
        # Social Pooling Layer
        self.social_pooling = SocialPoolingLayer(
            grid_size=(8, 8),
            spatial_radius=25000,  # 25km
            hidden_dim=config.hidden_dim
        )
        
        # 每個VD的輸出投影
        self.output_projections = nn.ModuleDict()
        for vd_id in vd_ids:
            self.output_projections[vd_id] = nn.Linear(config.hidden_dim, 3)
    
    def forward(self, vd_data, vd_coords):
        """
        Args:
            vd_data: {vd_id: tensor([batch, seq_len, 3])}
            vd_coords: {vd_id: (x, y)}
        """
        # 第1階段：每個VD個別處理
        vd_hidden_states = {}
        for vd_id in self.vd_ids:
            hidden_state = self.xlstm_models[vd_id](vd_data[vd_id])
            vd_hidden_states[vd_id] = hidden_state
        
        # 第2階段：Social Pooling更新隱藏狀態
        updated_hidden_states = {}
        for target_vd in self.vd_ids:
            # 找到鄰居VD
            neighbors = self.find_neighbors(target_vd, vd_coords)
            
            # Social Pooling
            social_context = self.social_pooling(
                target_hidden=vd_hidden_states[target_vd],
                neighbor_hiddens={nid: vd_hidden_states[nid] for nid in neighbors},
                target_coords=vd_coords[target_vd],
                neighbor_coords={nid: vd_coords[nid] for nid in neighbors}
            )
            
            # 結合原始隱藏狀態和社交上下文
            updated_hidden = self.combine_hidden_states(
                vd_hidden_states[target_vd], social_context
            )
            updated_hidden_states[target_vd] = updated_hidden
        
        # 第3階段：每個VD各自預測
        predictions = {}
        for vd_id in self.vd_ids:
            pred = self.output_projections[vd_id](updated_hidden_states[vd_id])
            predictions[vd_id] = pred
        
        return predictions
    
    def find_neighbors(self, target_vd, vd_coords, radius=25000):
        """找到目標VD的鄰居"""
        target_x, target_y = vd_coords[target_vd]
        neighbors = []
        
        for vd_id, (x, y) in vd_coords.items():
            if vd_id != target_vd:
                distance = ((x - target_x)**2 + (y - target_y)**2)**0.5
                if distance <= radius:
                    neighbors.append(vd_id)
        
        return neighbors
    
    def combine_hidden_states(self, original_hidden, social_context):
        """結合原始隱藏狀態和社交上下文"""
        # 簡單的加權組合
        return original_hidden + 0.1 * social_context
```

### Social Pooling Layer實現

```python
class SocialPoolingLayer(nn.Module):
    def __init__(self, grid_size=(8, 8), spatial_radius=25000, hidden_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.spatial_radius = spatial_radius
        self.hidden_dim = hidden_dim
        
        # 用於處理池化後特徵的全連接層
        self.feature_projection = nn.Linear(
            grid_size[0] * grid_size[1] * hidden_dim, 
            hidden_dim
        )
    
    def forward(self, target_hidden, neighbor_hiddens, target_coords, neighbor_coords):
        """
        Args:
            target_hidden: [batch, seq_len, hidden_dim]
            neighbor_hiddens: {neighbor_id: [batch, seq_len, hidden_dim]}
            target_coords: (x, y)
            neighbor_coords: {neighbor_id: (x, y)}
        """
        batch_size, seq_len, hidden_dim = target_hidden.shape
        M, N = self.grid_size
        
        # 初始化網格
        grid_tensor = torch.zeros(batch_size, seq_len, M, N, hidden_dim)
        
        target_x, target_y = target_coords
        
        # 將鄰居的隱藏狀態分配到網格
        for neighbor_id, neighbor_hidden in neighbor_hiddens.items():
            neighbor_x, neighbor_y = neighbor_coords[neighbor_id]
            
            # 計算相對位置
            rel_x = neighbor_x - target_x
            rel_y = neighbor_y - target_y
            
            # 分配到網格
            grid_x = int((rel_x + self.spatial_radius) / (2 * self.spatial_radius / M))
            grid_y = int((rel_y + self.spatial_radius) / (2 * self.spatial_radius / N))
            
            # 邊界檢查
            grid_x = max(0, min(M-1, grid_x))
            grid_y = max(0, min(N-1, grid_y))
            
            # 累加到網格
            grid_tensor[:, :, grid_x, grid_y, :] += neighbor_hidden
        
        # 展平並投影
        flattened = grid_tensor.reshape(batch_size, seq_len, -1)
        social_features = self.feature_projection(flattened)
        
        return social_features
```

## 🎯 實現優先級

### Phase 1: 個別VD模型
```python
# 為每個VD創建獨立的xLSTM模型
for vd_id in vd_ids:
    xlstm_models[vd_id] = xLSTMBlockStack(config)
```

### Phase 2: Social Pooling實現
```python
# 實現隱藏狀態的空間交互
social_context = social_pooling(hidden_states, coordinates)
```

### Phase 3: 整合與優化
```python
# 端到端訓練和評估
predictions = social_xlstm(vd_data, vd_coords)
```

## 💡 關鍵創新點

1. **個別化建模**: 每個VD保持自己的"個性"
2. **隱藏狀態交互**: Social Pooling作用於隱藏狀態，不是原始特徵
3. **分散式預測**: 每個VD預測自己的未來狀態
4. **空間感知**: 通過座標驅動的網格分配學習空間關係

## 📊 與現有系統的關係

### 層次結構
```
基礎層: 單VD xLSTM 模型
         ↓
空間層: Social Pooling (隱藏狀態交互)
         ↓  
完整層: Social xLSTM (每VD獨立預測)
```

### 數據需求
- **輸入**: 每個VD的時間序列數據 + 座標信息
- **處理**: 個別VD模型 + 隱藏狀態交互
- **輸出**: 每個VD的獨立預測

## 🔄 訓練策略

### 損失函數設計
```python
def compute_loss(predictions, targets, vd_ids):
    total_loss = 0
    for vd_id in vd_ids:
        vd_loss = F.mse_loss(predictions[vd_id], targets[vd_id])
        total_loss += vd_loss
    return total_loss / len(vd_ids)
```

### 批次處理
```python
# 每個批次包含所有VD的數據
batch_vd_data = {vd_id: batch_data[vd_id] for vd_id in vd_ids}
predictions = model(batch_vd_data, vd_coords)
```

## 📝 與原始Social LSTM的對應

| Social LSTM (行人) | Social xLSTM (交通) |
|-------------------|-------------------|
| 每個行人獨立LSTM | 每個VD獨立xLSTM |
| 隱藏狀態池化 | 隱藏狀態池化 |
| 各自軌跡預測 | 各自流量預測 |
| 空間避讓行為 | 空間交通關聯 |

## 🚀 實現優勢

1. **可解釋性**: 可以分析每個VD如何受到鄰居影響
2. **靈活性**: 可以為不同VD使用不同的模型配置
3. **擴展性**: 新增VD時不需要重新訓練所有模型
4. **個別化**: 每個VD可以學習自己特有的交通模式

這種設計完全符合原始Social LSTM的核心思想，確保了架構的理論正確性和實際可行性。