# DataModule 比較分析：TrafficDataModule vs DistributedTrafficDataModule

> 關鍵差異記錄文檔 - 避免混淆功能重複性  
> 建立日期：2025-08-02

## 🎯 核心問題

**問題**: DistributedTrafficDataModule 跟 TrafficDataModule 功能看起來一樣？  
**答案**: 表面相似但架構目標完全不同

---

## 📊 詳細比較

### 相同點 (表面功能)
- 都是 PyTorch Lightning DataModule
- 都處理交通資料載入
- 都提供 train/val/test DataLoader
- 都支援批次處理

### 關鍵差異 (架構本質)

| 維度 | TrafficDataModule (現有) | DistributedTrafficDataModule (計劃) |
|------|-------------------------|----------------------------------|
| **輸出格式** | 標準張量 `[B, T, N, F]`* | VD 字典 `{vd_id: tensor}` |
| **架構目標** | 集中式處理 | 分散式處理 |
| **VD 處理** | 所有 VD 合併在一個張量 | 每個 VD 獨立張量 |
| **下游模型** | 標準 LSTM/xLSTM | VDXLSTMManager |
| **記憶體策略** | 全部載入 | 選擇性載入 |

> *張量維度定義: **B**=Batch (批次), **T**=Time (時間步), **N**=Nodes (VD數量), **F**=Features (特徵維度)

---

## 💻 程式碼差異範例

### TrafficDataModule 輸出
```python
batch = {
    'input_seq': Tensor[B, T, N, F],     # B=批次, T=時間步, N=VD數量, F=特徵維度
    'target_seq': Tensor[B, T, N, F],    # 所有 VD 合併在標準張量中
    'input_mask': Tensor[B, T, N],       # 對應的遮罩張量
    'vdids': List[str]                   # 僅作為 metadata
}

# 下游使用
model.forward(batch['input_seq'])  # 直接處理整個張量
```

### DistributedTrafficDataModule 輸出  
```python
batch = {
    'features': {                        # VD 字典結構
        'VD-001': Tensor[B, T, F],       # B=批次, T=時間步, F=特徵維度 (每個VD獨立)
        'VD-002': Tensor[B, T, F],       # N 維度被分解到字典 keys 中
        'VD-003': Tensor[B, T, F]
    },
    'targets': {
        'VD-001': Tensor[B, T, F],       # 目標張量同樣按 VD 分組
        'VD-002': Tensor[B, T, F], 
        'VD-003': Tensor[B, T, F]
    }
}

# 下游使用
hidden_states = vd_manager.forward(batch['features'])  # 字典處理 {vd_id: Tensor[B,T,H]}
social_context = social_pooling(hidden_states)         # per-VD 社交聚合
```

---

## 🏗️ 架構差異說明

### 集中式架構 (現有)
```
所有VD資料 → [大張量] → 單一模型 → 預測
```

### 分散式架構 (計劃)
```
VD-A → 獨立張量 → xLSTM-A → 隱狀態-A ┐
VD-B → 獨立張量 → xLSTM-B → 隱狀態-B ├→ Social Pooling → 融合預測
VD-C → 獨立張量 → xLSTM-C → 隱狀態-C ┘
```

---

## ⚡ 為什麼需要兩種不同的 DataModule？

### 1. **Social Pooling 需求**
- 分散式架構需要 per-VD 隱狀態來計算社交特徵
- 集中式無法提供獨立的隱狀態

### 2. **xLSTM 並行處理**
- 每個 VD 需要獨立的 xLSTM 實例
- VDXLSTMManager 需要字典格式輸入

### 3. **記憶體效率**
- 可以選擇性載入特定 VD 的資料
- 支援動態 VD 組合

### 4. **架構遷移**
- 從錯誤的集中式架構遷移到正確的分散式架構
- 兩者不能互換使用

---

## 🎯 類比說明

**就像餐廳服務模式**：
- **TrafficDataModule** = 普通餐廳 (廚師統一準備，一起上菜)
- **DistributedTrafficDataModule** = 自助餐廳 (每個區域獨立，客人選擇組合)

都提供食物，但服務模式完全不同。

---

## 📝 實施狀態

| 模組 | 狀態 | 檔案位置 |
|------|------|----------|
| TrafficDataModule | ✅ 已實現 | `src/social_xlstm/dataset/core/datamodule.py` |
| DistributedTrafficDataModule | ⏳ 計劃中 | `src/social_xlstm/data/distributed_datamodule.py` (todo.md Task 1.2) |

---

## ⚠️ 重要提醒

**不是功能重複！**  
這是為了支援不同架構模式的必要基礎設施差異。就像 TCP 和 UDP 都是網路協定，但使用場景完全不同。

---

**參考資料**：
- `todo.md` Task 1.2: 建立 DistributedTrafficDataModule
- 分散式 Social-xLSTM 架構設計 (ADR-0100)
- VDXLSTMManager 設計規範
- [`docs/architecture/data_pipeline.md`](../architecture/data_pipeline.md) ― 數據流程架構與切片策略詳解