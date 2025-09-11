# Social‑xLSTM 架構：Per‑VD 與 Shared Encoder 對照與遷移指引

本文件說明目前實作的「Per‑VD（每個 VD 一個 xLSTM）」架構，以及建議中的「Shared Encoder（單一 xLSTM 共用權重，對所有 VD 併批處理）」方案，重點比較資料流、模型結構、社交池化可行性與效能取捨，並提供漸進式遷移策略。

---

## 現況架構（Per‑VD Manager）

- 批次型態：`distributed` → DataLoader 產出 `Dict[vd_id, Tensor[B,T,F]]`
  - 建立與位置資料：`src/social_xlstm/dataset/core/collators.py:65`、`src/social_xlstm/dataset/core/collators.py:132`
  - 位置來源：HDF5 `metadata/vd_info` 的經緯度 → Mercator XY（若存在）
    - 轉換與注入：`src/social_xlstm/dataset/core/datamodule.py:61`、`src/social_xlstm/dataset/core/datamodule.py:74`

- 編碼器：每個 VD 一個 `TrafficXLSTM` 實例，由 `VDXLSTMManager` 管理
  - 逐 VD 前向：`src/social_xlstm/models/vd_xlstm_manager.py:192`
  - 取完整時間的隱狀態 `[B,T,E]`，下游取最後一步 `[B,E]`

- 社交池化：需要 `positions` 時觸發；否則回傳零向量（no‑op）
  - 觸發條件：`src/social_xlstm/models/distributed_social_xlstm.py:125`
  - 融合與預測：`src/social_xlstm/models/distributed_social_xlstm.py:146`、`src/social_xlstm/models/distributed_social_xlstm.py:152`

- 關鍵特性
  - 優點：每個 VD 有獨立權重，表達力高；實作簡潔直觀
  - 缺點：Python 迴圈 + 小型 CUDA kernel 使 GPU 利用率偏低；模型副本多、顯存與初始化成本高

---

## 共享編碼器提案（Shared Encoder，建議做法）

目標：以「一個 xLSTM 權重」對「所有 VD」做批次化前向，保留 per‑VD 表徵且使社交池化仍可用，同時提升 GPU 吞吐與利用率。

- 核心想法：把 VD 維度併到 batch 維度
  - 形狀轉換：`[B, T, N, F] → [B×N, T, F]`，餵入「單一」`TrafficXLSTM`
  - 得到隱狀態：`[B×N, T, E] → reshape → [B, N, T, E]`
  - 取最後一步：`[B, N, E]`，即可做 per‑VD 的社交池化（依 VD 對應的座標）

- 與 `multi_vd_mode`（特別小心）
  - 目前 `TrafficXLSTM` 的 `multi_vd_mode` 是把 `[B,T,N,F]` 攤平成 `[B,T,N*F]` 再做一條網路，這會「混合所有 VD 的特徵到同一表示」→ 不適合需要 per‑VD 表徵的社交池化。
  - 本提案不依賴 `multi_vd_mode`；改由外部 reshape 成 `[B×N,T,F]`，保持 per‑VD 隱狀態可回復。

- 變更重點
  - 用「單一」`TrafficXLSTM` 取代 `VDXLSTMManager` 多模型實例
  - DataLoader 可改回 `centralized` 批次（`[B,T,N,F]`），或沿用 `distributed` 再在模型層合併
  - 社交池化邏輯不變，仍需 `positions`（XY）

- 優缺點
  - 優點：
    - GPU kernel 大且連續，通常可把 GPU 利用率提升到 70–95%
    - 參數共享，顯存降低、初始化更快
  - 取捨：
    - 失去 per‑VD 獨立權重；若需要可在共享 encoder 後加「極輕量 per‑VD head」（例如小型線性層）

---

## 資料流對照

- 既有（Per‑VD）：
  - Collate：`[B,T,N,F] → {vd_id: [B,T,F]}`（附 positions）
  - 編碼：每個 VD 個別跑 `TrafficXLSTM` → `{vd_id: [B,T,E]}`
  - Pooling：距離半徑聚合（只取 `t=-1`）
  - 融合/預測：`cat([B,E],[B,E]) → [B,2E] → head`

- 共享（Shared Encoder）：
  - Collate（建議 centralized）：`[B,T,N,F]`
  - 編碼：reshape `[B,N,T,F] → [B×N,T,F] → xLSTM → [B×N,T,E] → [B,N,T,E]`
  - Pooling：用 `vd_id` 或 index 對照座標，對 `[B,N,E]` 做鄰近聚合
  - 融合/預測：與現有相同（只是資料結構從 dict 變成張量 + 索引映射）

---

## 社交池化與座標

- 啟用條件：`positions is not None` 時才會計算，否則回退為零向量（no‑op）
  - 判斷：`src/social_xlstm/models/distributed_social_xlstm.py:125`
- 來源與建置：
  - HDF5 `metadata/vd_info/<vdid>` 的 `position_lat/position_lon` 若存在，DataModule 會自動讀取、用 Mercator 投影成 XY 並注入 batch
    - 轉換與注入：`src/social_xlstm/dataset/core/datamodule.py:61`、`src/social_xlstm/dataset/core/datamodule.py:74`
  - Collate 會回傳 `positions: Dict[vd_id, Tensor[B,T,2]]`
    - 建立：`src/social_xlstm/dataset/core/collators.py:114`

---

## 遷移策略（建議漸進）

1) Quick Wins（維持 Per‑VD 架構）
- 已接通 `positions`，只要 HDF5 有經緯度就會啟用社交池化
- 先以混合精度、增大 batch、提高 DataLoader 吞吐改善 GPU 利用率

2) Shared Encoder PoC
- 在模型層將 `{vd_id: [B,T,F]}` 轉為 `[B,N,T,F] → [B×N,T,F]`，用單一 `TrafficXLSTM` 前向，還原 `[B,N,T,E]`
- 用 VD index ↔ vd_id 映射產生對應 `positions`，做社交池化與後續預測
- 僅替換 `VDXLSTMManager` 路徑，其他模組最小變動

3) 進一步優化（視需要）
- 在共享 encoder 之後增加 per‑VD 輕量 head（例如 `nn.Embedding(num_vds, small_dim)` 做條件化，或 per‑VD 線性層）
- 若資料充分、表達力不足，再考慮增大 `embedding_dim` 或 block 數

---

## 小結與建議

- 若目標是「先讓社交池化真正起作用」：目前的 Per‑VD 架構已可做到（DataModule/Collate 已注入 `positions`）。
- 若目標是「提升 GPU 利用率」：建議實作 Shared Encoder（N 併入 batch 的方案），能大幅減少 Python 迴圈、放大 kernel。
- `multi_vd_mode` 的攤平成 `N*F` 雖然簡單，但會破壞 per‑VD 分離，與社交池化需求不相容，不建議用於本專案的社交池化路徑。

---

## 相關程式碼參考（點擊可開啟）
- `src/social_xlstm/models/vd_xlstm_manager.py:192`（逐 VD 前向）
- `src/social_xlstm/models/distributed_social_xlstm.py:125`（positions 判斷啟用 pooling）
- `src/social_xlstm/models/distributed_social_xlstm.py:146`（個別 VD 融合與預測）
- `src/social_xlstm/dataset/core/datamodule.py:61`（準備 distributed collate）
- `src/social_xlstm/dataset/core/datamodule.py:74`（經緯度→XY 注入 collate）
- `src/social_xlstm/dataset/core/collators.py:114`（positions 產出）
- `src/social_xlstm/models/xlstm.py:318`（`get_hidden_states`、multi_vd_mode 攤平判斷）

---

## 附：Shared Encoder 迷你範例（概念示意）

```python
# x: Dict[vd_id, Tensor[B,T,F]] → stack → [B,N,T,F]
vd_ids = list(x.keys())
X = torch.stack([x[v] for v in vd_ids], dim=1)  # [B,N,T,F]
B, N, T, F = X.shape

# 合批成 [B*N,T,F]，共用一個 TrafficXLSTM
X_flat = X.reshape(B*N, T, F)
H_flat = shared_xlstm.get_hidden_states(X_flat)  # [B*N, T, E]
H = H_flat.reshape(B, N, T, -1)                  # [B,N,T,E]

# 取最後一步 [B,N,E]，再配合 positions 做社交池化
H_last = H[:, :, -1, :]  # [B,N,E]
# pooling(...) 依據 vd_ids × positions 取得 [B,N,E] 的社交上下文
```

此路徑保留 per‑VD 表徵、能直接套用現有的社交池化機制，同時顯著提升 GPU 利用率。
