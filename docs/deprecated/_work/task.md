# GPU 利用率優化（Social‑xLSTM 訓練）

本頁記錄目前 GPU 利用率偏低（<40%）的主因、可立即採取的改善、與中長期重構方向；並附上驗證方式與後續 TODO。

## 主要瓶頸（Root Causes）
- VDs 逐一前向（小 kernel 串行）
  - `src/social_xlstm/models/vd_xlstm_manager.py` 逐個 VD 呼叫 xLSTM，造成大量 Python 迴圈與小型 CUDA kernel 啟動，GPU 難以吃滿。
- Collate 在 CPU 端做 per‑VD 切片
  - DataLoader 先堆 [B,T,N,F]，`src/social_xlstm/dataset/core/collators.py` 再切為 dict-of-VD，增加 CPU 索引/複製負擔，延後 GPU 飛輪。
- 模型與批量偏小 + 全精度
  - 預設 `precision: "32-true"`（`cfgs/training/standard.yaml`），PEMS-BAY 樣例僅單一特徵（`avg_speed`），計算量小、kernel 不大。
- 社交池化未實際計算
  - `DistributedSocialXLSTMModel.forward` 若 `positions=None` 直接回傳 0 向量（`src/social_xlstm/models/distributed_social_xlstm.py`），等於跳過社交池化 → 更小計算量。

## 立即改善（Quick Wins）
- Precision/AMP
  - `cfgs/training/standard.yaml`：將 `precision` 改為 `"16-mixed"`（或支援 bf16 時 `"bf16-mixed"`）。
  - 可加入 `accumulate_grad_batches: 2`（或 4）提升有效批量。
- DataLoader 吞吐
  - `cfgs/data/standard.yaml`：將 `data.loader.num_workers` 調至 8–16（依 CPU），維持 `pin_memory: true`。
  - 在 `TrafficDataModule._make_dataloader` 支援 `prefetch_factor`（建議 4–8），由 `data.loader.prefetch_factor` 傳入。
- 批量/日誌
  - 視顯存提高 `data.loader.batch_size`（例如 64/128/256）；
  - `trainer.log_every_n_steps` 可拉大（如 50），減少控制面同步負擔。
- PyTorch 2.x 最佳化（可選）
  - 啟動前設定 `torch.set_float32_matmul_precision("high")`；
  - 嘗試 `torch.compile(model)`（注意與 xLSTM/第三方庫相容性）。

## 中長期重構（高影響）
– 共享 encoder + centralized batch（建議首選）
  - 以單一 xLSTM 處理整個 [B,T,N,F]，放大 kernel、減少 Python 迴圈：
    - 不使用 `multi_vd_mode` 的 N×F 攤平；改以 `[B,N,T,F] → [B×N,T,F]` 前向，再還原 `[B,N,T,E]`。
    - 使用 centralized 批次：取消 `scripts/train/with_social_pooling/train_multi_vd.py` 中 `dataset_config.batch_format = 'distributed'`；
    - 模型端以一次前向產出所有 VD 表徵，必要時再以輕量 head 分 VD 輸出。
- 若必須保留 per‑VD 獨立權重
  - 重構為「共享 encoder + per‑VD 輕量 head」或將多個 VD 在 batch 維度合併成「小組批次」做前向，盡量減少 Python 單體迴圈。
- 讓社交池化真正生效（可選）
  - 由 HDF5 讀取 VD 座標（經緯度轉近似平面座標），由 DataModule 提供 `positions`，使 `XLSTMSocialPoolingLayer` 參與計算（提升模型表達力，也增加 GPU 工作量）。

## 建議配置修改（範例片段）
- `cfgs/training/standard.yaml`
  ```yaml
  trainer:
    precision: "16-mixed"
    # 可選：
    accumulate_grad_batches: 2
    log_every_n_steps: 50
  ```
- `cfgs/data/standard.yaml`
  ```yaml
  data:
    loader:
      num_workers: 12           # 依 CPU 調整 8–16
      pin_memory: true
      # 新增：
      prefetch_factor: 4
      # 視顯存調整：
      batch_size: 128
  ```
- `scripts/train/with_social_pooling/train_multi_vd.py`
  - 若採 centralized：移除或改為 `dataset_config.batch_format = 'centralized'`。

## 驗證與量測（Diagnose/Measure）
- Lightning Profiler：在 `trainer` 啟用 `profiler: "advanced"`，跑 100–200 steps，檢視 DataLoader vs training_step 時間占比。
- 監控
  - `nvidia-smi dmon -s pucm` 或 `watch -n 0.5 nvidia-smi` 觀察 GPU 利用率/顯存；
  - `htop` 查看 DataLoader worker 是否成為瓶頸（CPU 滿載且 GPU 閒置）。

## TODO（落地清單）
- [x] `precision` 改為 `16-mixed`（standard/dev 皆已更新），並在訓練腳本打印 Trainer precision；後續測試 `batch_size`/`accumulate_grad_batches` 的上限。
- [ ] `num_workers` 提升、加 `prefetch_factor` 支援（DataModule 讀取 YAML 並傳入 DataLoader）。
- [x] 規劃並落地 centralized 流程 PoC（Shared Encoder），新增 `train_shared.py`，評估速度與指標變化。
- [ ] 若保留 distributed 格式：嘗試「小組批次」策略或共享 encoder + per‑VD head 的替代方案。
- [x] 導入 `positions` 並在嚴格模式下強制使用（缺失直接報錯）。
- [ ] 以 Lightning Profiler 固化量測流程（訓練前 200 steps 例行檢查）。
- [x] 統一 HDF5 規格：轉檔寫入 `metadata/vd_info/<vdid>{position_lat,position_lon}`（PEMS‑BAY、METR‑LA 皆已更新）。
- [ ] 重建 PEMS‑BAY / METR‑LA H5（或撰寫 backfill 腳本）以補齊既有檔的 `vd_info`。

---

# P0：Shared Encoder 重構與可訓練 Social Pooling 設計（最優先）

目標與範圍（可以大改、兼容性非必須）
- 共享一份 xLSTM 權重，批次化處理所有 VD，移除逐 VD 前向瓶頸，顯著提升 GPU 利用率與記憶體效率。
- 保留 per‑VD 表徵（不混合 N×F 到同一向量），在 encoder 輸出上做 per‑VD 的社交池化。
- 將社交池化升級為可訓練（trainable）機制，至少支援：
  - Learnable 半徑/軟遮罩（soft radius mask）
  - 注意力權重（dot‑product 或 MLP attention），可結合相對位置嵌入（Δx, Δy）
- 改為 centralized 批次（[B,T,N,F]）的資料流水，簡化 collate 與 CPU 端開銷。

關鍵設計（Architecture Delta）
- Encoder：
  - 現況：`VDXLSTMManager` 逐 VD 跑 `TrafficXLSTM`，dict‑of‑VD → 多個小 kernel。
  - 目標：單一 `TrafficXLSTM`，將 `[B,T,N,F]` 重排為 `[B×N,T,F]` 前向，再 reshape 回 `[B,N,T,E]`；取最後步 `[B,N,E]`。
  - 注意：不使用現有 `multi_vd_mode` 的 N*F 攤平（會破壞 per‑VD 分離）。
- Social Pooling（可訓練）：
  - 新增 `SocialPoolingAttentionLayer`，支援：
    - 權重計算：
      - dot‑product：`α_ij = softmax(q_i · k_j)`，其中 `q_i=W_q h_i`、`k_j=W_k h_j`
      - MLP attention：`α_ij = softmax(MLP([h_i, h_j, φ(Δx,Δy)]))`
    - 位置嵌入：`φ(Δx,Δy) = MLP_pos([Δx,Δy])` 或 RFF/pe（簡化先用 MLP）
    - 半徑/遮罩：
      - Learnable 全域半徑 `r`（nn.Parameter），或 learnable 溫度 `β` 用於 `σ(β·(r-d_ij))` 形成軟遮罩；與 attention 權重相乘。
    - 輸出：`c_i = Σ_j α_ij · v_j`（`v_j=W_v h_j`）；`c_i` 與 `h_i` 融合後送 head。
  - 配置：`cfgs/social_pooling/attention.yaml`（新）
    - `enabled: true`
    - `type: attention | mean | max | weighted_mean`
    - `use_positional: true`
    - `learnable_radius: true`
    - `mask_mode: soft`（soft/sigmoid，或 hard/距離門檻）
- Data pipeline：
  - DataModule 改預設 centralized；保留 distributed 僅為過渡（可標記 deprecated）。
  - batch 結構：`features: Tensor[B,T,N,F]`、`positions: Tensor[N,2]` 或 `Tensor[B,N,2]`（靜態座標可用 [N,2] 節省複製）；
  - collate：去除 per‑VD dict 切片。

核心變更點（API/檔案）
- 新增：`src/social_xlstm/pooling/attention.py`（`SocialPoolingAttentionLayer`）
- 新增：`src/social_xlstm/models/shared_social_xlstm.py`（Shared Encoder 版本 LightningModule）
- 更新：`src/social_xlstm/dataset/core/datamodule.py` 支援 centralized 預設與 `positions` 為 `[N,2]`
- 更新：`scripts/train/with_social_pooling/train_multi_vd.py` → 指向 shared 模型；或新增 `train_shared.py`
- 更新：配置
  - `cfgs/models/xlstm.yaml` 新增 `model.shared_encoder: true`
  - 新增 `cfgs/social_pooling/attention.yaml`

驗收標準（Acceptance）
- 功能：
  - 前向支援 `[B,T,N,F]`，輸出與現有任務對齊（每 VD 預測 `prediction_length×features`）。
  - 社交池化在有座標時必定啟用，權重對梯度非零（可訓練）。
  - 零鄰居/缺座標時能安全退化（回退 mean/零向量或僅個體向量）。
- 效能：
  - 單卡 GPU 利用率 ≥ 80%（同機測試基準）。
  - 記憶體：參數/優化器狀態顯著下降（對應 N→1）。
- 準確度：
  - 與現況基線相當或更好（以 MAE/RMSE/R2 監控）。

開發步驟與排程（建議）
- Phase 1：資料/模型骨架（1–2 天）
  - [x] DataModule 支援 centralized 與 distributed，產出 `positions_xy: [N,2]` 或 `positions[vd_id]: [B,T,2]`
  - [x] `SharedSocialXLSTMModel`：完成 `[B×N,T,F] → [B,N,T,E]` 流程與融合/頭部
  - [x] 接上現有 metrics/optimizer/Lightning plumbing；新增 `scripts/train/with_social_pooling/train_shared.py`
- Phase 2：可訓練社交池化（1–2 天）
  - [ ] `SocialPoolingAttentionLayer`（dot‑product/MLP、相對座標、learnable 半徑軟遮罩）
  - [ ] 與 shared 模型整合、配置檔新增
  - [ ] 單元測試：shape、mask、權重歸一、梯度存在；零鄰居退化行為
- Phase 3：效能與文件（1 天）
  - [ ] AMP、batch/accumulate 調參，Lightning Profiler 報告
  - [x] 文檔：新增 `docs/guides/training-with-shared-encoder.md`、更新 `training-with-sp.md` 嚴格模式；README 重寫

風險與因應
- OOM：`B×N` 過大 → 降低 `B` 或引入「VD 分塊」（2–4 塊）做多次前向累積；啟用 gradient checkpointing。
- 品質退化：新增 per‑VD 輕量 head 或條件化（`nn.Embedding(num_vds, d)`）微調。
- 座標缺失：以距離遮罩=0 的退化策略 + 僅個體表示；或以鄰近圖補充。

配置草案（示例）
```yaml
model:
  name: "SharedSocialXLSTM"
  xlstm:
    input_size: 3
    embedding_dim: 256
    num_blocks: 6
    output_size: 3
    sequence_length: 12
    prediction_length: 3
    slstm_at: [1,3]
    slstm_backend: "vanilla"
    mlstm_backend: "vanilla"
    context_length: 256
    dropout: 0.1
  shared_encoder: true

social_pooling:
  enabled: true
  type: "attention"           # attention | mean | max | weighted_mean
  learnable_radius: true
  mask_mode: "soft"           # soft(sigmoid) or hard(threshold)
  use_positional: true         # 使用 Δx,Δy 嵌入

trainer:
  precision: "16-mixed"
  accumulate_grad_batches: 2
```

里程碑（Milestones）
- M1（shared encoder 可跑）→ M2（attention pooling 可訓練）→ M3（效能達標/文件完成）

—

目前進度（摘要）
- 已完成：
  - Shared Encoder 模型與訓練腳本（centralized 批次）。
  - DataModule/Collate：centralized 與 distributed 皆可產生 positions；centralized 直接輸出 `positions_xy: [N,2]`。
  - 嚴格模式：開啟 social pooling 時，缺座標/NaN/shape 錯誤即報錯（shared/distributed 皆適用）。
  - 轉檔統一：PEMS‑BAY、METR‑LA 轉檔腳本已寫入 `metadata/vd_info/<vdid>`（位置：
    `scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py`、
    `scripts/dataset/pre_process/metr_la/convert_metr_la_to_hdf5.py`）。
  - 文檔/README 更新：新增 shared encoder 指南、明確資料規格與嚴格模式。

- 待辦（高優先）：
  - 重新產生 PEMS‑BAY/METR‑LA H5 或提供 backfill 腳本，確保 `vd_info` 齊全。
  - Phase 2 可訓練社交池化（attention + learnable radius/soft mask）。
  - 效能量測與報告（GPU 利用率 ≥ 80% 目標）。

執行指令（驗證 Shared Encoder）
```bash
python scripts/train/with_social_pooling/train_shared.py \
  --config cfgs/profiles/pems_bay/standard.yaml \
  --output_dir blob/experiments/shared_pems
```
注意：若使用舊 H5 未含 `metadata/vd_info`，會在第一個 batch 報錯。請先用更新後的轉檔腳本重建 H5（或等 backfill 腳本）。

資料集重建指令（統一規格）
```bash
# PEMS-BAY 重新轉檔並驗證
python scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py \
  --data-csv <PEMS-BAY.csv> \
  --meta-csv <PEMS-BAY-META.csv> \
  --output-h5 blob/dataset/processed/pems_bay.h5 \
  --validate

# METR-LA 重新轉檔並驗證
python scripts/dataset/pre_process/metr_la/convert_metr_la_to_hdf5.py \
  --data-csv <metr-la.csv> \
  --meta-csv <metr-la_sensor_locations.csv> \
  --output-h5 blob/dataset/processed/metr_la.h5 \
  --validate
```
