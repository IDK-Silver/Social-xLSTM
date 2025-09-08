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
- 共享 encoder + centralized batch（建議首選）
  - 以單一 xLSTM 處理整個 [B,T,N,F]，放大 kernel、減少 Python 迴圈：
    - 設 `model.xlstm.multi_vd_mode: true` 並提供 `num_vds`；
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
- [ ] `precision` 改為 `16-mixed`，並測試 `batch_size`/`accumulate_grad_batches` 的上限。
- [ ] `num_workers` 提升、加 `prefetch_factor` 支援（DataModule 讀取 YAML 並傳入 DataLoader）。
- [ ] 規劃 centralized 流程 PoC：multi‑VD 模式（共享 encoder），移除 per‑VD 前向迴圈，評估速度與指標變化。
- [ ] 若保留 distributed 格式：嘗試「小組批次」策略或共享 encoder + per‑VD head 的替代方案。
- [ ] 導入 `positions` 使社交池化生效（有資料時）。
- [ ] 以 Lightning Profiler 固化量測流程（訓練前 200 steps 例行檢查）。

