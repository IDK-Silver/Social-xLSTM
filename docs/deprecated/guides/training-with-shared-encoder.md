# Training with Shared Encoder + Social Pooling

本指南說明如何使用 Shared‑Encoder Social‑xLSTM（單一 xLSTM 權重處理所有 VD）進行訓練，並在隱狀態上套用空間社交池化（social pooling）。

## 概述
- Shared Encoder：將 `[B,T,N,F]` 攤平成 `[B×N,T,F]`，用一個 `TrafficXLSTM` 進行前向，再還原為 `[B,N,T,E]`。
- 保留 per‑VD 表徵：還原後取 `t=-1` 得到 `[B,N,E]`，依 VD 的座標做鄰近聚合與融合預測。
- 優點：GPU 利用率高（大 kernel、少迴圈）、參數/優化器狀態只有 1 份、記憶體更省。

對應代碼
- 模型：`src/social_xlstm/models/shared_social_xlstm.py`
- 訓練腳本：`scripts/train/with_social_pooling/train_shared.py`
- 中央批次 collate：`src/social_xlstm/dataset/core/collators.py: CentralizedCollator`
- DataModule 產生 `positions_xy`：`src/social_xlstm/dataset/core/datamodule.py: _prepare_centralized_collate`

## 嚴格模式（Strict Mode）
當 `social_pooling.enabled: true` 時：
- 需要在 batch 中提供 `positions_xy: Tensor[N,2]`（所有 VD 的 XY 座標，單位公尺）。
- 若缺失、含 NaN、或維度不符，會直接 `RuntimeError` 中止訓練。
- 來源：HDF5 `metadata/vd_info/<vdid>` 的 `position_lat`/`position_lon`，DataModule 會自動轉為 Mercator XY。

錯誤訊息（摘錄）
- Shared Encoder：`Social pooling is enabled but 'positions_xy' is missing...`
- 分散式：`Social pooling is enabled but 'positions' is missing...`、`Missing positions for VD 'xxxx' ...`

如何檢查/修復
- 檢查 HDF5 是否含有 `metadata/vd_info/<vdid>/{position_lat, position_lon}` 屬性。
- 若缺失，請在資料轉換階段補齊座標後重建 HDF5（見 `src/social_xlstm/dataset/storage/h5_converter.py`）。

## 使用步驟
1) 啟動環境
```bash
conda activate social_xlstm
cd /path/to/Social-xLSTM
```

2) 準備配置（可用現有 profile，如 PEMS‑BAY 標準）
- `cfgs/profiles/pems_bay/standard.yaml`（包含 `cfgs/social_pooling/spatial_basic.yaml`）
- 建議調整：
  - `trainer.precision: "16-mixed"`
  - `data.loader.batch_size` 視顯存提高
  - `data.loader.num_workers: 8-16`

3) 執行訓練（Shared Encoder）
```bash
python scripts/train/with_social_pooling/train_shared.py \
  --config cfgs/profiles/pems_bay/standard.yaml \
  --output_dir blob/experiments/shared_pems
```

4) 觀察訓練
- 若社交池化啟用且 HDF5 含座標：會正常訓練。
- 若座標缺失：會在第一個 batch 前向直接報錯（嚴格模式），請補齊座標或關閉 pooling。

## 設計細節
- 模型融合與預測與分散式版本等價（個體 `[B,E]` 與社交 `[B,E]` concat 後送入 head）。
- 中央批次 collate 會回傳 `positions_xy: [N,2]`，模型內部在前向時複製為 `[B,T,2]` 供 pooling 使用。
- 如需進階（可訓練）社交池化，請參考 docs/_work/task.md 的 P0 計畫（attention 與 learnable radius）。

## 常見問題
- OOM：降低 `batch_size` 或提高 `accumulate_grad_batches`；開啟 gradient checkpointing。
- GPU 利用率低：提高 `batch_size`、使用 `precision: "16-mixed"`、增加 `num_workers`/`prefetch_factor`。
- 社交無效：確認 HDF5 是否含座標；或暫時關閉 `social_pooling.enabled`。
