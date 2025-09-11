# Social-xLSTM

Social-xLSTM 是一個結合 xLSTM 與座標驅動社交池化（social pooling）的交通時序預測框架，支援共享編碼器與分散式兩種訓練路徑。

核心能力
- xLSTM 時序建模（sLSTM + mLSTM）
- 座標距離式社交池化（需提供每個 VD 的經緯度）
- 嚴格模式：啟用 pooling 時必須提供座標，否則報錯

## 安裝

```bash
conda env create -f environment.yaml
conda activate social_xlstm
pip install -e .
```

# 資料格式

## 數據與 HDF5 結構（統一規格）
- `data/features`: 時序特徵張量，形狀 `[T, N, F]`
- `metadata/vdids`: `N` 個 VDID，順序對應 `features` 的第 2 維
- `metadata/feature_names`: `F` 個特徵名
- `metadata/timestamps` 與 `metadata/timestamps_epoch`
- `metadata/vd_info/<vdid>`（必備）：子群組屬性
  - `position_lat`, `position_lon`（WGS84）
  - 可選：`lanes`, `length`, `direction`；群組屬性 `coord_crs='EPSG:4326'`

轉檔腳本（已寫入 vd_info）：
- `scripts/dataset/pre_process/pems_bay/convert_pems_bay_to_hdf5.py`
- `scripts/dataset/pre_process/metr_la/convert_metr_la_to_hdf5.py`

若你的舊 H5 未包含 `metadata/vd_info`，請重新轉檔或回填；否則嚴格模式下的社交池化會報缺座標錯誤。

## 訓練方式

- 共享編碼器（建議，GPU 利用率高）
  - 批次：`[B, T, N, F]`
  - 指令：`scripts/train/with_social_pooling/train_shared.py`
    - 例：`python scripts/train/with_social_pooling/train_shared.py --config cfgs/profiles/pems_bay/standard.yaml --output_dir blob/experiments/shared_pems`
  - 條件：`social_pooling.enabled: true` 時，批次需含 `positions_xy: [N,2]`（由 DataModule 依 `vd_info` 產生）。

- 分散式（每 VD 單獨編碼）
  - 批次：`{vd_id: [B, T, F]}`
  - 指令：`scripts/train/with_social_pooling/train_multi_vd.py`
    - 例：`python scripts/train/with_social_pooling/train_multi_vd.py --config cfgs/profiles/pems_bay/standard.yaml --output_dir blob/experiments/distributed`
  - 條件：`social_pooling.enabled: true` 時，批次需含 `positions[vd_id]: [B,T,2]`。

## 快速測試與工具
- 時間切片：`scripts/utils/h5_time_slice.py`
- 指標可視化：`scripts/utils/generate_metrics_plots.py`

## 設定建議
- 配置入口：`cfgs/profiles/pems_bay/standard.yaml`
- 精度：`trainer.precision: "16-mixed"`
- DataLoader：`data.loader.num_workers: 8-16`，視顯存提升 `batch_size`

#### 🔧 標準快速測試（全 VD，2-3分鐘完成）

適用於完整功能驗證：

```bash
# 使用標準快速測試 Profile 進行訓練（約 2-3 分鐘完成）
python scripts/train/with_social_pooling/train_multi_vd.py \
  --config cfgs/profiles/pems_bay_fast_test.yaml \
  --output_dir blob/experiments/fast_test

# 比較結果並迭代優化
python scripts/utils/generate_metrics_plots.py \
  --experiment_dir blob/experiments/fast_test/metrics
```

## 嚴格模式（Social Pooling）
- 啟用 `social_pooling.enabled: true` 時：
  - 共享編碼器需 `positions_xy: [N,2]`
  - 分散式需 `positions[vd_id]: [B,T,2]`
  - 缺失、NaN、shape 錯誤將直接 RuntimeError
- 座標來源：HDF5 `metadata/vd_info/<vdid>`（轉檔腳本已寫入）

## 專案結構
```
src/social_xlstm/            核心程式碼（models, dataset, metrics, utils）
scripts/                     訓練與資料處理腳本
cfgs/                        設定檔（profiles 合併 data/model/training/social_pooling）
blob/                        資料與實驗輸出（ignored）
docs/                        說明文件與指引
```

## 文件
- `docs/guides/training-with-shared-encoder.md`：共享編碼器訓練指南
- `docs/guides/training-with-sp.md`：分散式訓練與社交池化指南
- `docs/reference/configuration-reference.md`：設定參考

## 系統需求
- Python 3.11+
- PyTorch 2.0+
- CUDA 驅動（GPU 訓練可選）

## 授權
MIT License（見 `LICENSE`）

**基於 YAGNI 原則的現代化架構** | **支持 PEMS-BAY 和 Taiwan VD 數據集** | **輕量級指標系統**
