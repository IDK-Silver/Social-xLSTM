## 兩種訓練型態（簡介）

- Independent xLSTM：每個 VD 一個獨立的 xLSTM。彼此不交互，參數量大、速度慢，但隔離性高，適合小規模或異質 VD 基準。
- Shared Encoder（社交池化）：所有 VD 共用一個編碼器，最後用社交池化融合鄰近資訊。參數少、速度快，適合多 VD 場景與互動建模。

## 什麼時候用哪個

- 用 Independent：VD 數量少、需要逐 VD 可比的獨立基準。
- 用 Shared Encoder：VD 多、重視效率與互動（空間鄰近）訊息。

## 快速開始（Shared Encoder）

```bash
# 快速評估（小清單、快訓練）
python scripts/train/with_social_pooling/shared_encoder.py \
  --config cfgs/profiles/pems_bay/fast_evaluation.yaml \
  --output_dir blob/experiments/pems_bay/fast_evaluation

python scripts/train/with_social_pooling/shared_encoder.py \
  --config cfgs/profiles/metr_la/fast_evaluation.yaml \
  --output_dir blob/experiments/metr_la/fast_evaluation


# 標準設定（較完整）
python scripts/train/with_social_pooling/shared_encoder.py \
  --config cfgs/profiles/pems_bay/standard.yaml \
  --output_dir blob/experiments/pems_bay/standard
```

選配：訓練後執行測試集評估（並寫入同一份 metrics 檔）

```bash
... --eval_test --test_ckpt_mode best   # 使用最佳權重（需開啟 ModelCheckpoint）
# 或指定 checkpoint：
... --eval_test --test_ckpt_path path/to/epoch=XX.ckpt
```

## 指標與報表（重點）

- 主要指標（MAE/MSE/RMSE/R²）在「反標準化後的真實單位」上計算與輸出，可直接與論文對比。
- 亦會在 log 中輸出標準化版本（`*_norm`）供除錯；若要一併寫入 CSV，可在 profile 設：

```yaml
trainer:
  callbacks:
    training_metrics:
      include_normalized: true
      splits: [train, val, test]
```

- 指標輸出：`<output_dir>/metrics/metrics.csv` 與 `metrics_summary.json`。
- 產圖：`python scripts/utils/generate_metrics_plots.py --metrics-csv <metrics.csv>`。