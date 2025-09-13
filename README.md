## 訓練程式
* --config     要是 profiles 裡面的 yaml ，裡面配置所有的細解子設定檔案
* --output_dir 為結果的路徑
```
python scripts/train/with_social_pooling/train_shared.py \              
       --config cfgs/profiles/pems_bay/standard.yaml    \
       --output_dir blob/experiments/pems_bar/standard/
```
## 繪製結果圖
```
python scripts/utils/generate_metrics_plots.py \
       --experiment_dir blob/experiments/pems_bar/standard/metrics
```