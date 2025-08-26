# Social-xLSTM

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-189%2F189%20Passing-green.svg)](tests/)
[![Code Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](tests/)

> 🚀 **基於 Social Pooling 與 xLSTM 的無拓撲依賴交通流量預測系統**

Social-xLSTM 結合座標驅動的社會聚合機制與擴展長短期記憶網路，實現在缺乏完整道路拓撲資訊的情況下進行高精度交通預測。

## ✨ 核心亮點

- **🎯 無拓撲依賴**: 自動學習空間互動關係，無需預先定義道路網絡
- **⚡ 動態配置系統**: CLI 參數減少 **70-75%**（從 25+ 個減少到 4 個配置文件）
- **🔄 一鍵切換**: 支援多種社會聚合方法（attention、weighted_mean、weighted_sum）
- **📊 完整報告**: 自動生成訓練報告、視覺化圖表、比較分析
- **🏗️ 現代架構**: 分散式 xLSTM、混合記憶機制、模組化設計

## 🚀 快速開始

### 環境設置

```bash
# 建立環境
conda env create -f environment.yaml
conda activate social_xlstm
pip install -e .
```

### 一鍵訓練（推薦）

使用動態配置系統進行 Social-xLSTM 訓練：

```bash
# Attention-based 社會聚合
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/attention.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd --cores 2

# 切換聚合方法只需更改一個配置文件
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/weighted_mean.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd --cores 2

# 無社會聚合（基準比較）
python workflow/snakemake_warp.py \
  --configfile cfgs/models/xlstm.yaml \
  --configfile cfgs/social_pooling/off.yaml \
  --configfile cfgs/vd_modes/multi.yaml \
  --configfile cfgs/training/default.yaml \
  train_social_xlstm_multi_vd --cores 2
```

### 數據處理

```bash
# 完整數據管線
snakemake --cores 4

# 或手動執行關鍵步驟
python scripts/dataset/pre-process/create_h5_file.py \
  --source_dir blob/dataset/pre-processed/unzip_to_json \
  --output_path blob/dataset/pre-processed/h5/traffic_features_default.h5
```

## 📋 配置系統

### snakemake_warp.py - 統一工作流程工具

`snakemake_warp.py` 是項目的核心工作流程工具，負責配置合併和自動化執行：

**核心功能**：
- **配置合併**: 自動合併多個 YAML 配置文件
- **環境變數傳遞**: 設置 `SNAKEMAKE_MERGED_CONFIG` 給下游使用
- **簡化參數**: 從 25+ CLI 參數減少到 4 個配置文件
- **統一執行**: 取代直接使用 `snakemake` 指令

**基本語法**：
```bash
python workflow/snakemake_warp.py \
  --configfile config1.yaml \
  --configfile config2.yaml \
  --configfile config3.yaml \
  target_rule --cores N
```

**為什麼使用 snakemake_warp.py？**
- ✅ 避免 Snakemake 多配置文件時序問題
- ✅ 確保配置正確合併和傳遞
- ✅ 統一的實驗管理方式
- ✅ 支援複雜的消融研究配置

### 四層 YAML 配置架構

```
cfgs/
├── models/           # 純模型架構配置
│   ├── lstm.yaml    # 傳統 LSTM
│   └── xlstm.yaml   # 擴展 xLSTM
├── social_pooling/   # 社會聚合配置
│   ├── off.yaml     # 無聚合（基準）
│   ├── weighted_mean.yaml
│   ├── weighted_sum.yaml
│   └── attention.yaml
├── vd_modes/        # VD 模式配置
│   ├── single.yaml  # 單點預測
│   └── multi.yaml   # 多點預測
└── training/        # 訓練超參數
    └── default.yaml
```

### 配置範例

**模型配置** (`xlstm.yaml`):
```yaml
model:
  name: "TrafficXLSTM"
  xlstm:
    input_size: 3
    embedding_dim: 64
    num_blocks: 4
    slstm_at: [1, 3]
    dropout: 0.5
```

**社會聚合配置** (`attention.yaml`):
```yaml
social:
  enabled: true
  pooling_radius: 2500.0
  max_neighbors: 10
  aggregation_method: "attention"
  distance_metric: "euclidean"
```

## 🎛️ 實驗工作流

### 消融研究支援

```bash
# 比較不同聚合方法
for method in off weighted_mean weighted_sum attention; do
  python workflow/snakemake_warp.py \
    --configfile cfgs/models/xlstm.yaml \
    --configfile cfgs/social_pooling/${method}.yaml \
    --configfile cfgs/vd_modes/multi.yaml \
    --configfile cfgs/training/default.yaml \
    train_social_xlstm_multi_vd --cores 2
done
```

### 報告生成

```bash
# 生成單一實驗報告
python scripts/utils/generate_training_report.py \
  --experiment_dir blob/experiments/dev/social_xlstm/multi_vd

# 生成模型比較報告
python workflow/snakemake_warp.py generate_model_comparison_report --cores 1

# 生成社會聚合方法比較
python workflow/snakemake_warp.py generate_social_pooling_comparison_report --cores 1
```

## 🏗️ 項目架構

```
Social-xLSTM/
├── cfgs/                     # 🔧 四層配置系統
│   ├── models/              # 模型架構配置
│   ├── social_pooling/      # 社會聚合配置
│   ├── vd_modes/           # VD 模式配置
│   └── training/           # 訓練參數配置
├── scripts/
│   ├── train/
│   │   ├── with_social_pooling/    # 🚀 Social-xLSTM 訓練
│   │   └── without_social_pooling/ # 基準模型訓練
│   └── utils/              # 報告生成工具
├── src/social_xlstm/        # 📦 核心套件
│   ├── models/             # 模型實現
│   │   ├── xlstm.py       # xLSTM 核心
│   │   ├── social_pooling.py  # 社會聚合
│   │   └── distributed_social_xlstm.py
│   ├── config/             # 動態配置管理
│   ├── training/           # 訓練框架
│   └── visualization/      # 報告視覺化
├── workflow/
│   ├── snakemake_warp.py   # 🔄 配置合併工具
│   └── rules/              # Snakemake 規則
└── docs/                   # 📚 完整文檔系統
```

## 📊 系統特性

### 性能指標
- **參數效率**: 從 25+ CLI 參數減少到 4 個配置文件（**70-75% 減少**）
- **模型規模**: TrafficXLSTM (654K 參數)，Multi-VD (1.4M 參數)
- **測試覆蓋**: 189/189 測試通過（**100% 通過率**）
- **數據規模**: 66,371 筆台灣交通流量資料

### 支援的模型
| 模型 | 參數量 | 特性 |
|------|--------|------|
| TrafficLSTM | 226K | 單VD基準模型 |
| TrafficXLSTM | 655K | sLSTM + mLSTM 混合 |
| Multi-VD LSTM | 1.4M | 多點空間關聯 |
| Social-xLSTM | 1.4M+ | 無拓撲社會聚合 |

### 社會聚合方法
- **Off**: 無聚合（基準比較）
- **Weighted Mean**: 距離加權平均（行歸一化）
- **Weighted Sum**: 距離加權求和（無歸一化）
- **Attention**: 注意力機制（Softmax 歸一化）

## 🧪 使用範例

### 基本訓練

```bash
# 單 VD 訓練（基準）
python scripts/train/without_social_pooling/train_single_vd.py \
  --data_path blob/dataset/pre-processed/h5/traffic_features_dev.h5 \
  --epochs 50 --batch_size 16

# Multi-VD 訓練（空間關聯）
python scripts/train/without_social_pooling/train_multi_vd.py \
  --data_path blob/dataset/pre-processed/h5/traffic_features_dev.h5 \
  --selected_vdids VD-28-0740-000-001 VD-11-0020-008-001 VD-13-0660-000-002

# Social-xLSTM 訓練（完整功能）
python scripts/train/with_social_pooling/train_distributed_social_xlstm.py \
  --config-file cfgs/merged_config.yaml \
  --data_path blob/dataset/pre-processed/h5/traffic_features_dev.h5
```

### 批量實驗

```bash
# 使用 Snakemake 批量執行
python workflow/snakemake_warp.py \
  train_single_vd_without_social_pooling \
  train_multi_vd_without_social_pooling \
  train_social_xlstm_multi_vd --cores 3

# 生成完整報告
python workflow/snakemake_warp.py generate_experiment_summary_report --cores 1
```

## 📚 文檔資源

- **[動態配置系統指南](docs/guides/dynamic-configuration-system.md)** - 完整配置使用說明
- **[Social Pooling 訓練指南](docs/guides/training-with-sp.md)** - 社會聚合訓練流程
- **[快速入門系列](docs/guides/quickstart/)** - 新手入門指南
- **[API 參考](docs/reference/api-reference.md)** - 完整 API 文檔
- **[數學規格](docs/concepts/mathematical-specifications.md)** - 算法數學定義

## 💻 系統需求

- **Python**: 3.11+
- **GPU**: CUDA 12.4+ （推薦）
- **RAM**: 16GB+
- **Storage**: 50GB+

### 主要依賴
```yaml
pytorch: 2.0+
pytorch-lightning: 2.0+
xlstm: latest
h5py: 3.8+
snakemake: 7.0+
matplotlib: 3.7+
```

## 🔬 研究背景

**項目資訊**：
- **編號**: NUTN-CSIE-PRJ-115-006
- **學校**: 國立臺南大學資訊工程學系
- **指導教授**: 陳宗禧 教授
- **研究團隊**: 黃毓峰 (S11159005)、唐翊靜 (S11159028)

**核心創新**：
1. **座標驅動社會聚合** - 使用連續空間距離取代傳統網格方法
2. **混合記憶架構** - 結合 sLSTM 和 mLSTM 的高容量記憶
3. **無拓撲依賴** - 自動學習節點間空間互動關係
4. **動態配置管理** - 大幅簡化實驗配置和消融研究

## 🤝 開發貢獻

```bash
# Fork 專案
git fork https://github.com/your-org/Social-xLSTM

# 創建功能分支
git checkout -b feature/amazing-feature

# 執行測試
pytest -n auto

# 提交更改
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

## 📄 授權條款

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 檔案

## 📞 聯絡支援

- **Issues**: [GitHub Issues](https://github.com/your-org/Social-xLSTM/issues)
- **文檔**: [完整文檔系統](docs/)
- **討論**: [GitHub Discussions](https://github.com/your-org/Social-xLSTM/discussions)

---

⭐ **如果這個專案對您有幫助，請給我們一個 Star！**