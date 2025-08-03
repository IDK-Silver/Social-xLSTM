# Social-xLSTM

基於社交池化與擴展長短期記憶網路（xLSTM）的交通流量預測研究

## 專案概述

Social-xLSTM 是一個創新的交通流量預測研究專案，旨在解決現有時空圖模型依賴完整道路拓撲資訊的限制。本專案結合 Social Pooling 機制與 xLSTM 技術，提出能在缺乏完整拓撲資訊的情況下，透過座標驅動的社交池化自動學習節點間空間互動關係的交通預測模型。

### 核心創新
- **無拓撲依賴**：不需預先指定道路拓撲或鄰接關係
- **距離基礎社交池化**：使用連續空間距離進行社交聚合，改進原始 Social LSTM 的網格方法
- **混合記憶機制**：結合 sLSTM 和 mLSTM 的高容量記憶
- **自適應節點分佈**：適應不規則節點分佈和動態交通環境

> **架構說明**: 本實現採用距離基礎的連續社交池化方法，而非原始 Social LSTM 論文的網格基礎方法。詳見 [架構決策記錄](docs/decisions/adr-001-distance-based-social-pooling.md)。

詳細專案說明請參考 [專案狀態文檔](docs/legacy/PROJECT_STATUS.md)

## 專案狀態

### 已完成功能 ✅

- **數據蒐集與處理**
  - [x] 台灣公路總局即時交通資料蒐集（66,371 筆，約 1.5 個月）
  - [x] XML 到 JSON 轉換器
  - [x] JSON 到 HDF5 高效存儲
  - [x] 按地區分塊最佳化
  - [x] Snakemake 自動化工作流

- **核心模組**
  - [x] PyTorch Lightning 數據載入器
  - [x] 統一 LSTM 實現 (TrafficLSTM)
  - [x] 專業化訓練架構 (SingleVD, MultiVD, IndependentMultiVD)
  - [x] 模型評估框架（MAE, MSE, RMSE, MAPE, R²）
  - [x] 感測器座標處理系統

- **實用工具**
  - [x] VD 座標視覺化（高屏交界區域）
  - [x] 空間座標轉換
  - [x] 圖結構處理工具

### 開發中功能 🚧

- [ ] **Social xLSTM 核心實現**
  - [ ] 座標驅動社交池化層
  - [ ] Hybrid xLSTM（sLSTM + mLSTM）
  - [ ] 指數門控機制
- [ ] 模型訓練與驗證
- [ ] 效能評估與基準比較

### 待開發功能 📋

- [ ] **研究驗證**
  - [ ] 多種交通數據集實證研究
  - [ ] 與現有時空模型效能比較
  - [ ] 模型解釋性分析
- [ ] **應用設計**
  - [ ] 即時預測系統
  - [ ] 視覺化介面
  - [ ] 模型部署與服務化

## 快速開始

### 環境設置

```bash
# 創建 conda 環境
conda env create -f environment.yaml
conda activate social_xlstm

# 安裝套件（開發模式）
pip install -e .
```

### 數據處理

```bash
# 執行完整數據管線
snakemake --cores 4

# 或手動執行各步驟
python scripts/dataset/pre-process/list_all_zips.py --input_folder_list <folders> --output_file_path <output>
python scripts/dataset/pre-process/unzip_and_to_json.py --input_zip_list_path <input> --output_folder_path <output>
python scripts/dataset/pre-process/create_h5_file.py --source_dir <dir> --output_path <path>
```

### 模型訓練

```bash
# 確保環境已激活
conda activate social_xlstm

# 單VD訓練 (基準模型)
python scripts/train/without_social_pooling/train_single_vd.py

# 多VD訓練 (空間關係)
python scripts/train/without_social_pooling/train_multi_vd.py

# 獨立多VD訓練 (基準比較)
python scripts/train/without_social_pooling/train_independent_multi_vd.py

# 或使用 Snakemake 執行
snakemake train_single_vd_without_social_pooling --cores 1
snakemake train_multi_vd_without_social_pooling --cores 1
snakemake train_independent_multi_vd_without_social_pooling --cores 1

# 並行執行所有訓練
snakemake train_single_vd_without_social_pooling train_multi_vd_without_social_pooling train_independent_multi_vd_without_social_pooling --cores 3

# 訓練 Social xLSTM 模型（開發中）
# python scripts/train/with_social_pooling/train_social_xlstm.py
```

## 專案結構

```
Social-xLSTM/
├── docs/                      # 專案文檔
│   ├── architecture/         # 架構設計文檔
│   ├── decisions/            # 架構決策記錄 (ADRs)
│   ├── papers/               # 相關論文整理
│   ├── quickstart/           # 快速入門指南
│   ├── technical/            # 技術文檔
│   └── reference/            # 參考文檔
├── src/social_xlstm/         # 核心套件
│   ├── dataset/              # 數據處理模組
│   ├── models/               # 深度學習模型
│   ├── evaluation/           # 評估指標
│   ├── utils/                # 實用工具
│   └── visualization/        # 視覺化工具
├── scripts/                  # 主要執行腳本
│   ├── dataset/pre-process/  # 數據預處理
│   ├── train/                # 訓練腳本
│   │   ├── without_social_pooling/  # 無社交池化訓練
│   │   └── with_social_pooling/     # 社交池化訓練 (開發中)
│   └── utils/                # 核心執行工具
├── tools/                    # 開發者工具
│   ├── config/               # 配置生成工具
│   ├── analysis/             # 數據分析工具
│   ├── diagnostics/          # 診斷與檢查工具
│   └── validation/           # 驗證工具
├── tests/                    # 測試套件
│   ├── unit/                 # 單元測試
│   ├── integration/          # 整合測試
│   └── functional/           # 功能測試
├── notebooks/                # 探索性分析 (本地開發)
├── Snakefile                 # 工作流定義
├── cfgs/                     # 配置檔案
└── environment.yaml          # Conda 環境
```

## 系統需求

- Python 3.11+
- CUDA 12.4+ (GPU 訓練)
- 至少 16GB RAM
- 50GB+ 儲存空間（視數據量而定）

## 主要依賴

- PyTorch 2.0+
- PyTorch Lightning
- xlstm (擴展 LSTM 實現)
- h5py (HDF5 支援)
- Snakemake (工作流管理)
- scikit-learn
- NumPy, Pandas (數據處理)
- Matplotlib, Seaborn (視覺化)

完整依賴列表請參考 `environment.yaml`

## 開發者工具

### Tools 目錄
`tools/` 目錄包含專案開發和維護所需的各種工具，這些工具不是核心功能的一部分，但對開發者很有用：

```bash
# 配置工具
python tools/config/config_generator.py --type optimized --h5_path data.h5

# 分析工具
python tools/analysis/data_quality_analysis.py --input data.h5
python tools/analysis/temporal_pattern_analysis.py --data_path data.h5

# 診斷工具
python tools/diagnostics/h5_structure_inspector.py --input data.h5
python tools/diagnostics/data_stability_tools.py --check stability

# 驗證工具
python tools/validation/temporal_split_validation.py --data_path data.h5
python tools/validation/training_validation.py --model_path model.pt
```

### Notebooks 目錄
`notebooks/` 目錄用於探索性分析和實驗，該目錄已加入 `.gitignore`，適合個人開發和調試使用。

詳細的工具使用說明請參考 [tools/README.md](tools/README.md)

## 模組功能

詳細的模組功能說明請參考 [模組文檔](docs/implementation/modules.md)

## 貢獻指南

1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 研究資訊

**專案編號**：NUTN-CSIE-PRJ-115-006  
**研究團隊**：
- 黃毓峰 (S11159005)
- 唐翊靜 (S11159028)

**指導教授**：陳宗禧 教授  
**學校**：國立臺南大學資訊工程學系

## 授權

本專案採用 MIT 授權 - 詳見 LICENSE 檔案

## 聯絡資訊

如有問題或建議，請透過 GitHub Issues 聯繫我們。