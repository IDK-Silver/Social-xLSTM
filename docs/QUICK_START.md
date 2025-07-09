# 🚀 Claude Code 快速入門指南

**目標**: 讓新的 Claude Code 會話能在 5 分鐘內了解專案狀態並開始工作

## 📋 必讀文檔清單 (按順序)

### 1. 專案概況 (1 分鐘)
```bash
# 快速了解專案狀態
python scripts/utils/claude_init.py --quick
```

### 2. 技術決策 (2 分鐘)
```bash
# 核心技術選擇
cat docs/adr/0100-social-pooling-vs-graph-networks.md  # Social Pooling 決策
cat docs/adr/0101-xlstm-vs-traditional-lstm.md         # xLSTM 決策
```

### 3. 當前狀態 (1 分鐘)
```bash
# 開發優先級
cat docs/adr/0300-next-development-priorities.md
```

### 4. 程式碼結構 (1 分鐘)
```bash
# 關鍵實現
head -50 src/social_xlstm/models/lstm.py              # 統一 LSTM
head -50 src/social_xlstm/utils/spatial_coords.py     # 座標系統
head -50 src/social_xlstm/training/trainer.py         # 訓練系統
```

## 🎯 當前開發重點

### ✅ 已完成
- LSTM 實現統一 (5→1 個實現)
- 訓練腳本重構 (減少 48% 重複)
- ADR 系統建立 (7 個核心決策)
- 座標系統實現
- 評估框架建立

### 📋 下一步 (按優先級)
1. **Social Pooling 算法** - 基於座標的空間聚合
2. **xLSTM 整合** - sLSTM + mLSTM 混合架構
3. **Social-xLSTM 模型** - 結合兩者的完整模型
4. **實驗驗證** - 效果評估與基準比較

## 🛠️ 開發環境設置

```bash
# 環境激活
conda activate social_xlstm
pip install -e .

# 測試環境
python scripts/train/test_training_scripts.py --quick
```

## 📖 關鍵 ADR 決策摘要

| ADR | 決策 | 狀態 | 影響 |
|-----|------|------|------|
| 0100 | Social Pooling vs Graph Networks | ✅ 已決策 | 選擇座標驅動方法 |
| 0101 | xLSTM vs Traditional LSTM | ✅ 已決策 | 選擇混合架構 |
| 0200 | 座標系統選擇 | ✅ 已實施 | 墨卡托投影系統 |
| 0002 | LSTM 實現統一 | ✅ 已完成 | 5→1 個實現 |
| 0400 | 訓練腳本重構 | ✅ 已完成 | 減少 48% 重複 |

## 🧪 快速測試命令

```bash
# 測試訓練腳本
python scripts/train/test_training_scripts.py --quick

# 單VD訓練測試
python scripts/train/train_single_vd.py --epochs 5 --batch_size 16

# 多VD訓練測試
python scripts/train/train_multi_vd.py --epochs 5 --batch_size 8 --num_vds 3
```

## 📁 關鍵檔案位置

```
Social-xLSTM/
├── src/social_xlstm/
│   ├── models/lstm.py                 # 統一LSTM實現
│   ├── utils/spatial_coords.py        # 座標系統
│   ├── training/trainer.py            # 訓練系統
│   └── evaluation/evaluator.py        # 評估框架
├── scripts/train/
│   ├── train_single_vd.py             # 單VD訓練
│   ├── train_multi_vd.py              # 多VD訓練
│   └── common.py                      # 共用函數
└── docs/adr/                          # 架構決策記錄
    ├── 0100-social-pooling-vs-graph-networks.md
    ├── 0101-xlstm-vs-traditional-lstm.md
    └── 0300-next-development-priorities.md
```

## 🚨 重要提醒

1. **環境**: 必須在 `social_xlstm` conda 環境中工作
2. **決策**: 所有技術選擇都已在 ADR 中決定
3. **重點**: 當前專注於 Social Pooling 實現
4. **基礎**: 所有必要的技術基礎設施已就緒

## 🎮 開始開發

```bash
# 1. 環境檢查
conda activate social_xlstm

# 2. 快速狀態檢查
python scripts/utils/claude_init.py --quick

# 3. 開始 Social Pooling 實現
# 參考 ADR-0100 的技術設計
# 基於 src/social_xlstm/utils/spatial_coords.py
```

---

**⚡ 5 分鐘快速入門完成！現在可以開始 Social Pooling 開發工作。**