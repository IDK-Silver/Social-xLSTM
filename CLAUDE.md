# CLAUDE.md

> 入口文件；聚焦 Claude 對話流程、啟動檢查與常用指令  
> 盡量控制在 120 行內以降低摘要損耗

---

## 🧭 Claude 對話流程

1. **理解**：完整理解中文需求  
2. **思考**：以英文進行技術推理與決策  
3. **回覆**：以中文回覆，保留關鍵英文術語與程式碼  
4. **檔案修改**：直接編輯現有檔案；使用原始檔名進行更新

---

## 🚀 啟動檢查（每次新會話）

- `docs/_work/task.md` - **架構重整計劃與進度** ⭐  
- `todo.md` - 優先任務  
- **重要**：了解當前 YAGNI 重構階段與已完成 PR-1 狀況

---

## ⚡ Quick Start

```bash
conda env create -f environment.yaml
conda activate social_xlstm
pip install -e .
snakemake --configfile cfgs/snakemake/dev.yaml --cores 4
```

---

## 📂 核心檔案

| 位置 | 角色 |
|------|------|
| `src/social_xlstm/models/xlstm.py` | xLSTM 核心實現 |
| `src/social_xlstm/models/vd_xlstm_manager.py` | VD 實例管理器 ⭐ |
| `src/social_xlstm/models/distributed_social_xlstm.py` | 主要模型整合 |
| `src/social_xlstm/utils/tensor_checks.py` | 張量驗證工具 (新) |

---

## 💻 常用工作流

| 目標 | 指令 |
|------|------|
| 單/多 VD 訓練 | `snakemake train_*_without_social_pooling --cores 4 --configfile cfgs/snakemake/dev.yaml` |
| 測試 | `pytest -n auto` |

---

## 📑 Immutable Assets（唯讀資源）

| 資源 | 指引 |
|------|------|
| `blob/dataset/` | 僅供讀取，任何寫入請改用暫存路徑 |
| `environment.yaml` | 由 Maintainer 管理；如需變更請開 Issue |
| 檔案結構 | 修改請直接覆寫原檔，保持原始檔名 |

---

## 📝 提交規範

| 變更類型 | Commit 標頭範例 |
|----------|------------------|
| 結構重構 | `Refactor: extract coordinate helper` |
| 新增功能 | `Add: implement Social Pooling aggregation` |
| 修復問題 | `Fix: handle NaN in processor` |
| 測試/文檔 | `Test: add unit tests` / `docs: add guide` |

---

## 🧪 TDD Shortcut

**Red** → **Green** → **Refactor**：失敗測試 → 最少實作 → 重構優化

---

## ⚠️  開發原則

**極簡優先**：新功能先實作最基本版本，無花裡胡哨功能
- 無早停機制：不要添加 EarlyStopping、複雜 callbacks
- 無複雜監控：避免過度的 logging、metrics 收集
- 核心功能先行：基本功能穩定後再考慮高級功能
- YAML 驅動：參數管理優先使用配置文件而非 CLI

**後續擴展**：當基本功能驗證無誤後，再逐步添加進階功能

## 🧹 架構整理中（2025-08-26 進行中）

**基於 YAGNI 原則的過度設計清理**：
- **問題**：大量 LLM 生成程式碼導致重複實現和不必要抽象
- **目標**：40-50% 程式碼減少，消除架構債務
- **進度**：已完成 PR-1 (TensorSpec移除，~200行淨減少)

**重點清理區域**：
- `src/social_xlstm/interfaces/` - 單一實現的抽象基類
- `src/social_xlstm/dataset/storage/` - 過度設計的資料結構
- 重複類別名稱衝突 (如兩個 `VDXLSTMManager`)

**保留核心架構**：
- `VDXLSTMManager` - Social-xLSTM 分散式處理核心組件
- `TrafficXLSTM` - xLSTM 實現本體
- `DistributedSocialXLSTMModel` - 主要模型整合

**實施計劃**：詳見 `docs/_work/task.md` 的 5 階段 PR 計劃

---

## 📐 張量維度標準規範

**B T N F 標準** (⚠️ 必須遵循)：**B** (Batch) | **T** (Time Steps) | **N** (VDs 數量) | **F** (特徵維度)

**格式**: 集中式 `Tensor[B, T, N, F]` | 分散式 `Dict[VD_ID, Tensor[B, T, F]]`

**註釋**: `# [B, T, N, F] - B=批次, T=時間步, N=VD數量, F=特徵`

---

## 📚 文檔結構（2025-08-06 重組）

**三層架構**：簡化為 3 個主目錄，提升 LLM 檢索效率
- `docs/guides/` - 使用指南（如何做）
- `docs/concepts/` - 概念說明（為什麼）  
- `docs/reference/` - API 參考（是什麼）

## 🔗 深入閱讀

`docs/guides/quickstart/` | `docs/concepts/architecture/`

## 🔔 記憶備忘

- 每當完成 todo.md 的內容要更新打勾狀態
- **重構狀態**：PR-1 完成，下步 PR-2 統一 VDXLSTMManager
- **禁止過度設計**：發現單一實現抽象立即移除
- **TensorSpec 已移除**：使用 `utils.tensor_checks` 簡單函數

## 🎯 當前重構階段

**已完成 PR-1**：TensorSpec 移除 (236→53行)
**進行中 PR-2**：VDXLSTMManager 統一 (移除 interfaces 版本)
**待辦 PR-3-5**：TrafficFeature 簡化、interfaces 清理、最終整理