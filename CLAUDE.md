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

- `docs/guides/quickstart/` - 快速入門指南  
- `docs/PROJECT_STATUS.md` - 專案狀態  
- `todo.md` - 優先任務

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
| `src/social_xlstm/interfaces/vd_manager.py` | VD 實例管理器 |
| `src/social_xlstm/models/social_pooling.py` | 空間聚合 |
| `src/social_xlstm/training/trainer.py` | 訓練框架 |

---

## 💻 常用工作流

| 目標 | 指令 |
|------|------|
| 單/多 VD 訓練 | `snakemake train_*_without_social_pooling --cores 4 --configfile cfgs/snakemake/dev.yaml` |
| 測試 | `pytest -n auto` |
| 開發檢查 | `scripts/quick-dev-check.sh` |
| 架構檢查 | `python scripts/check_architecture_rules.py` |

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
| *開發階段可用 `[skip ci]` 跳過自動檢查* | |

---

## 🧪 TDD Shortcut

**Red** → **Green** → **Refactor**：失敗測試 → 最少實作 → 重構優化

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