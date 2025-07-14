# 架構決策記錄 (ADR) 系統

## 概述

本目錄包含 Social-xLSTM 專案的所有架構決策記錄。ADR 系統用於記錄重要的技術決策，包括決策背景、考慮的選項、最終決定及其後果。

## ADR 狀態說明

- **Proposed**: 提議中，待審核
- **Accepted**: 已接受，待實施
- **Partially Implemented**: 部分實施
- **Implemented**: 已完全實施
- **Deprecated**: 已棄用
- **Superseded**: 被其他 ADR 取代

## 當前 ADR 清單

### 核心架構 (0001-0099)
- [ADR-0001](0001-project-architecture-cleanup.md) - 專案架構清理與重組 (Partially Implemented)
- [ADR-0002](0002-lstm-implementation-unification.md) - LSTM 實現統一與標準化 (Partially Implemented)

### 技術選擇 (0100-0199)
- [ADR-0100](0100-social-pooling-vs-graph-networks.md) - Social Pooling vs Graph Networks (Accepted)
- [ADR-0101](0101-xlstm-vs-traditional-lstm.md) - xLSTM vs Traditional LSTM (Accepted)

### 座標系統 (0200-0299)
- [ADR-0200](0200-coordinate-system-selection.md) - 座標系統選擇 (Implemented)

### 開發流程 (0300-0399)
- [ADR-0300](0300-next-development-priorities.md) - 下一階段開發優先級規劃 (Accepted)

### 實施細節 (0400-0499)
- [ADR-0400](0400-training-script-refactoring.md) - 訓練腳本重構策略 (Implemented)

### 階段性總結 (0500-0599)
- [ADR-0500](0500-lstm-baseline-completion-next-steps.md) - LSTM 基準完成與下一步規劃 (Accepted)
- [ADR-0501](0501-xlstm-integration-strategy.md) - xLSTM 整合策略與並行訓練架構 (Proposed)

### Social Pooling 架構 (0600-0699)
- [ADR-0600](0600-social-pooling-integration-strategy.md) - Social Pooling 整合策略與雙重實現架構 (Accepted)

## 使用指南

### 查看 ADR 狀態
```bash
# 查看所有 ADR 狀態
grep -r "**狀態**" docs/adr/ | grep -v README

# 查看特定狀態的 ADR
grep -r "**狀態**: Accepted" docs/adr/
```

### 創建新 ADR
1. 使用下一個可用的編號
2. 複製 ADR 模板
3. 填寫所有必要章節
4. 更新此 README 文件

### 更新 ADR 狀態
1. 修改 ADR 文件頭部的狀態欄位
2. 添加更新日期
3. 如有需要，添加實施狀態章節
4. 更新此 README 文件

## 狀態追蹤機制

### 自動狀態檢查
```bash
# 檢查所有 ADR 狀態
cd docs/adr
for adr in *.md; do
    if [[ $adr != "README.md" ]]; then
        echo "=== $adr ==="
        grep "**狀態**" "$adr"
        echo
    fi
done
```

### 實施進度追蹤
每個 ADR 都應包含以下章節來追蹤實施進度：

```markdown
## 實施狀態
- [ ] 任務 1: 描述
- [ ] 任務 2: 描述
- [x] 任務 3: 描述 (已完成)

## 實施記錄
- 2025-01-09: 完成任務 3
- 2025-01-08: 開始實施
```

### 定期審查
建議每週審查 ADR 狀態：
1. 檢查是否有新的決策需要記錄
2. 更新正在實施的 ADR 進度
3. 標記已完成的 ADR 為 "Implemented"
4. 處理相互衝突的決策

## 參考資料

- [ADR 介紹](https://github.com/joelparkerhenderson/architecture-decision-record)
- [ADR 最佳實踐](https://docs.aws.amazon.com/prescriptive-guidance/latest/architectural-decision-records/)
- [專案 ADR 模板](../template/adr-template.md)

## 注意事項

1. **狀態更新**: 請及時更新 ADR 狀態以反映實際情況
2. **交叉引用**: 確保 ADR 間的引用正確無誤
3. **實施追蹤**: 重要決策應包含明確的實施計畫和進度追蹤
4. **定期審查**: 建議定期審查 ADR 系統的有效性

---

最後更新: 2025-07-14