# 專案整理計劃 (Project Cleanup Plan)

> 生成日期: 2025-08-06
> 目標: 清理 LLM 生成的一次性文件，恢復專案整潔性

## 現況評估

### 問題區域統計
```
實驗日誌: 104MB (logs/ + blob/experiments/)
臨時文件: 490 個 (.log, .tmp, .bak, ~)
大型報告: 108KB (vd_data_quality_report.md)
文檔分散: 19 個 MD 文件散布於 docs/
```

### 主要混亂來源
1. **實驗產出積累** - logs/default (55MB), blob/experiments (49MB)
2. **一次性分析報告** - vd_data_quality_report.md 等大型文檔
3. **臨時檔案未清理** - 490 個暫存檔案
4. **文檔結構不明確** - 缺乏 legacy/ 目錄歸檔舊文件

---

## 清理策略

### Phase 1: 立即清理 (高優先級)

#### 1.1 清理臨時文件
```bash
# 備份重要日誌
mkdir -p archive/logs_backup_$(date +%Y%m%d)
cp -r logs/reports archive/logs_backup_$(date +%Y%m%d)/

# 清理臨時文件
find . -type f \( -name "*.tmp" -o -name "*~" -o -name "*.bak" \) -delete
find . -type f -name "*.log" -mtime +7 -delete  # 刪除 7 天前的日誌
```

#### 1.2 壓縮實驗日誌
```bash
# 壓縮舊實驗
tar -czf archive/experiments_$(date +%Y%m%d).tar.gz blob/experiments/default
tar -czf archive/logs_default_$(date +%Y%m%d).tar.gz logs/default

# 清理已壓縮的目錄
rm -rf blob/experiments/default/*
rm -rf logs/default/*
```

### Phase 2: 文檔重組 (中優先級)

#### 2.1 建立歸檔結構
```bash
# 創建 legacy 目錄
mkdir -p docs/legacy
mkdir -p docs/reports
mkdir -p docs/analysis
```

#### 2.2 歸檔一次性報告
```bash
# 移動大型分析報告
mv vd_data_quality_report.md docs/reports/vd_data_quality_report_$(date +%Y%m%d).md

# 壓縮歸檔超大報告
gzip docs/reports/vd_data_quality_report_*.md
```

#### 2.3 整理 TODO 文件
```bash
# 備份當前 TODO
cp todo.md archive/todo_$(date +%Y%m%d).md

# 提取未完成項目到新文件
grep -E "^\[ \]" todo.md > todo_active.md
mv todo_active.md todo.md
```

### Phase 3: 建立維護機制 (低優先級)

#### 3.1 自動清理腳本
創建 `scripts/auto-cleanup.sh`:
```bash
#!/bin/bash
# 每週執行的清理腳本

# 清理 7 天前的日誌
find logs/ -name "*.log" -mtime +7 -delete

# 壓縮 30 天前的實驗
find blob/experiments/ -type d -mtime +30 | while read dir; do
    tar -czf "$dir.tar.gz" "$dir" && rm -rf "$dir"
done

# 清理空目錄
find . -type d -empty -delete
```

#### 3.2 Git 忽略規則更新
更新 `.gitignore`:
```gitignore
# 實驗產出
logs/default/
logs/dev/
blob/experiments/*/

# 臨時文件
*.tmp
*.bak
*~
*.log

# 大型報告
*_report_*.md
docs/reports/*.gz
```

---

## 執行檢查清單

### 立即執行 (5 分鐘)
- [ ] 備份 logs/reports 到 archive/
- [ ] 執行臨時文件清理指令
- [ ] 壓縮 logs/default 和 blob/experiments/default

### 短期執行 (30 分鐘)
- [ ] 建立 docs/legacy, docs/reports, docs/analysis 目錄
- [ ] 歸檔 vd_data_quality_report.md
- [ ] 精簡 todo.md 只保留未完成項目
- [ ] 更新 .gitignore

### 長期維護 (每週)
- [ ] 執行 scripts/auto-cleanup.sh
- [ ] 檢查 archive/ 目錄大小
- [ ] 評估是否需要外部儲存

---

## 保留原則

### 必須保留
- 最近 7 天的所有日誌
- docs/guides/ 中的操作指南
- docs/technical/ 中的技術文檔
- 最新版本的實驗結果

### 可以清理
- 超過 7 天的 .log 文件
- 所有 .tmp, .bak, ~ 文件
- 重複的實驗運行結果
- 一次性的分析報告

### 需要歸檔
- 重要的歷史實驗結果
- 大型的數據品質報告
- 已完成的 TODO 項目

---

## 預期成果

執行完整清理後：
- **空間節省**: ~100MB
- **文件減少**: ~500 個
- **結構改善**: 清晰的文檔分類
- **維護簡化**: 自動清理機制

---

## 安全提醒

1. **執行前備份**: 所有刪除操作前先備份
2. **測試指令**: 先用 `ls` 或 `echo` 測試檔案選擇
3. **分階段執行**: 不要一次執行所有清理
4. **檢查 Git 狀態**: 確保重要變更已提交