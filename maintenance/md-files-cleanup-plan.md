# Markdown 文件整理計劃

> 執行日期: 2025-08-06  
> 目標: 將所有 .md 文件整理到 docs/ 目錄（除了 todo.md）

## 現況分析

### 文件分布統計
- **根目錄**: 5 個文件 (包含 todo.md)
- **子目錄 README**: 9 個散落在各處
- **大型報告**: vd_data_quality_report.md (108KB)
- **核心文件**: README.md, CLAUDE.md, CHANGELOG.md

### 需要保留在根目錄
```
/
├── todo.md          # 主要任務清單（保留）
├── README.md        # 專案入口（保留）
├── CLAUDE.md        # Claude 對話流程（保留）
└── CHANGELOG.md     # 變更紀錄（保留）
```

## 整理策略

### 1. 報告類文件歸檔
```bash
# 創建報告目錄
mkdir -p docs/reports/

# 移動大型報告
mv vd_data_quality_report.md docs/reports/vd_data_quality_report_20250805.md
```

### 2. 工具說明文件整合
```bash
# 創建工具文檔目錄
mkdir -p docs/tools/

# 整合工具 README
cat tools/README.md > docs/tools/tools-overview.md
cat tools/analysis/README.md >> docs/tools/analysis-tools.md
cat tools/validation/README.md >> docs/tools/validation-tools.md

# 刪除原始 README
rm tools/README.md tools/analysis/README.md tools/validation/README.md
```

### 3. 腳本文檔整合
```bash
# 創建腳本文檔目錄
mkdir -p docs/scripts/

# 整合訓練腳本說明
cat scripts/utils/README.md > docs/scripts/utils-guide.md
cat scripts/train/without_social_pooling/README.md > docs/scripts/training-without-sp.md
cat scripts/train/with_social_pooling/post_fusion/README.md > docs/scripts/training-with-sp.md

# 刪除原始 README
rm scripts/utils/README.md
rm scripts/train/without_social_pooling/README.md
rm scripts/train/with_social_pooling/post_fusion/README.md
```

### 4. 配置文檔整合
```bash
# 移動配置說明
mv cfgs/README.md docs/reference/configuration-guide.md
```

### 5. 測試文檔整合
```bash
# 移動測試說明
mv tests/README.md docs/reference/testing-guide.md
```

## 執行命令序列

```bash
# 步驟 1: 創建目錄結構
mkdir -p docs/reports
mkdir -p docs/tools
mkdir -p docs/scripts

# 步驟 2: 移動報告文件
mv vd_data_quality_report.md docs/reports/vd_data_quality_report_20250805.md

# 步驟 3: 整合工具文檔
if [ -f tools/README.md ]; then
    mv tools/README.md docs/tools/tools-overview.md
fi
if [ -f tools/analysis/README.md ]; then
    mv tools/analysis/README.md docs/tools/analysis-tools.md
fi
if [ -f tools/validation/README.md ]; then
    mv tools/validation/README.md docs/tools/validation-tools.md
fi

# 步驟 4: 整合腳本文檔
if [ -f scripts/utils/README.md ]; then
    mv scripts/utils/README.md docs/scripts/utils-guide.md
fi
if [ -f scripts/train/without_social_pooling/README.md ]; then
    mv scripts/train/without_social_pooling/README.md docs/scripts/training-without-sp.md
fi
if [ -f scripts/train/with_social_pooling/post_fusion/README.md ]; then
    mv scripts/train/with_social_pooling/post_fusion/README.md docs/scripts/training-with-sp.md
fi

# 步驟 5: 移動配置和測試文檔
if [ -f cfgs/README.md ]; then
    mv cfgs/README.md docs/reference/configuration-guide.md
fi
if [ -f tests/README.md ]; then
    mv tests/README.md docs/reference/testing-guide.md
fi
```

## 預期結果

### 整理後結構
```
/
├── README.md          # 專案入口
├── CLAUDE.md          # Claude 對話流程
├── CHANGELOG.md       # 變更紀錄
├── todo.md            # 任務清單
└── docs/
    ├── architecture/  # 架構文檔
    ├── decisions/     # 決策記錄
    ├── guides/        # 使用指南
    ├── papers/        # 論文相關
    ├── quickstart/    # 快速開始
    ├── reference/     # 參考文檔
    │   ├── configuration-guide.md
    │   └── testing-guide.md
    ├── reports/       # 分析報告
    │   └── vd_data_quality_report_20250805.md
    ├── scripts/       # 腳本說明
    │   ├── utils-guide.md
    │   ├── training-without-sp.md
    │   └── training-with-sp.md
    ├── technical/     # 技術規範
    └── tools/         # 工具文檔
        ├── tools-overview.md
        ├── analysis-tools.md
        └── validation-tools.md
```

### 效益評估
- **文件數量**: 減少 9 個散落的 README
- **結構清晰**: 所有文檔集中在 docs/
- **易於維護**: 統一的文檔管理
- **符合規範**: 遵循標準專案結構

## 注意事項

1. **保留原則**:
   - README.md, CLAUDE.md, CHANGELOG.md 保留在根目錄
   - todo.md 保留在根目錄供頻繁訪問
   
2. **執行前備份**:
   ```bash
   tar -czf docs_backup_$(date +%Y%m%d).tar.gz *.md tools/ scripts/ cfgs/ tests/
   ```

3. **更新引用**:
   - 檢查是否有其他文件引用這些 README
   - 更新 Snakemake 規則中的文檔路徑（如有）

4. **Git 提交**:
   ```bash
   git add -A
   git commit -m "Refactor: consolidate all markdown documentation into docs/ directory"
   ```