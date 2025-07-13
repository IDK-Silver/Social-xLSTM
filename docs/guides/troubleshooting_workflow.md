# 故障排除工作流程指南

本文檔記錄了解決專案中常見問題的標準工作流程。

## 🔍 問題診斷流程

### 1. 錯誤分類
```
📋 輸出錯誤 → 檢查日誌檔案
🔧 配置錯誤 → 檢查參數和路徑
💾 數據錯誤 → 檢查資料格式和內容
🏗️ 架構錯誤 → 檢查模型和訓練器配置
```

### 2. 標準調試步驟
1. **檢查最新日誌**：`logs/` 目錄下的錯誤訊息
2. **確認配置**：比較 `cfgs/snakemake/dev.yaml` 和 `cfgs/snakemake/default.yaml`
3. **驗證數據**：檢查輸入檔案格式和大小
4. **測試環境**：確認 conda 環境和依賴

## 📝 問題記錄流程

### 遇到新問題時：
1. **記錄錯誤**：完整的錯誤訊息和堆疊追蹤
2. **分析根因**：找出問題的根本原因
3. **實施修正**：編寫和測試解決方案
4. **更新文檔**：將問題和解決方案記錄到 `docs/technical/known_errors.md`

### 文檔更新檢查清單：
- [ ] 在 `known_errors.md` 中添加新錯誤條目
- [ ] 記錄根本原因和解決方案
- [ ] 更新 `project_status.md` 中的進度
- [ ] 如有必要，創建新的 ADR 記錄

## 🛠️ 具體問題解決模式

### 路徑/配置問題模式
```bash
# 1. 檢查配置文件
vim cfgs/snakemake/dev.yaml

# 2. 比較 Snakemake 規則
grep -n "save_dir" Snakefile

# 3. 驗證路徑計算
snakemake --dry-run target_rule

# 4. 測試修正
snakemake --configfile cfgs/snakemake/dev.yaml target_rule
```

### 模型/數據不匹配問題模式
```bash
# 1. 檢查數據格式
python -c "import h5py; print(h5py.File('data.h5')['data/features'].shape)"

# 2. 檢查模型配置
grep -A 10 "TrafficXLSTMConfig" src/social_xlstm/models/xlstm.py

# 3. 添加調試輸出
# 在 common.py 中添加 logger.info(f"Input shape: {input_shape}")

# 4. 逐步測試
python scripts/train/without_social_pooling/train_single_vd.py --epochs 1
```

## 📚 2025-07-13 修正案例回顧

### 案例1：Double Nested Directory
**問題**：`blob/experiments/dev/xlstm/single_vd/single_vd/`
**根因**：Snakemake 路徑計算 `$(dirname {output})` + `experiment_name`
**解決**：修正為 `$(dirname $(dirname {output}))`
**教訓**：路徑計算需要考慮完整的目錄結構

### 案例2：xLSTM Multi-VD 輸入不匹配
**問題**：`Expected input_size=5, got 15`
**根因**：4D 數據格式未正確處理，特徵檢測邏輯錯誤
**解決**：
1. 修正 `create_data_module` 中的 VD 選擇邏輯
2. 修正 `create_model_for_multi_vd` 中的特徵計算
3. 在 TrafficXLSTM 中添加 4D 輸入處理

**教訓**：新模型需要與現有數據管線完全兼容

## 🎯 預防措施

### 開發最佳實踐
1. **小步迭代**：每次只修改一個組件
2. **測試驅動**：修改後立即測試
3. **文檔同步**：修正後立即更新文檔
4. **回歸測試**：確保修正不破壞現有功能

### 配置管理
1. **同步更新**：修改配置時同時更新 dev 和 default 版本
2. **路徑檢查**：新增路徑配置時驗證完整性
3. **參數驗證**：添加參數檢查和錯誤處理

### 測試策略
1. **端到端測試**：完整的訓練流程測試
2. **單元測試**：關鍵組件的獨立測試
3. **配置測試**：不同配置組合的測試

## 📞 支援資源

- **錯誤記錄**：`docs/technical/known_errors.md`
- **設計問題**：`docs/technical/design_issues_refactoring.md`
- **專案狀態**：`docs/reports/project_status.md`
- **ADR 決策**：`docs/adr/`

記住：每個解決的問題都是專案知識庫的一部分，良好的文檔化有助於避免重複問題。