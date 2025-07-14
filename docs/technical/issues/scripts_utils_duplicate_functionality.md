# Scripts/Utils 重複功能問題分析

**日期**: 2025-07-14
**狀態**: 已識別，待處理
**優先級**: 中等

## 問題描述

在 `scripts/utils/` 目錄中發現多個腳本存在重複功能，特別是在實驗處理和訓練歷史數據處理方面。

## 重複功能分析

### 🔴 主要重複：訓練歷史處理邏輯

**涉及腳本**:
- `generate_training_plots.py` (148 行)
- `generate_training_report.py` (254 行)

**重複的功能**:
1. **文件加載**: 兩個腳本都從 JSON 文件加載 `training_history.json`
2. **數據處理**: 都解析實驗元數據、訓練指標和配置
3. **錯誤處理**: 相似的錯誤處理模式
4. **目錄結構**: 都期望相同的實驗目錄結構

**估計重複代碼量**: ~80 行

### 🟡 中度重複：實驗目錄處理

**涉及腳本**:
- `generate_training_plots.py`
- `generate_training_report.py`
- `run_all_plots.py` (部分)

**重複的功能**:
- 實驗目錄驗證邏輯
- 路徑處理模式
- 參數解析模式

### 🟢 輕微重複：工具函數

**涉及腳本**: 多個腳本
**重複的功能**:
- 參數解析模式
- 日誌設置
- 路徑驗證邏輯

## 各腳本功能總結

| 腳本 | 主要功能 | 獨特性 | 建議 |
|------|----------|--------|------|
| `claude_init.py` | Claude 初始化助手 | 高 | 保持獨立 |
| `generate_training_plots.py` | 生成訓練可視化圖表 | 中 | 需要整合 |
| `generate_training_report.py` | 生成訓練報告 | 中 | 需要整合 |
| `plot_vd_point.py` | VD 座標繪圖 | 高 | 保持獨立 |
| `quality_check.py` | 數據品質分析 | 高 | 保持獨立 |
| `run_all_plots.py` | 並行執行繪圖規則 | 高 | 保持獨立 |

## 建議的整合方案

### 高優先級：創建共享實驗處理庫

**目標**: 消除 `generate_training_plots.py` 和 `generate_training_report.py` 之間的重複

**建議結構**:
```python
# src/social_xlstm/experiment/processor.py
class ExperimentProcessor:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.history = self.load_training_history()
    
    def load_training_history(self) -> Dict:
        """通用訓練歷史加載邏輯"""
        
    def validate_experiment_dir(self) -> bool:
        """通用驗證邏輯"""
        
    def get_experiment_metadata(self) -> Dict:
        """通用元數據提取"""
```

**預期好處**:
- 消除 80+ 行重複代碼
- 集中實驗處理邏輯
- 改善維護性
- 確保一致的錯誤處理

### 中優先級：統一參數解析

**目標**: 減少參數解析的重複

**建議結構**:
```python
# scripts/utils/common_args.py
class ExperimentArgParser:
    def get_base_parser(self) -> argparse.ArgumentParser:
        """實驗腳本的通用參數解析"""
```

## 實施計畫

### 階段 1：創建共享實驗處理庫
1. 創建 `src/social_xlstm/experiment/processor.py`
2. 提取通用加載/驗證邏輯
3. 更新 `generate_training_plots.py` 使用共享處理器
4. 更新 `generate_training_report.py` 使用共享處理器

### 階段 2：重構參數解析
1. 創建 `scripts/utils/common_args.py`
2. 更新相關腳本使用通用參數解析

### 階段 3：測試和驗證
1. 確保所有腳本維持相同功能
2. 測試現有實驗目錄
3. 驗證錯誤處理一致性

## 影響評估

**風險**: 低
- 整合專注於工具函數，不改變核心功能
- 腳本保持其主要目的
- 不破壞現有接口

**好處**:
- 減少約 25% 的代碼重複
- 改善維護性
- 集中實驗處理邏輯
- 確保一致的錯誤處理

## 相關問題

- 與 #issue-training-scripts-consolidation 相關
- 可能影響實驗工作流程的一致性

## 下一步行動

1. 評估和批准整合方案
2. 實施階段 1：共享實驗處理庫
3. 更新相關腳本
4. 測試和驗證功能

---
**記錄者**: Claude Code
**最後更新**: 2025-07-14