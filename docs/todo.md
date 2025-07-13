# Social-xLSTM 待辦事項清單

最後更新：2025-07-13

## 🚨 緊急任務（本週 P0）

### 1. ✅ LSTM 實現統一 - 已完成
- [x] 審核 ADR-0002 並決定是否執行
- [x] 執行 LSTM 統一計劃：
  - [x] 階段 1：功能分析與精華提取（1天）
  - [x] 階段 2：標準實現建立（2天）
  - [x] 階段 3：遷移與清理（2天）
  - [x] 階段 4：測試與驗證（1天）
- [x] 統一的 TrafficLSTM 實現 (src/social_xlstm/models/lstm.py)
- [x] 支援單VD和多VD模式
- [x] 完整的訓練系統整合

### 2. Social Pooling 核心實現 🎯
- [ ] 實現座標驅動社交池化算法
- [ ] 設計網格劃分機制
- [ ] 整合現有 CoordinateSystem
- [ ] 建立 Social Pooling 單元測試

### 3. xLSTM 基礎架構建立 🏗️
- [ ] 創建獨立的 TrafficXLSTM 類 (src/social_xlstm/models/xlstm.py)
- [ ] 實現 TrafficXLSTMConfig dataclass 配置系統
- [ ] 整合 xlstm 庫的 xLSTMBlockStack
- [ ] 實現基本的前向傳播和訓練接口
- [ ] 建立 xLSTM 單元測試框架

## 📋 重要任務（下週 P1）

### 4. xLSTM 核心功能實現
- [ ] **階段1：基礎 xLSTM 架構**（根據 ADR-0501）
  - [ ] 實現 TrafficXLSTM 類的完整功能
  - [ ] 配置 sLSTM blocks（位置：[1, 3]）
  - [ ] 配置 mLSTM blocks（其餘位置）
  - [ ] 使用 vanilla backend（穩定性優先）
- [ ] **階段2：訓練系統整合**
  - [ ] 創建 xLSTM 專用訓練腳本
  - [ ] 修改 Snakefile 支援 xLSTM 訓練規則
  - [ ] 實現 LSTM vs xLSTM 並行訓練
  - [ ] 配置文件分離（training_xlstm 區塊）
- [ ] **階段3：混合架構設計**
  - [ ] sLSTM 處理時間序列（單變量）
  - [ ] mLSTM 處理空間特徵（多變量）
  - [ ] 設計 Social Pooling 整合點

### 5. ✅ Dataset 模組重構 - 已完成
- [x] 重構 Dataset 模組為結構化子包
- [x] 從 2 個大文件（430+499行）分離為專門化文件
- [x] 清晰的關注點分離：config/, core/, storage/, utils/
- [x] 正規化器共享問題修復
- [x] 71/72 測試通過，向後兼容性確保
- [x] 冗餘代碼清理

### 6. 專案結構重組（持續中）
- [ ] 移除 sanbox/ 重複代碼
- [ ] 清理空文件
- [ ] 重組目錄結構為：
  ```
  src/social_xlstm/
  ├── models/
  │   ├── lstm/
  │   ├── xlstm/
  │   └── base/
  ├── configs/
  │   ├── models/
  │   └── training/
  └── trainers/
  ```

### 7. ✅ 訓練流程建立 - 已完成
- [x] 實現統一的訓練器介面 (src/social_xlstm/training/trainer.py)
- [x] 整合評估指標 (src/social_xlstm/evaluation/evaluator.py)
- [x] 建立實驗管理系統
- [x] 訓練腳本重構（ADR-0400 實施完成）
- [x] **完整的多模式訓練系統** (2025-07-13 新增)
  - [x] Single VD 訓練流程
  - [x] Multi-VD 訓練流程  
  - [x] Independent Multi-VD 訓練流程
- [x] **完整的視覺化與評估系統** (2025-07-13 新增)
  - [x] 自動化圖表生成 (Snakemake workflow)
  - [x] 訓練指標說明文檔
  - [x] 完整的實驗報告生成

## 🔄 持續任務（第三週 P2）

### 8. 模型訓練與評估
- [x] **LSTM 基準系統完整建立** (2025-07-13 完成)
  - [x] 三種 LSTM 訓練模式驗證
  - [x] 完整的性能分析和視覺化
  - [x] 結果記錄和分析文檔
- [ ] **xLSTM vs LSTM 基準比較**
  - [ ] 並行訓練框架設置
  - [ ] 性能指標對比（MAE, R², 過擬合程度）
  - [ ] 記憶體使用和訓練時間分析
  - [ ] 結果可視化和報告生成
- [ ] **xLSTM 超參數調優**
  - [ ] num_blocks 調整（4, 6, 8）
  - [ ] sLSTM 位置優化
  - [ ] embedding_dim 調整
  - [ ] dropout 率優化
- [ ] **Social-xLSTM 完整評估**
  - [ ] Social Pooling + xLSTM 整合效果
  - [ ] 空間相關性改善程度
  - [ ] 長期預測能力評估

### 9. 測試覆蓋率提升
- [ ] 單元測試覆蓋率 > 70%
- [ ] 整合測試
- [ ] 性能基準測試

## 📝 文檔任務（第四週 P3）

### 10. 期末報告準備
- [ ] 實驗結果整理
- [ ] 技術文檔更新
- [ ] 論文撰寫準備
- [ ] 程式碼最終清理

## 🔍 已完成任務 ✅

### 2025-01-08
- [x] 專案狀態分析與整理
- [x] 期中報告分析（使用 MCP markdownify）
- [x] 文檔結構重組（移除數字前綴）
- [x] 數學公式定義（LaTeX）
- [x] ADR 系統建立
- [x] 座標處理系統分析（發現已有完整實現）
- [x] LSTM 實現分析和統一策略設計
- [x] ADR-0002 創建（LSTM 統一方案）

### 2025-01-09
- [x] Dataset 模組重構（ADR-0002 LSTM 統一方案實施）
- [x] 從 2 個大文件（430+499行）分離為結構化子包
- [x] 建立清晰的關注點分離：config/, core/, storage/, utils/
- [x] 修復正規化器共享問題
- [x] 71/72 測試通過，確保向後兼容性
- [x] 冗餘代碼清理和文檔更新
- [x] 訓練腳本重構（ADR-0400 實施完成）

### 2025-07-13
- [x] **完整的多模式訓練系統建立**
  - [x] Single VD 訓練 (226,309 參數)
  - [x] Multi-VD 訓練 (1,433,537 參數)
  - [x] Independent Multi-VD 訓練 (1,421,317 參數)
- [x] **Snakemake 工作流程完善**
  - [x] 修復輸出衝突問題
  - [x] 新增多VD圖表生成規則
  - [x] 完整的自動化訓練與視覺化管線
- [x] **訓練指標理解文檔**
  - [x] understanding_training_metrics.md 創建
  - [x] 6大核心指標詳解（Loss, MAE, MSE, RMSE, MAPE, R²）
  - [x] 圖表解讀指南與問題診斷

## 📊 關鍵決策待定

1. ✅ **ADR-0002 批准**：LSTM 統一計劃已執行完成
2. ✅ **並行開發策略**：架構清理與核心功能並行進行
3. **配置系統**：YAML 配置系統的具體實現細節
4. **下一步核心功能**：根據 ADR-0100 和 ADR-0101，開始 Social Pooling 和 xLSTM 實現

## 🎯 成功指標

### 第一週結束（本週）
- [x] ✅ Dataset 模組重構完成（2025-01-09）
- [x] ✅ LSTM 統一實現完成（2025-01-09）
- [x] ✅ 訓練系統重構完成（2025-01-09）
- [ ] 社交池化演算法原型實現
- [ ] **xLSTM 基礎架構完成**
  - [ ] TrafficXLSTM 類創建並可初始化
  - [ ] 基本的 xLSTM 訓練腳本運行
  - [ ] 單元測試框架建立

### 第二週結束
- [ ] **完整的 xLSTM 實現**（根據 ADR-0501）
  - [ ] xLSTM 能在開發數據集上訓練
  - [ ] LSTM vs xLSTM 並行訓練完成
  - [ ] 初步性能比較結果
- [ ] **Social Pooling 與 xLSTM 整合設計**
  - [ ] 整合架構設計完成
  - [ ] 技術可行性驗證
- [ ] 專案結構重組完成

### 第三週結束
- [ ] Social-xLSTM vs LSTM 基準比較完成
- [ ] 測試覆蓋率 > 70%
- [ ] 初步實驗結果可用

### 第四週結束
- [ ] 期末報告完成
- [ ] 程式碼品質達到發布標準
- [ ] 所有實驗結果文檔化

## 💡 注意事項

1. **優先級調整原則**：如遇技術困難，優先保證核心功能（Social-xLSTM）實現
2. **程式碼品質**：每次提交前確保通過 linting 和基本測試
3. **文檔同步**：重要變更需同步更新 ADR 和相關文檔
4. **進度回顧**：每週五進行進度回顧和優先級調整

## 📐 xLSTM 實施技術細節

### 架構決策（根據 ADR-0501）
- **獨立類別**：創建 TrafficXLSTM，不擴展 TrafficLSTM
- **配置分離**：使用 TrafficXLSTMConfig dataclass
- **並行開發**：保持 LSTM 和 xLSTM 可獨立運行

### 技術實施重點
1. **xLSTM 區塊配置**
   - 總共 6 個區塊（num_blocks=6）
   - sLSTM 在位置 [1, 3]（處理時間序列）
   - mLSTM 在其他位置（處理空間特徵）

2. **後端選擇**
   - 初期使用 vanilla backend（穩定性優先）
   - 暫不使用 mlstm_kernels（避免依賴複雜性）

3. **性能目標**
   - 改善過擬合問題（目前 LSTM R² 為負值）
   - 訓練/驗證指標差距 < 2倍
   - 記憶體使用在可接受範圍內

## 🔗 相關文檔

- [專案概述](overview/project_overview.md)
- [ADR 列表](adr/README.md)
- [專案變更記錄](reports/project_changelog.md)
- [關鍵決策記錄](overview/key_decisions.md)

---

**提醒**：此文檔應每日更新，反映最新的專案狀態和優先級變化。