# 下一輪 Session 待解決任務

## 📋 任務概述
基於完整的模組文檔分析和 Dataset 重構完成狀態，以下是需要下一輪 session 解決的關鍵問題和改進任務。

## ✅ 已完成任務 (2025-01-09)

### 1. Dataset 模組重構 ✅
**狀態**: 已完成
- **loader.py**: 已分拆為多個專門化文件
- **h5_utils.py**: 已分離為 h5_converter.py 和 h5_reader.py
- **新結構**: 已實現為結構化子包

### 2. LSTM 基準測試系統 ✅
**狀態**: 已完成 (2025-01-09)
- **腳本**: `scripts/baseline_test.py` 
- **功能**: 單VD/多VD/多步驟預測基準測試
- **評估指標**: MAE, MSE, RMSE, MAPE, R²
- **目的**: 為未來 Social-xLSTM 性能比較建立基準

**使用方式**:
```bash
# 快速測試
python scripts/baseline_test.py --quick --test_type single_vd

# 完整基準測試
python scripts/baseline_test.py --test_type all
```

### 3. 語言處理指南 ✅
**狀態**: 已完成 (2025-01-09)
- **位置**: `CLAUDE.md` 中新增語言處理指南
- **目的**: 中文輸入 → 英文技術思考 → 中文結果回報
- **效果**: 提升技術準確性的同時保持中文溝通體驗

**實際實現結構**:
```
src/social_xlstm/dataset/
├── config/
│   └── base.py          # TrafficDatasetConfig, TrafficHDF5Config
├── core/
│   ├── processor.py     # TrafficDataProcessor
│   ├── timeseries.py    # TrafficTimeSeries
│   └── datamodule.py    # TrafficDataModule
├── storage/
│   ├── h5_converter.py  # TrafficHDF5Converter
│   ├── h5_reader.py     # TrafficHDF5Reader
│   └── feature.py       # TrafficFeature
└── utils/
    ├── json_utils.py    # VDInfo, VDLiveList
    ├── xml_utils.py     # XML 處理
    └── zip_utils.py     # 壓縮檔案處理
```

### 4. 正規化器共享問題 ✅
**狀態**: 已修復
- **問題**: 測試集正規化器處理不當
- **解決**: 實現正規化器的正確共享機制
- **實現**: 在 TrafficTimeSeries 中加入 scaler 參數，在 TrafficDataModule 中正確傳遞

### 5. 代碼重複清理 ✅
**狀態**: 已完成
- **清理的檔案**: 
  - 移除 `loader.py` 和 `h5_utils.py` (已重構)
  - 移除 `utils/pure_text.py` (未使用且功能簡單)
- **測試驗證**: 71/72 測試通過，向後兼容性確保

## 🔥 當前優先級任務 (P0)

基於 ADR-0100 和 ADR-0101 決策，Dataset 重構已完成，現在應專注於核心功能開發：

### 0. 緊急修復問題 🚨

#### ❌ 多VD訓練錯誤
**狀態**: 需要修復
**問題**: 發現了 docs/technical/known_errors.md 中記錄的張量形狀不匹配問題
- **錯誤**: RuntimeError: The size of tensor a (15) must match the size of tensor b (5)
- **原因**: 模型輸出 [4, 1, 15] vs 目標 [4, 1, 3, 5] 形狀不匹配
- **位置**: Multi-VD 訓練時的損失函數計算

#### ❌ 訓練歷史記錄缺失  
**狀態**: 需要修復
**問題**: 訓練完成後缺少重要的記錄文件
- **缺少檔案**: 
  - `best_model.pt` - 最佳模型權重 (日誌說有保存但實際沒有)
  - `training_history.json` - 訓練歷史記錄
  - `test_evaluation.json` - 測試評估結果
- **根因**: 
  - 沒有驗證數據 (VAL dataset: 0 samples)，無法觸發 `save_best_only`
  - `checkpoint_interval=10` 但只訓練 2 epochs，未觸發定期保存
  - 訓練結束後沒有強制保存最終模型

**修復優先級**: P0 (緊急) - 影響訓練系統基本功能

### 1. Social Pooling 算法實現 (ADR-0100)
**狀態**: 待實施
**目標**: 實現座標驅動的社交池化算法
**技術基礎**: ✅ 座標系統 (spatial_coords.py) 已完成，✅ 統一 LSTM 已完成

**實施步驟**:
- 實現座標驅動社交池化算法
- 設計網格劃分機制
- 整合現有 CoordinateSystem
- 建立 Social Pooling 單元測試

### 2. xLSTM 整合 (ADR-0101)
**狀態**: 待實施
**目標**: 整合真正的 xLSTM 實現（sLSTM + mLSTM）
**當前**: 使用標準 LSTM 作為占位符

## 🚀 高優先級任務 (P1)

### 3. Social-xLSTM 模型整合
**狀態**: 技術決策已完成，待實施
**目標**: 整合真正的 xLSTM 實現（sLSTM + mLSTM）
**當前**: 使用標準 LSTM 作為占位符

**實施步驟**:
```python
# 當前占位符
class VDxLSTM(nn.Module):
    def __init__(self, config):
        self.lstm = nn.LSTM(...)  # 標準 LSTM

# 目標實現
class VDxLSTM(nn.Module):
    def __init__(self, config):
        from xlstm import sLSTM, mLSTM  # 真正的 xLSTM
        self.temporal_encoder = sLSTM(...)
        self.spatial_encoder = mLSTM(...)
```

### 5. 性能優化
**問題**: 座標轉換效率低下
- 每次前向傳播重複計算座標轉換
- 社交池化中的迴圈處理效率低

**解決方案**:
```python
# 預計算相對座標矩陣
class SocialPoolingLayer(nn.Module):
    def __init__(self, config, vd_coordinates):
        self.relative_coords_cache = self._precompute_relative_coords(vd_coordinates)
        
    def forward(self, ...):
        # 使用預計算的座標，避免重複計算
        relative_coords = self.relative_coords_cache[batch_indices]
```

## 🛠️ 中等優先級任務 (P2)

### 6. 測試覆蓋率提升
**現狀**: < 30%
**目標**: > 70%
**重點**: 核心模型和數據處理邏輯

### 7. 錯誤處理改進
**問題**: 部分函數缺乏完整的錯誤處理
**行動**: 添加詳細的異常處理和日誌記錄

### 8. 文檔完善
**需要**: 更多使用示例和參數說明
**重點**: 複雜函數的詳細文檔

## 🎯 長期改進任務 (P3)

### 9. 架構進一步優化
- 向量化操作替代迴圈處理
- 批次處理效率提升
- 記憶體使用優化

### 10. 功能擴展
- 更多的池化方法
- 更靈活的配置選項
- 更豐富的評估指標

## 📁 相關文件位置

### 需要開發的文件
- `src/social_xlstm/models/social_pooling.py` (需實現 Social Pooling 算法)
- `src/social_xlstm/models/social_xlstm.py` (需 xLSTM 整合)

### 已完成的新增檔案
- ✅ `scripts/baseline_test.py` (LSTM 基準測試腳本)
- ✅ `CLAUDE.md` (語言處理指南更新)

### 已完成的重構
- ✅ `src/social_xlstm/dataset/loader.py` (已分拆為結構化子包)
- ✅ `src/social_xlstm/dataset/h5_utils.py` (已分拆為 h5_converter.py 和 h5_reader.py)
- ✅ `src/social_xlstm/utils/pure_text.py` (已刪除)

### 參考文檔
- `docs/modules/dataset_documentation.md` - Dataset 模組詳細分析
- `docs/modules/models_documentation.md` - Models 模組詳細分析
- `docs/modules/comprehensive_documentation.md` - 完整模組分析
- `docs/adr/0101-xlstm-vs-traditional-lstm.md` - xLSTM 技術決策
- `docs/adr/0100-social-pooling-vs-graph-networks.md` - Social Pooling 技術決策

## 🎯 成功指標

### 第一週目標
- [x] ✅ Dataset 模組重構完成
- [x] ✅ 正規化器共享問題解決
- [x] ✅ 重複代碼清理完成
- [x] ✅ LSTM 基準測試系統完成
- [x] ✅ 語言處理指南完成

### 第二週目標
- [ ] Social Pooling 算法實現完成
- [ ] xLSTM 整合完成
- [ ] 測試覆蓋率提升到 50%+

### 第三週目標
- [ ] 測試覆蓋率達到 70%+
- [ ] 文檔完善
- [ ] 性能基準測試

## 💡 實施建議

### 開發策略
1. **先重構，後優化**: 先解決架構問題，再進行性能優化
2. **增量測試**: 每次修改後立即測試，確保功能不會受損
3. **保持向後兼容**: 重構時保持現有 API 的兼容性

### 風險緩解
1. **備份重要文件**: 重構前備份原始實現
2. **分階段實施**: 避免一次性大幅修改
3. **持續測試**: 每個階段都進行充分測試

## 📞 技術支援

### 關鍵技術決策
- ADR-0100: Social Pooling vs Graph Networks (已批准)
- ADR-0101: xLSTM vs Traditional LSTM (已批准)
- ADR-0002: LSTM 實現統一方案 (已實施)

### 技術基礎
- ✅ 座標系統: `src/social_xlstm/utils/spatial_coords.py` (完善)
- ✅ 統一 LSTM: `src/social_xlstm/models/lstm.py` (完成)
- ✅ 評估框架: `src/social_xlstm/evaluation/evaluator.py` (基本完成)
- ✅ 統一訓練系統: `src/social_xlstm/training/trainer.py` (完成)
- ✅ Dataset 結構化子包: `src/social_xlstm/dataset/` (完成)

---

## 🎯 當前 Session 完成摘要 (2025-07-12)

### 配置系統重組 ✅
1. **配置結構重組**: 移動配置文件到 cfgs/snakemake/ 層次結構
2. **自動化路徑管理**: 解決手動資料夾建立問題，實現自動路徑創建
3. **開發/生產分離**: default.yaml (生產) 與 dev.yaml (開發) 配置分離
4. **文檔完善**: 創建 cfgs/README.md 說明配置使用方式

### 發現的關鍵問題 🚨
1. **多VD訓練錯誤**: 張量形狀不匹配 (docs/technical/known_errors.md 中已記錄)
2. **模型保存機制缺陷**: 
   - 無驗證數據時不保存最佳模型
   - checkpoint_interval 設定過大
   - 缺少訓練結束強制保存
3. **訓練歷史記錄缺失**: 缺少 training_history.json 和 test_evaluation.json

### 配置結構
```
cfgs/
├── README.md           # 配置說明文檔
└── snakemake/         # Snakemake 工作流程配置
    ├── default.yaml   # 預設/生產環境配置
    └── dev.yaml       # 開發環境配置
```

---

## 🎯 之前 Session 完成摘要 (2025-01-09)

### 新增功能
1. **LSTM 基準測試系統**: 完整的無 Social 機制 LSTM 基準測試
2. **語言處理優化**: 中文輸入 → 英文技術思考 → 中文結果回報的工作流程

### 為 Social-xLSTM 開發做好準備
- ✅ 基準測試系統就緒，可進行性能比較
- ✅ 語言處理指南優化技術討論效率
- ✅ 所有技術基礎已完成，可開始核心算法開發

### 下一步重點
根據 ADR-0100 和 ADR-0101 決策，下一階段應：
1. **P0**: 實現 Social Pooling 算法
2. **P1**: 整合 xLSTM 架構
3. **P2**: 完成 Social-xLSTM 模型

---

**備註**: 本任務清單基於 2025-01-09 的完整模組分析和當前 session 完成內容，請在下一輪 session 開始時參考此文檔進行工作規劃。