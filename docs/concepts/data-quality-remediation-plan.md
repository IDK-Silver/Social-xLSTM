# Social-xLSTM 數據品質修復計劃

## 🚨 問題診斷

基於專家模型共識分析，當前 Social-xLSTM 面臨嚴重數據品質問題：

- **60% 特徵完全是 NaN** (Features 0, 2, 3)
- **20% 特徵是常數** (Feature 4 = 2)
- **僅 20% 特徵有效** (Feature 1, 範圍 0-56)
- **導致訓練完全失敗** (R² = -1000.6)

## 🎯 修復策略：三階段實施

### Phase 1: 緊急修復 (1週) - CRITICAL
**目標**: 立即恢復模型訓練能力

**實施步驟**:
1. **創建數據清理模組** (`src/social_xlstm/utils/data_cleaner.py`)
   ```python
   def emergency_cleanup(data_path):
       # 自動移除全 NaN 和常數特徵
       # 基礎正規化處理
       # 返回清理後數據
   ```

2. **整合到訓練腳本** (`scripts/train/with_social_pooling/train_distributed_social_xlstm.py`)
   - 在數據載入後立即調用清理函數
   - 添加基礎 assertions
   - 記錄清理統計

3. **驗證修復效果**
   - 運行訓練確認 R² 恢復正常範圍
   - 確保指標計算有意義

**成功標準**: R² > -10, MAE < 50

### Phase 2: 正規管線建設 (2-3週) - HIGH
**目標**: 建立生產級數據品質管線

**核心組件**:

#### 2.1 數據驗證層
```python
# src/social_xlstm/data/validators.py
class DataQualityValidator:
    def validate_schema(self, df):
        # 檢查必要欄位存在
        
    def validate_quality(self, df):
        # NaN 比例檢查
        # 常數特徵檢查
        # 數值範圍檢查
        
    def validate_distribution(self, df):
        # 統計分佈檢查
        # 異常值檢測
```

#### 2.2 特徵工程管線
```python
# src/social_xlstm/data/preprocessing.py
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline():
    return Pipeline([
        ('validator', DataQualityValidator()),
        ('cleaner', DataCleaner()),
        ('imputer', AdvancedImputer()),
        ('scaler', RobustScaler()),
        ('feature_selector', VarianceThreshold()),
        ('final_check', FinalAssertion())
    ])
```

#### 2.3 防禦性編程
```python
# 關鍵檢查點
assert not torch.isnan(input_tensor).any(), "NaN detected in model input"
assert input_tensor.shape[1] == expected_features, "Feature count mismatch"
assert torch.isfinite(input_tensor).all(), "Infinite values detected"
```

### Phase 3: 監控與自動化 (1週) - MEDIUM
**目標**: 持續監控數據品質，防止問題復發

#### 3.1 數據品質儀表板
- 特徵分佈可視化
- NaN 趨勢監控
- 數據漂移檢測

#### 3.2 自動化警報
```python
# 品質閾值監控
quality_thresholds = {
    'max_nan_percentage': 0.1,
    'min_feature_variance': 1e-5,
    'max_constant_features': 1
}
```

#### 3.3 MLflow 整合
- 數據品質指標記錄
- 管線版本控制
- 自動化報告生成

## 🛠️ 技術堆疊

**核心工具**:
- **驗證**: Great Expectations / Pandera
- **處理**: scikit-learn Pipeline
- **監控**: MLflow + Custom Dashboard
- **自動化**: 整合到現有 Snakemake 流程

**整合點**:
```python
# scripts/train/with_social_pooling/train_distributed_social_xlstm.py 修改
def setup_data_pipeline(args):
    # 建立數據管線
    pipeline = create_preprocessing_pipeline()
    
    # 數據載入與驗證
    raw_data = load_data(args.data_path)
    clean_data = pipeline.fit_transform(raw_data)
    
    # 品質檢查
    validate_final_data(clean_data)
    
    return clean_data
```

## 📊 成功指標

### 技術指標
- [ ] 訓練 R² > 0.5
- [ ] NaN 檢測覆蓋率 100%
- [ ] 自動化測試通過率 > 95%
- [ ] 管線執行時間 < 5分鐘

### 業務指標
- [ ] 模型訓練成功率 100%
- [ ] 實驗重現性 100%
- [ ] 數據問題檢測時間 < 1小時

## ⏰ 實施時程

| 階段 | 時程 | 負責任務 | 里程碑 |
|------|------|----------|---------|
| Phase 1 | Week 1 | 緊急數據清理 | 恢復模型訓練 |
| Phase 2 | Week 2-4 | 正規管線建設 | 生產級品質保證 |
| Phase 3 | Week 5 | 監控自動化 | 持續品質監控 |

## 🔄 維護計劃

### 日常維護
- 每日數據品質報告
- 週週管線健康檢查
- 月度品質指標回顧

### 持續改進
- 季度管線優化
- 新特徵驗證規則更新
- 監控閾值調整

## 📚 參考資源

- [Great Expectations 文檔](https://docs.greatexpectations.io/)
- [MLOps 最佳實踐](https://ml-ops.org/)
- [數據品質框架](https://github.com/great-expectations/great_expectations)

---

**注意**: 此計劃基於三個頂級 AI 模型 (OpenAI O3-Pro, Google Gemini 2.5 Pro, Claude Opus 4) 的專家共識制定，平均信心度 8.7/10。