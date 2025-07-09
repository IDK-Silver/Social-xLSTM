# ADR-0002: LSTM 實現統一與標準化

**狀態**: Partially Implemented  
**日期**: 2025-01-08  
**更新**: 2025-01-09  
**決策者**: 專案團隊  
**技術故事**: 解決專案中 5 個重複 LSTM 實現的問題

## 背景與問題陳述

專案健康狀況分析發現嚴重的 LSTM 實現重複問題：

### 現有實現清單
1. `src/social_xlstm/models/traffic_lstm.py` (284行) - 完整但冗長的實現
2. `src/social_xlstm/models/pure/traffic_lstm.py` (88行) - 精簡專業的實現
3. `src/social_xlstm/models/traffic_xlstm.py` (24行) - 簡化的 xLSTM 包裝器
4. `sanbox/train/loader_lstm.py` (461行) - 自包含的 LSTM 實現
5. `sanbox/train/loader_xlstm.py` (498行) - 問題較多的 xLSTM 實現

### 問題影響
- **維護困難**：修改 LSTM 邏輯需要同步多個文件
- **程式碼重複**：約 1,200+ 行重複邏輯
- **品質不一**：不同實現的品質差異極大
- **新人困惑**：不知道該使用哪個實現
- **測試複雜**：需要為多個相似實現寫測試

## 決策驅動因素

- **可維護性**：減少重複代碼，提高維護效率
- **程式碼品質**：保留最佳實現，移除有問題的代碼
- **學術嚴謹性**：統一的高品質實現有助於論文發表
- **開發效率**：明確的標準減少決策時間
- **Social-xLSTM 基礎**：為核心創新提供穩固的 LSTM 基線

## 考慮的選項

### 選項 1: 保持現狀，不進行統一

- **優點**:
  - 不需要額外開發工作
  - 保留所有現有功能
  - 不會破壞現有引用
- **缺點**:
  - 維護成本持續增加
  - 新功能開發困惑
  - 技術債務累積
  - 影響專案專業形象

### 選項 2: 完全重寫，從零開始

- **優點**:
  - 完全符合當前需求
  - 架構最優化
  - 沒有歷史包袱
- **缺點**:
  - 開發時間長（估計 1-2 週）
  - 可能遺失現有功能
  - 引入新 bug 風險高
  - 延誤核心研究進度

### 選項 3: 基於最佳實現進行統一

- **優點**:
  - 保留最高品質的實現
  - 整合其他實現的精華
  - 開發時間可控（3-5 天）
  - 風險相對較低
- **缺點**:
  - 需要仔細分析每個實現
  - 遷移工作量不小
  - 可能需要調整現有引用

### 選項 4: 漸進式重構，逐步統一

- **優點**:
  - 風險最低
  - 可以逐步驗證
  - 不影響當前開發
- **缺點**:
  - 耗時最長
  - 在相當長時間內仍有重複代碼
  - 可能缺乏推進動力

## 決策結果

**選擇**: 選項 3 - 基於最佳實現進行統一

**理由**: 
1. **時間效率**：3-5 天的投入可以解決長期維護問題
2. **品質保證**：基於已驗證的最佳實現
3. **風險可控**：保留精華，移除冗餘
4. **戰略價值**：為 Social-xLSTM 實現提供穩固基礎

## 實施細節

### 階段 1: 分析與保存 (1天)

#### 1.1 詳細功能分析
```bash
# 分析每個實現的獨特功能
- traffic_lstm.py: 可視化功能、詳細注釋
- pure/traffic_lstm.py: 專業架構、多步預測
- loader_lstm.py: 批次處理邏輯
- pure_lstm.py: 現代化訓練配置
- loader_xlstm.py: xLSTM 整合嘗試（問題較多）
```

#### 1.2 精華功能提取
```python
# 提取有價值的組件
valuable_components = {
    'visualization': 'from traffic_lstm.py',
    'batch_processing': 'from loader_lstm.py', 
    'training_config': 'from pure_lstm.py',
    'model_architecture': 'from pure/traffic_lstm.py',
    'evaluation_metrics': 'from multiple sources'
}
```

### 階段 2: 標準實現建立 (2天)

#### 2.1 模型命名策略與架構設計

**新的命名策略**（移除 "Standard" 前綴）：
```python
# 基礎模型系列
class TrafficLSTM(nn.Module):
    """基礎交通 LSTM 模型 - 統一後的標準實現"""

class TrafficExtendLSTM(nn.Module):
    """交通 xLSTM 模型（Extended LSTM）"""

# 未來擴展 - Social Pooling 相關
class SocialTrafficLSTM(nn.Module):
    """Social Pooling 交通 LSTM 模型（核心創新）"""

class SocialTrafficExtendLSTM(nn.Module):
    """Social Pooling 交通 xLSTM 模型（核心創新 + xLSTM）"""
```

**統一後的基礎實現**：
```python
# 新的標準實現: src/social_xlstm/models/lstm/traffic_lstm.py
class TrafficLSTM(nn.Module):
    """
    基礎交通 LSTM 模型
    
    基於 pure/traffic_lstm.py 的專業架構
    整合其他實現的精華功能
    支援 YAML 配置驅動
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,
                 sequence_length: int = 12,
                 dropout: float = 0.2):
        super().__init__()
        
        # 基於 pure 實現的核心架構
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 增強的輸出層
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 保存配置
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'sequence_length': sequence_length
        }
    
    def forward(self, x):
        """標準前向傳播"""
        # 基於 pure 實現的邏輯
        
    def predict_multi_step(self, x, steps):
        """多步預測 - 保留 pure 實現的優勢"""
        
    def get_model_info(self):
        """模型資訊 - 方便調試和記錄"""
        return {
            'model_type': 'TrafficLSTM',
            'parameters': sum(p.numel() for p in self.parameters()),
            'config': self.config
        }
```

#### 2.2 標準訓練腳本設計

**統一訓練腳本** (`scripts/train/train_baseline.py`)：
```python
#!/usr/bin/env python3
"""
標準訓練腳本 - 支援 LSTM 和 xLSTM 模型
根據配置文件動態選擇模型類型
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', required=True, help='模型配置文件路徑')
    parser.add_argument('--train-config', required=True, help='訓練配置文件路徑')
    args = parser.parse_args()
    
    # 載入配置
    config_loader = ConfigLoader()
    model_config = config_loader.load_model_config(args.model_config)
    train_config = config_loader.load_training_config(args.train_config)
    
    # 根據配置創建模型
    model = ModelFactory.create_model(model_config)
    
    # 標準訓練流程
    trainer = StandardTrainer(model, train_config)
    trainer.train()

# 使用範例:
# python scripts/train/train_baseline.py --model-config traffic_lstm --train-config baseline_training
# python scripts/train/train_baseline.py --model-config traffic_extend_lstm --train-config baseline_training
```

**Social Pooling 專用訓練腳本** (`scripts/train/train_social.py`)：
```python
#!/usr/bin/env python3
"""
Social Pooling 訓練腳本 - 專門處理 Social Pooling 相關模型
包含空間互動處理、座標系統等特殊功能
"""

def main():
    # Social Pooling 特殊的數據處理
    # 座標系統處理
    # 空間互動計算
    # 等專門功能
    pass

# 使用範例:
# python scripts/train/train_social.py --model-config social_traffic_lstm --train-config social_training
```

#### 2.3 簡化配置設計

**實際採用的簡化方案**：

基於實際需求，採用更簡潔的命令行配置而非複雜的 YAML 系統：

```python
# 實際統一實現: src/social_xlstm/models/lstm.py
class TrafficLSTM(nn.Module):
    """統一的交通 LSTM 模型"""
    
    @classmethod
    def create_single_vd_model(cls, input_size=3, hidden_size=128, **kwargs):
        """創建單VD模型"""
        return cls(input_size=input_size, hidden_size=hidden_size, 
                  multi_vd_mode=False, **kwargs)
    
    @classmethod
    def create_multi_vd_model(cls, num_vds=5, input_size=3, **kwargs):
        """創建多VD模型"""
        return cls(input_size=input_size, num_vds=num_vds, 
                  multi_vd_mode=True, **kwargs)
```

**統一訓練介面**：
```python
# 透過 scripts/train/common.py 提供統一配置
def create_base_training_config(args, mode='single_vd'):
    """創建基礎訓練配置"""
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # ... 其他配置
    )
```

### 階段 3: 遷移與清理 (2天)

#### 3.1 功能遷移驗證
```python
# 確保所有關鍵功能正常運作
test_cases = [
    'single_vd_training',      # 單一 VD 訓練
    'multi_vd_training',       # 多 VD 訓練  
    'multi_step_prediction',   # 多步預測
    'model_evaluation',        # 模型評估
    'result_visualization',    # 結果可視化
    'model_save_load',         # 模型儲存載入
]
```

#### 3.2 清理重複實現
```bash
# 備份和移除
mkdir -p backup/removed_lstm_implementations/
mv sanbox/train/loader_lstm.py backup/removed_lstm_implementations/
mv sanbox/train/loader_xlstm.py backup/removed_lstm_implementations/
mv src/social_xlstm/models/traffic_lstm.py backup/removed_lstm_implementations/

# 更新引用
# 搜索並更新所有對舊實現的引用
grep -r "from.*loader_lstm" . 
grep -r "from.*traffic_lstm" .
```

#### 3.3 文檔更新
```markdown
# 更新相關文檔
- docs/implementation/modules.md
- docs/technical/mathematical_formulation.tex  
- README.md
- 訓練範例和教學
```

### 階段 4: 測試與驗證 (1天)

#### 4.1 功能測試
```python
# 新增完整的單元測試
test_modules = [
    'test_standard_traffic_lstm.py',      # 模型功能測試
    'test_lstm_trainer.py',               # 訓練器測試
    'test_lstm_config.py',                # 配置測試
    'test_lstm_integration.py',           # 整合測試
]
```

#### 4.2 性能驗證
```python
# 確保統一後性能不下降
performance_metrics = [
    'training_speed',          # 訓練速度
    'memory_usage',           # 記憶體使用
    'prediction_accuracy',    # 預測精度
    'model_size',            # 模型大小
]
```

## 後果

### 正面後果

- **維護效率提升**：單一實現，修改只需要改一處
- **程式碼品質改善**：保留最佳實現，移除有問題的代碼
- **開發效率提升**：明確的標準，減少決策時間
- **專案形象改善**：統一的高品質實現更專業
- **Social-xLSTM 基礎**：為核心創新提供穩固基線
- **文檔簡化**：減少需要維護的文檔數量
- **測試簡化**：減少需要測試的重複功能

### 負面後果

- **短期開發成本**：需要 3-5 天的重構工作
- **引用更新成本**：需要更新現有的引用
- **學習成本**：團隊需要適應新的統一介面
- **功能遺失風險**：可能遺失某些邊緣功能

### 風險與緩解措施

- **功能遺失風險**: 統一過程中可能遺失某些功能 / **緩解**: 詳細分析每個實現，建立功能清單，逐一驗證
- **引用破壞風險**: 更新引用時可能遺漏某些文件 / **緩解**: 使用 grep 全域搜索，建立更新清單
- **性能下降風險**: 新實現可能性能不如舊實現 / **緩解**: 建立性能基準測試，確保性能不下降
- **時程延誤風險**: 重構耗時超出預期 / **緩解**: 分階段執行，每階段都有明確的時間限制
- **新 bug 風險**: 重構可能引入新的錯誤 / **緩解**: 充分的測試，逐步部署

## 實施時程

| 階段 | 任務 | 時間 | 負責人 | 完成標準 |
|------|------|------|--------|----------|
| 1 | 功能分析與精華提取 | 1天 | 開發者 | 功能清單完成 |
| 2 | 標準實現建立 | 2天 | 開發者 | 新實現通過基本測試 |
| 3 | 遷移與清理 | 2天 | 開發者 | 舊實現移除，引用更新 |
| 4 | 測試與驗證 | 1天 | 開發者 | 所有測試通過 |

**總計**: 6天（包含 1天緩衝時間）

## 成功指標

### 量化指標
- [x] LSTM 實現數量從 5 個減少到 1 個 (已完成)
- [x] 重複代碼行數減少 > 80% (已完成)
- [x] 測試覆蓋率 > 90%（針對統一實現）(已完成)
- [x] 所有現有功能保持正常運作 (已完成)

### 質量指標
- [x] 代碼審查通過 (已完成)
- [x] 文檔更新完成 (已完成)
- [x] 性能測試通過（不低於最佳舊實現）(已完成)
- [x] 團隊成員能夠順利使用新實現 (已完成)

## 實施狀態更新 (2025-01-09)

### 已完成項目
1. **統一 LSTM 實現**: 基於 `src/social_xlstm/models/lstm.py` 的統一實現已完成
2. **訓練腳本重構**: 完成 ADR-0400 訓練腳本重構，減少 48% 代碼重複
3. **重複實現清理**: 移除了 `lab/train/pure_lstm.py` 等重複實現
4. **共用函數提取**: 建立 `scripts/train/common.py` 共用函數庫

### 部分完成項目
1. **配置系統**: 簡化後不使用 YAML 配置，直接使用命令行參數
2. **模型工廠**: 不需要複雜的模型工廠，直接使用 TrafficLSTM 類方法

### 實際結果與偏差
- **簡化實現**: 避免過度工程化，採用更簡潔的實現方式
- **命令行優先**: 使用命令行參數而非 YAML 配置
- **統一介面**: 透過 TrafficLSTM 類的統一介面支援單VD和多VD模式

## 替代方案考量

### 如果此方案被拒絕
1. **最小化方案**: 只移除明顯有問題的實現（loader_xlstm.py）
2. **文檔化方案**: 建立清晰的使用指南，說明何時使用哪個實現
3. **逐步淘汰方案**: 標記某些實現為 deprecated，逐步引導使用者遷移

## 相關決策

- [ADR-0001: 專案架構清理與重組](0001-project-architecture-cleanup.md)
- [ADR-0100: Social Pooling vs Graph Networks](0100-social-pooling-vs-graph-networks.md)
- [ADR-0101: xLSTM vs Traditional LSTM](0101-xlstm-vs-traditional-lstm.md)
- [ADR-0300: 下一階段開發優先級規劃](0300-next-development-priorities.md)
- [ADR-0400: 訓練腳本重構策略](0400-training-script-refactoring.md)

## 註記

### 基準實現選擇理由

**選擇 `pure/traffic_lstm.py` 作為基準的詳細原因**:

1. **架構設計**: 88行精練實現，邏輯清晰，沒有冗餘代碼
2. **專業標準**: 英文注釋，符合國際學術標準
3. **功能完整**: 支援多步預測，核心功能齊全
4. **擴展性**: 介面設計清晰，易於擴展和修改
5. **品質指標**: 代碼品質最高，最容易理解和維護

### 保留功能清單

從其他實現中需要保留的功能：
- 可視化功能（來自 traffic_lstm.py）
- 批次處理邏輯（來自 loader_lstm.py）
- 現代化配置管理（來自 pure_lstm.py）
- 評估指標整合（來自多個來源）

### 技術債務清理

此次統一不僅解決 LSTM 重複問題，也為後續 Social-xLSTM 實現提供：
- 清晰的基線模型
- 統一的訓練介面
- 標準化的評估方法
- 專業的代碼品質標準