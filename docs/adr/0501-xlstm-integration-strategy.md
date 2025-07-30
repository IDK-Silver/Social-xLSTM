# ADR-0501: xLSTM 整合策略與並行訓練架構

**狀態**: 已批准  
**日期**: 2025-07-13  
**決策者**: 開發團隊  

## 📋 背景

經過 LSTM 基準系統的完整建立（ADR-0502），我們已具備穩固的評估基礎。下一階段需要整合 xLSTM（Extended LSTM）技術，以驗證其是否能改善當前 LSTM 系統的嚴重過擬合問題。

### 當前 LSTM 基準表現
- **Single VD**: 訓練 R² = 0.93，驗證 R² = -6（嚴重過擬合）
- **Multi-VD**: 測試 R² = -1.11，MAE = 0.737
- **Independent Multi-VD**: 測試 R² = -3.76，MAE = 1.58

### xLSTM 技術特點
基於 2024 年 NeurIPS 論文，xLSTM 包含：
- **sLSTM (Scalar LSTM)**: 指數門控、標量記憶體、數值穩定性改進
- **mLSTM (Matrix LSTM)**: 矩陣記憶體、協方差更新、完全並行化
- **混合架構**: sLSTM 和 mLSTM 塊的殘差堆疊

## 🎯 問題陳述

需要決定如何將 xLSTM 整合到現有系統中，同時保持：
1. **向後兼容性**: 現有 LSTM 基準不受影響
2. **比較能力**: 能夠系統性比較 LSTM vs xLSTM 性能
3. **開發效率**: 最小化代碼重複和維護複雜度
4. **實驗管理**: 支援並行訓練和結果比較

## 🔍 考慮的選項

### 選項 A: 擴展現有 TrafficLSTM 類
```python
# 在現有類中添加 xLSTM 支援
class TrafficLSTM(nn.Module):
    def __init__(self, config):
        if config.model_type == 'xlstm':
            self._init_xlstm()
        else:
            self._init_lstm()
```

**優點**:
- 代碼統一，維護簡單
- 配置相對集中

**缺點**:
- 架構差異巨大，強行統一會導致複雜度爆炸
- xLSTM 需要完全不同的配置參數
- 測試和調試困難
- 違反單一職責原則

### 選項 B: 創建獨立的 TrafficXLSTM 類 ⭐ **推薦**
```python
# 完全獨立的實現
class TrafficXLSTM(nn.Module):
    def __init__(self, config: TrafficXLSTMConfig):
        # 專門的 xLSTM 實現
```

**優點**:
- 架構清晰，職責分離
- 各自有專門的配置系統
- 易於測試和調優
- 擴展性好，未來可添加其他架構

**缺點**:
- 需要額外的代碼文件
- 需要修改訓練腳本和工作流程

### 選項 C: 包裝器模式
```python
# 統一接口，內部委託
class TrafficModel(nn.Module):
    def __init__(self, model_type, config):
        if model_type == 'lstm':
            self.model = TrafficLSTM(config)
        elif model_type == 'xlstm':
            self.model = TrafficXLSTM(config)
```

**優點**:
- 統一的外部接口
- 內部實現完全獨立

**缺點**:
- 增加抽象層複雜度
- 配置管理仍然複雜

## 🎯 決策

**選擇選項 B**: 創建獨立的 TrafficXLSTM 類

### 理由
1. **架構差異過大**: xLSTM 的配置需求與 LSTM 完全不同
2. **清晰的職責分離**: 每個類專注於一種架構
3. **測試獨立性**: 可以分別開發和調優
4. **未來擴展性**: 為 Social-xLSTM 和其他架構留出空間

## 🛠️ 實施方案

### 1. 文件結構
```
src/social_xlstm/models/
├── lstm.py          # 現有的 TrafficLSTM
├── xlstm.py         # 新的 TrafficXLSTM ⭐ 新增
└── __init__.py      # 導出兩個類別
```

### 2. 配置系統分離
```python
@dataclass
class TrafficLSTMConfig:
    """LSTM 專用配置"""
    input_size: int = 5
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    # ... LSTM 特有參數

@dataclass  
class TrafficXLSTMConfig:
    """xLSTM 專用配置"""
    input_size: int = 5
    embedding_dim: int = 128
    num_blocks: int = 6
    slstm_at: List[int] = field(default_factory=lambda: [1, 3])
    # ... xLSTM 特有參數
```

### 3. Snakemake 工作流程分離
```yaml
# 支援並行訓練的配置結構
training_lstm:
  single_vd:
    model_type: "lstm"
    experiment_dir: "blob/experiments/dev/single_vd_lstm"
    # ... LSTM 參數

training_xlstm:
  single_vd:
    model_type: "xlstm"  
    experiment_dir: "blob/experiments/dev/single_vd_xlstm"
    # ... xLSTM 參數
```

### 4. 訓練規則重構
```python
# Snakefile 中的並行規則
rule train_single_vd_lstm:
    # LSTM 訓練規則

rule train_single_vd_xlstm:
    # xLSTM 訓練規則

rule compare_lstm_xlstm:
    # 性能比較規則
```

## 📊 實施階段

### 階段 1: 基礎架構 (第一週)
- [ ] 創建 `TrafficXLSTM` 類和配置
- [ ] 實現基本的 xLSTM 功能
- [ ] 單元測試驗證

### 階段 2: 工作流程整合 (第二週)
- [ ] 修改 Snakefile 支援並行訓練
- [ ] 創建 xLSTM 專用訓練腳本
- [ ] 配置文件分離

### 階段 3: 性能驗證 (第三週)
- [ ] LSTM vs xLSTM 並行訓練
- [ ] 性能比較和分析
- [ ] 結果可視化和報告

## 🔗 技術依賴

### 新增依賴
```bash
pip install xlstm
# 注意：暫時不安裝 mlstm_kernels，先用 vanilla 後端驗證基本功能
```

#### 依賴策略說明

**階段 1: 基礎驗證** (推薦)
```python
# 使用 vanilla 後端
slstm_backend: 'vanilla'  # 穩定、兼容性好、避免額外複雜性
```

**階段 2: 性能優化** (後續考慮)
```bash
pip install mlstm_kernels  # Tiled Flash Linear Attention (TFLA) 優化內核
```

#### mlstm_kernels 說明
`mlstm_kernels` 是 NX-AI 開發的性能優化庫：

**優勢**:
- **TFLA 內核**: 提供 Tiled Flash Linear Attention 快速訓練推理
- **顯著加速**: 相比 vanilla 後端有明顯性能提升
- **CUDA 優化**: 專為 GPU 加速設計

**當前決策 - 暫不使用**:
- ⚠️ **複雜性風險**: 可能引入意外的編譯或相依性問題
- 🎯 **功能優先**: 先確保 xLSTM 基本功能正常運作
- 🔄 **漸進式升級**: 基礎功能穩定後再考慮性能優化

**未來升級路徑**:
1. 第一週：vanilla 後端驗證功能
2. 第二週：如無問題，考慮升級到 CUDA 後端
3. 性能測試：比較 vanilla vs CUDA 後端的實際效果

### 硬體需求
- CUDA 支援（推薦）
- 額外的 GPU 記憶體（xLSTM 可能需要更多資源）

## 📈 成功指標

### 技術指標
- [ ] TrafficXLSTM 能夠成功訓練
- [ ] 訓練穩定性良好
- [ ] 記憶體使用在可接受範圍內

### 性能指標
- [ ] xLSTM 相比 LSTM 的過擬合改善
- [ ] 驗證 R² > 0（目前 LSTM 為負值）
- [ ] 訓練/驗證指標差距 < 2 倍

### 系統指標
- [ ] 並行訓練工作流程正常運行
- [ ] 結果比較和視覺化功能完整
- [ ] 代碼維護性良好

## ⚠️ 風險與緩解

### 風險 1: xLSTM 性能不如預期
**機率**: 中等  
**影響**: 高  
**緩解**: 保持 LSTM 基準完整，作為備選方案

### 風險 2: 記憶體需求過高
**機率**: 中等  
**影響**: 中等  
**緩解**: 調整 batch_size、使用梯度累積

### 風險 3: 實施複雜度超預期
**機率**: 低  
**影響**: 中等  
**緩解**: 分階段實施，每階段都有可交付成果

## 🔄 相關 ADR

- **ADR-0502**: LSTM 基準完成與下一步規劃
- **ADR-0100**: Social Pooling vs Graph Networks  
- **ADR-0101**: xLSTM vs Traditional LSTM

## 📝 決策記錄

**決策**: 批准選項 B - 創建獨立的 TrafficXLSTM 類

**批准條件**:
1. 保持現有 LSTM 系統完全不變
2. 實施分階段進行，每階段有明確交付物
3. 建立完整的性能比較框架
4. 確保代碼品質和測試覆蓋率

**實施開始日期**: 2025-07-14  
**預期完成日期**: 2025-08-03