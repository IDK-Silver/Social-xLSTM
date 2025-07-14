# VD Batch Processing and Selection: Technical Guide

## 概覽

本文檔詳細說明 Social-xLSTM 專案中 Vehicle Detector (VD) 批次處理和選擇機制的設計原理、實現方法和技術細節。這是一個關鍵的系統組件，確保多 VD 時間序列資料能正確處理。

## 核心問題與解決方案

### 問題背景

在交通預測系統中，我們需要處理多個 Vehicle Detector (VD) 的時間序列資料。每個訓練樣本包含：
- **時間維度**：一個時間窗口內的序列資料
- **空間維度**：多個 VD 檢測器的同步資料
- **特徵維度**：每個 VD 的交通特徵（流量、速度、佔有率等）

### 資料結構設計

#### 1. 張量結構
```python
# 批次資料的張量形狀
input_seq:  [batch_size, sequence_length, num_vds, num_features]
           [2,          5,               3,        5]

target_seq: [batch_size, prediction_length, num_vds, num_features]
           [2,          1,                  3,        5]

# 實際範例
batch = {
    'input_seq': torch.Tensor([2, 5, 3, 5]),  # 2個樣本，5步輸入，3個VD，5個特徵
    'target_seq': torch.Tensor([2, 1, 3, 5]), # 2個樣本，1步預測，3個VD，5個特徵
    'vdids': [
        ['VD-A', 'VD-B', 'VD-C'],             # 樣本0的VD順序
        ['VD-A', 'VD-B', 'VD-C']              # 樣本1的VD順序
    ]
}
```

#### 2. 空間一致性原則

**關鍵設計決策**：每個批次中的所有樣本都必須包含相同的 VD 集合，但時間窗口不同。

```python
# 正確的批次組織方式
樣本0: 時間 T1~T5，VD ['VD-A', 'VD-B', 'VD-C']
樣本1: 時間 T6~T10，VD ['VD-A', 'VD-B', 'VD-C']  # 相同VD，不同時間
樣本2: 時間 T11~T15，VD ['VD-A', 'VD-B', 'VD-C'] # 相同VD，不同時間
```

**為什麼這樣設計？**
1. **索引一致性**：`tensor[batch_idx, time_idx, vd_idx, feature_idx]` 中的 `vd_idx` 在所有樣本中指向相同的 VD
2. **Social Pooling 準備**：未來的空間聚合需要一致的空間結構
3. **批次計算效率**：能夠進行高效的向量化運算

## PyTorch DataLoader 的影響

### 原始樣本結構
```python
# TrafficTimeSeries.__getitem__ 的輸出
sample_0 = {
    'input_seq': tensor([5, 3, 5]),           # [seq_len, num_vds, features]
    'vdids': ['VD-A', 'VD-B', 'VD-C']        # VD識別符列表
}

sample_1 = {
    'input_seq': tensor([5, 3, 5]),           # [seq_len, num_vds, features]
    'vdids': ['VD-A', 'VD-B', 'VD-C']        # 相同的VD識別符列表
}
```

### PyTorch Collate 行為

PyTorch 的 `default_collate` 函數會自動合併批次：

```python
# 輸入（多個樣本）
samples = [
    {'vdids': ['VD-A', 'VD-B', 'VD-C']},  # 樣本0
    {'vdids': ['VD-A', 'VD-B', 'VD-C']}   # 樣本1
]

# 輸出（批次）
batch = {
    'vdids': [
        ['VD-A', 'VD-B', 'VD-C'],         # 來自樣本0
        ['VD-A', 'VD-B', 'VD-C']          # 來自樣本1
    ]
}
```

### 特殊情況：PyTorch Collate 重排

在某些情況下，PyTorch 可能會重新組織巢狀結構：

```python
# 正常格式
vdids = [
    ['VD-A', 'VD-B', 'VD-C'],  # 樣本0
    ['VD-A', 'VD-B', 'VD-C']   # 樣本1
]

# PyTorch collate 可能產生的格式
vdids = [
    ['VD-A', 'VD-A'],  # 所有樣本的第1個VD
    ['VD-B', 'VD-B'],  # 所有樣本的第2個VD
    ['VD-C', 'VD-C']   # 所有樣本的第3個VD
]
```

## Single VD Selection 處理邏輯

### 核心挑戰

當訓練 Single VD 模型時，我們需要：
1. 從多 VD 批次中選擇特定的 VD
2. 處理 PyTorch 可能產生的不同資料格式
3. 確保選擇到正確的張量維度

### 解決方案實現

```python
def prepare_batch(self, batch):
    """準備 Single VD 訓練的批次資料"""
    
    # 預設使用第一個VD
    vd_idx = 0
    
    if self.select_vd_id and 'vdids' in batch:
        vdids = batch['vdids']
        
        if isinstance(vdids, list) and len(vdids) > 0:
            # 處理兩種可能的格式
            if isinstance(vdids[0], list):
                # 格式檢測與重建
                sample_vdids = self._extract_vd_list(vdids)
                
                # 查找目標VD的索引
                if self.select_vd_id in sample_vdids:
                    vd_idx = sample_vdids.index(self.select_vd_id)
    
    # 選擇特定VD的資料
    inputs = batch['input_seq'][:, :, vd_idx, :]   # [batch, seq, features]
    targets = batch['target_seq'][:, :, vd_idx, :] # [batch, pred, features]
    
    return inputs, targets
```

### 格式檢測算法

```python
def _extract_vd_list(self, vdids):
    """從可能的兩種格式中提取正確的VD列表"""
    
    first_sample_vdids = vdids[0]
    
    # 檢查是否為PyTorch collate格式
    if len(vdids) > 1 and len(vdids[0]) == len(vdids[1]):
        # 檢測重複模式：每個子列表的元素是否相同
        is_pytorch_format = all(
            vdids[i][0] == vdids[i][1] if len(vdids[i]) >= 2 else True
            for i in range(len(vdids))
        )
        
        if is_pytorch_format:
            # 重建VD列表：從每個子列表取第一個元素
            return [vdids[i][0] for i in range(len(vdids))]
    
    # 使用正常格式
    return first_sample_vdids
```

## 技術實現細節

### 1. 索引映射關係

```python
# VD在張量中的位置
# batch['input_seq'][樣本索引, 時間索引, VD索引, 特徵索引]

# 範例：選擇VD-B（索引1）的資料
vd_b_data = batch['input_seq'][:, :, 1, :]  # 所有樣本的VD-B資料

# 詳細映射
batch['input_seq'][0, :, 1, :] ← 樣本0中VD-B的時間序列
batch['input_seq'][1, :, 1, :] ← 樣本1中VD-B的時間序列
```

### 2. 錯誤處理機制

```python
def prepare_batch(self, batch):
    try:
        # 嘗試選擇指定的VD
        vd_idx = self._find_vd_index(batch['vdids'], self.select_vd_id)
    except (KeyError, ValueError, IndexError) as e:
        # 回退到第一個VD
        logger.warning(f"VD {self.select_vd_id} not found, using first VD: {e}")
        vd_idx = 0
    
    return self._extract_vd_data(batch, vd_idx)
```

### 3. 記錄和偵錯

```python
# 詳細的除錯訊息
logger.debug(f"Batch VD structure: {batch['vdids']}")
logger.debug(f"Target VD: {self.select_vd_id}")
logger.debug(f"Selected VD index: {vd_idx}")
logger.debug(f"Extracted VD list: {sample_vdids}")
```

## 驗證和測試

### 單元測試覆蓋

1. **正常格式測試**：`[['VD-A', 'VD-B'], ['VD-A', 'VD-B']]`
2. **PyTorch格式測試**：`[['VD-A', 'VD-A'], ['VD-B', 'VD-B']]`
3. **邊界情況測試**：空列表、不存在的VD、格式錯誤
4. **批次大小變化測試**：1, 2, 4, 8等不同批次大小

### 整合測試

```python
def test_real_data_integration():
    """使用真實HDF5資料測試VD選擇功能"""
    
    # 載入真實資料
    data_config = TrafficDatasetConfig(
        hdf5_path="blob/dataset/pre-processed/h5/traffic_features.h5",
        selected_vdids=["VD-11-0020-002-001", "VD-28-0740-000-001"]
    )
    
    # 測試VD選擇
    for target_vd in data_config.selected_vdids:
        trainer = SingleVDTrainer(model, config, train_loader)
        inputs, targets = trainer.prepare_batch(real_batch)
        
        # 驗證選擇正確性
        assert inputs.shape == (batch_size, seq_len, num_features)
        assert not torch.isnan(inputs).any()
```

## 性能考量

### 計算複雜度

- **VD檢測**：O(B × V)，其中 B 是批次大小，V 是 VD 數量
- **資料選擇**：O(1)，直接索引操作
- **總體複雜度**：可忽略，相對於模型計算微不足道

### 記憶體使用

```python
# 原始批次記憶體：[B, T, V, F] 
# 選擇後記憶體：[B, T, F]
# 記憶體減少：1/V（V是VD數量）

# 範例：3個VD → 記憶體使用減少至33%
```

## 最佳實踐

### 1. 配置建議

```python
# 推薦的訓練配置
config = SingleVDTrainingConfig(
    select_vd_id="VD-11-0020-002-001",  # 明確指定VD ID
    batch_size=32,                      # 適中的批次大小
    prediction_steps=1,                 # 單步預測
)
```

### 2. 除錯技巧

```python
# 啟用詳細記錄
logging.getLogger('social_xlstm.training').setLevel(logging.DEBUG)

# 檢查批次結構
print(f"VD structure: {batch['vdids']}")
print(f"Tensor shape: {batch['input_seq'].shape}")
```

### 3. 常見錯誤避免

```python
# ❌ 錯誤：假設固定的VD順序
vd_idx = 1  # 假設VD-B總是在索引1

# ✅ 正確：動態查找VD索引
vd_idx = sample_vdids.index(target_vd_id)
```

## 與 Social Pooling 的關係

### 未來擴展性

當前的批次處理設計為未來實現 Social Pooling 奠定了基礎：

```python
# Single VD：選擇一個VD進行訓練
inputs = batch['input_seq'][:, :, vd_idx, :]

# Social Pooling：使用所有VD進行空間聚合
social_features = self.social_pooling(
    batch['input_seq'],    # 所有VD的資料
    batch['vdids'],        # VD空間關係
    target_vd_idx=vd_idx   # 目標VD
)
```

### 空間關係保持

批次處理確保了空間關係的一致性，這對 Social Pooling 至關重要：

```python
# 所有樣本中的VD相對位置保持不變
vd_coordinates = {
    'VD-A': (x1, y1),  # 固定座標
    'VD-B': (x2, y2),  # 固定座標
    'VD-C': (x3, y3)   # 固定座標
}
```

## 結論

VD 批次處理和選擇機制是 Social-xLSTM 系統的關鍵組件，它：

1. **確保資料一致性**：所有批次樣本具有相同的空間結構
2. **處理 PyTorch 複雜性**：優雅地處理 DataLoader 的不同行為
3. **支援靈活選擇**：能夠選擇特定 VD 進行 Single VD 訓練
4. **為未來做準備**：設計支援即將實現的 Social Pooling 功能

這種設計不是權宜之計，而是經過深思熟慮的架構決策，平衡了**資料一致性**、**計算效率**和**系統可擴展性**。

## 參考文獻

- **ADR-0100**: Social Pooling vs Graph Networks 決策
- **ADR-0101**: xLSTM vs Traditional LSTM 決策  
- **ADR-0200**: 座標系統選擇決策
- **數學公式文檔**: `docs/technical/mathematical_formulation.tex`
- **測試文檔**: `test/test_vd_selection.py`, `test/test_vd_selection_real_data.py`