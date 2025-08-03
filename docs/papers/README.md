# 論文參考資料庫

本目錄包含 Social-xLSTM 專案的核心理論基礎論文，以 markdown 格式整理並保存。

## 📚 核心論文

### 1. Social LSTM (2016)
**檔案**: [`social-lstm-2016.md`](social-lstm-2016.md)  
**作者**: Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese  
**出版**: CVPR 2016  
**重要性**: Social Pooling 概念的原始論文，為本專案的分散式架構提供理論基礎

#### 關鍵貢獻
- 提出 Social Pooling 機制，用於建模多個序列間的空間交互
- 分散式 LSTM 架構：每個個體獨立的 LSTM + 鄰居隱狀態聚合
- 在人群軌跡預測任務上顯著優於傳統 Social Force 模型

#### 對 Social-xLSTM 的影響
- **架構設計**: 每個 VD 獨立 xLSTM 實例的分散式架構
- **Social Pooling**: 隱狀態級別的空間聚合機制
- **鄰居選擇**: 基於空間距離的鄰居篩選策略

### 2. xLSTM (2024) 
**檔案**: [`xlstm-2024.md`](xlstm-2024.md)  
**作者**: Maximilian Beck, Korbinian Pöppel, Markus Spanring, et al.  
**出版**: NeurIPS 2024 (Spotlight)  
**重要性**: LSTM 架構的現代化升級，為本專案提供增強的時序建模能力

#### 關鍵貢獻
- **sLSTM**: 指數門控 + 標量記憶體 + 正規化機制
- **mLSTM**: 矩陣記憶體 + 協方差更新 + 平行化計算
- **可擴展性**: 支援數十億參數的大規模訓練

#### 對 Social-xLSTM 的影響
- **核心創新**: 使用 xLSTM 替代傳統 LSTM 進行時序建模
- **記憶體增強**: 更好的長期依賴建模能力
- **計算效率**: mLSTM 的平行化特性提升訓練效率

## 🔗 理論整合

### Social-xLSTM 架構演進

```
傳統方法:
個體特徵 → 手工社交力函數 → 軌跡預測

Social LSTM (2016):
個體序列 → LSTM → 隱狀態 → Social Pooling → 融合預測

Social-xLSTM (2025):
VD 序列 → xLSTM → 增強隱狀態 → 空間感知 Social Pooling → 交通預測
```

### 技術優勢累積

| 技術特徵 | Social LSTM | xLSTM | Social-xLSTM |
|---------|-------------|--------|--------------|
| 分散式架構 | ✅ | - | ✅ |
| 隱狀態聚合 | ✅ | - | ✅ |
| 增強記憶體 | - | ✅ | ✅ |
| 指數門控 | - | ✅ | ✅ |
| 矩陣記憶體 | - | ✅ | ✅ |
| 空間感知 | 基礎 | - | 增強 |

## 📖 閱讀建議

### 理解 Social Pooling 概念
1. **先讀**: `social-lstm-2016.md` 第3節 "Our Model: Social LSTM"
2. **重點**: Social Pooling 的網格化空間聚合機制
3. **對應**: 本專案中的 `XLSTMSocialPoolingLayer` 實現

### 理解 xLSTM 創新
1. **先讀**: `xlstm-2024.md` 第3節 "Architecture Details"  
2. **重點**: sLSTM 和 mLSTM 的技術差異
3. **對應**: 本專案中的 `TrafficXLSTM` 配置

### 整合理解
1. **架構對比**: 兩篇論文的架構圖對比分析
2. **實現細節**: 查看 `docs/technical/mathematical-specifications.md`
3. **代碼對應**: 檢查 `src/social_xlstm/models/distributed_social_xlstm.py`

## 🔍 快速查找

### Social Pooling 相關
- **原始概念**: `social-lstm-2016.md` → Section 3.1
- **數學公式**: `social-lstm-2016.md` → "Social Pooling of Hidden States"
- **實現映射**: 對應專案中兩種模式（傳統/空間感知）

### xLSTM 技術細節
- **sLSTM**: `xlstm-2024.md` → Section 3.1
- **mLSTM**: `xlstm-2024.md` → Section 3.2  
- **架構整合**: `xlstm-2024.md` → Section 3.3

### 性能對比
- **Social LSTM 基準**: `social-lstm-2016.md` → Section 4
- **xLSTM 擴展性**: `xlstm-2024.md` → Section 5
- **預期提升**: 兩種技術結合的協同效應

## 📝 引用格式

### Social LSTM
```bibtex
@inproceedings{alahi2016social,
  title={Social LSTM: Human Trajectory Prediction in Crowded Spaces},
  author={Alahi, Alexandre and Goel, Kratarth and Ramanathan, Vignesh and Robicquet, Alexandre and Fei-Fei, Li and Savarese, Silvio},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={961--971},
  year={2016}
}
```

### xLSTM  
```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and Pöppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, Günter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

## 🔄 更新記錄

- **2025-08-03**: 初始建立，包含 Social LSTM 和 xLSTM 論文整理
- **下載來源**: 
  - Social LSTM: https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf
  - xLSTM: https://arxiv.org/pdf/2405.04517

---

**注意**: 
1. 所有論文內容已轉換為 markdown 格式，便於版本控制和協作
2. 重點摘錄了與 Social-xLSTM 專案相關的技術細節
3. 完整論文請參考原始 PDF 連結