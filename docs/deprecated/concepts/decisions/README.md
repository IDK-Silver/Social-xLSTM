# Architecture Decision Records (ADRs)

本目錄包含 Social-xLSTM 專案的重要架構決策記錄，記錄了關鍵技術選擇的背景、原因和影響。

## 決策索引

### ADR-001: Distance-Based Social Pooling Implementation
**檔案**: [adr-001-distance-based-social-pooling.md](adr-001-distance-based-social-pooling.md)  
**狀態**: ✅ 已接受  
**日期**: 2025-08-03  
**摘要**: 選擇距離基礎連續社交池化而非原始 Social LSTM 的網格基礎離散化方法

**核心決策**:
- 使用歐幾里得距離進行鄰居選擇
- 連續空間表示取代離散網格
- 距離權重聚合機制

**影響範圍**:
- `src/social_xlstm/pooling/xlstm_pooling.py`
- Social-xLSTM 核心架構設計
- 與原始 Social LSTM 論文的實現差異

---

## ADR 編寫指南

### 何時建立 ADR
- 重要架構或設計決策
- 與現有標準或論文的偏離
- 影響多個模組的技術選擇
- 需要記錄原因的重構決定

### ADR 結構
1. **Status**: 建議 (Proposed) / 已接受 (Accepted) / 廢棄 (Deprecated)
2. **Context**: 決策背景和需要解決的問題
3. **Decision**: 具體決策內容
4. **Rationale**: 決策理由和考慮因素
5. **Consequences**: 正面和負面影響
6. **Implementation**: 實現細節和相關檔案
7. **References**: 相關文檔和論文

### 檔案命名規範
```
adr-{序號}-{簡短描述}.md
```

例如：
- `adr-001-distance-based-social-pooling.md`
- `adr-002-xlstm-vs-traditional-lstm.md`
- `adr-003-distributed-vs-centralized-architecture.md`

---

## 相關文檔

- [架構設計文檔](../architecture/)
- [技術規格文檔](../technical/)
- [實現指南](../implementation/)

## 更新記錄

- **2025-08-03**: 建立 ADR 框架和第一個決策記錄