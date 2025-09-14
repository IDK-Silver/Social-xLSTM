# ST-Attention Social Pooling（kNN + learnable τ）教學

本文件用教學方式完整解釋 `src/social_xlstm/pooling/st_attention_pooling.py` 的設計與使用方法，從動機、數學形式、維度對齊、效能取捨到實作細節，假設讀者不熟悉 Attention 或圖學習也能看懂。

目標：在交通序列預測中，同時建模「空間關聯（誰影響誰）」與「時間動態（近期趨勢）」；使用可學習的距離衰減與稀疏鄰接，讓模型又快又準、可擴充。

---

## 1. 介面與資料形狀

- 隱狀態輸入（Encoder 輸出）`hidden`: 形狀 `[B, N, T, E]`
  - `B`: batch size
  - `N`: VD（節點）數量
  - `T`: 時間步長（序列長度）
  - `E`: 每一節點的特徵表徵維度（embedding_dim）
- 節點座標 `positions_xy`: 形狀 `[N, 2]`
  - 每個 VD 的固定平面座標（單位建議為「公尺」，由資料模組使用墨卡托投影換算而來）
- 輸出 `context`: 形狀 `[B, N, E]`
  - 每個節點的「社交上下文」向量，後續會與自身隱狀態融合用於預測。

---

## 2. kNN 稀疏鄰接（空間關聯）

我們先在平面座標中計算節點兩兩的歐幾里得距離，得到距離矩陣 $D \in \mathbb{R}^{N\times N}$：

$$
D_{ij} = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2
$$

對每個節點 $i$，從 $D_{i,:}$ 取距離最小的 $k$ 個鄰居索引：

- `knn_k`: 每個節點選多少個鄰居（通常 8–32）
- 可選「半徑遮罩」`use_radius_mask=true`，把超過半徑 `radius` 的鄰居在後續加權時直接視為 0（硬遮罩）；預設關閉，用純 kNN。

這一步讓複雜度從 $O(N^2)$ 稀疏化為 $O(N\,k)$，是可伸縮的關鍵。

---

## 3. 距離核與可學習溫度 τ

為了讓空間影響隨距離遞減，同時保留可學習性，我們使用指數核（exponential kernel）：

$$
w(d) \,=\, \exp\!\left( -\frac{d}{\tau} \right), \qquad \tau>0
$$

- 其中 $\tau$ 是「溫度」；值越大，衰減越慢。程式中 `tau` 可設為 `nn.Parameter`，可在訓練中自動學習（`learnable_tau: true`）。
- 在注意力 logits 中，我們使用 $\log(w + \varepsilon)$ 作為加性偏置（bias），避免數值下溢：

$$
\text{logits}_{i,j} \;+=\; \log\big(w(D_{ij}) + \varepsilon\big)
$$

如果你同時開啟半徑遮罩，超過半徑者 $w=0$，其 $\log(w)$ 會被置換為 $-\infty$ 的「有效遮罩」。

---

## 4. 近窗時序表示（Temporal Compression）

鄰居的時間動態同樣重要。我們對每個鄰居節點 $j$ 取最近的 $L$ 個時間步 `time_window`，把它們的隱狀態做平均（也可擴充為 1D Conv）：

$$
\bar{\mathbf{h}}_j \,=\, \frac{1}{L} \sum_{t=T-L+1}^{T} \mathbf{h}_j(t) \in \mathbb{R}^{E}
$$

如此每個鄰居只保留一個代表向量 $\bar{\mathbf{h}}_j$，降低計算量、也讓注意力專注於「近期趨勢」。

---

## 5. 多頭注意力（Multi-Head Attention）

對每個目標節點 $i$：

- Query 取自身最後一步的隱狀態 $\mathbf{h}_i(T)$；
- Key/Value 取每個鄰居的近窗表示 $\bar{\mathbf{h}}_j$；
- 投影到多頭空間：

$$
\mathbf{Q}_i = W_Q \mathbf{h}_i(T),\quad
\mathbf{K}_j = W_K \bar{\mathbf{h}}_j,\quad
\mathbf{V}_j = W_V \bar{\mathbf{h}}_j
$$

對第 $h$ 個 head，注意力 logits（加入距離偏置）為：

$$
\text{logits}_{i,j}^{(h)} = \frac{\langle \mathbf{q}_i^{(h)}, \mathbf{k}_j^{(h)} \rangle}{\sqrt{d_h}} + \log\big(w(D_{ij}) + \varepsilon\big)
$$

softmax 後得到權重 $\alpha_{i,j}^{(h)}$，再做加權平均得到上下文：

$$
\mathbf{c}_i^{(h)} = \sum_{j\in \mathcal{N}_k(i)} \alpha_{i,j}^{(h)}\, \mathbf{v}_j^{(h)}
$$

把所有 head 串接再線性投影：

$$
\mathbf{c}_i = W_O\,[\mathbf{c}_i^{(1)}; \ldots; \mathbf{c}_i^{(H)}]
$$

輸出 `context` 張量即為所有 $\mathbf{c}_i$ 組成的 `[B,N,E]`。

> 提示：本實作自動計算 head 維度 $d_h$（`head_dim = E // heads`）並確保張量形狀對齊。

---

## 6. 計算複雜度與效能

- kNN 查詢：每個 batch 只對座標做一次，複雜度約為 $O(N^2)$（可接受，且與 T 無關）。
- 主體注意力：$O(B\,N\,k\,E)$，與完整 $O(B\,N^2\,E)$ 相比節省許多。
- 近窗平均：$O(B\,N\,k\,L\,E)$，其中 `L=time_window` 通常很小（如 4）。

實作採用全向量化，避免 Python 內層迴圈，讓 GPU 利用率更穩定。

---

## 7. 組態（YAML）

對應的設定在 `social_pooling` 區塊：

```yaml
social_pooling:
  enabled: true
  type: "st_attention"   # 切換到 ST-Attention 分支（非 legacy）

  # 稀疏與時序
  knn_k: 16               # 每節點選 k 個最近鄰
  time_window: 4          # 近窗長度（最近 L 個時間步）
  heads: 4                # 注意力 heads 數量

  # 距離核 w(d)=exp(-d/tau)
  learnable_tau: true
  tau_init: 1.0
  dropout: 0.1

  # 可選：半徑遮罩
  use_radius_mask: false
  radius: 100
```

你可以直接使用本庫提供的樣板：
- `cfgs/social_pooling/st_attention_basic.yaml`
- 並在 profile（如 `cfgs/profiles/pems_bay/standard.yaml`）把 `social_pooling` 設定指向它。

---

## 8. 與模型整合（如何被使用）

在 `SharedSocialXLSTMModel` 中：

1. Encoder 先對所有 VD 序列產生隱狀態：`H = [B,N,T,E]`。
2. 若 `social_pooling.type == 'st_attention'`，則呼叫 `STAttentionPooling(H, positions_xy)` 得到 `context [B,N,E]`。
3. 把每個 VD 的 `h_i(T)` 與 `context_i` 串接，經過融合與預測頭，輸出未來 `P*F` 的預測向量。

這樣的設計讓空間與時間訊息自然融合。

---

## 9. 常見問題（FAQ）

- Q: 為什麼要用 $\log(w)$ 當偏置？
  - A: 直接乘上 $w$ 容易造成梯度縮放與數值不穩；加到 logits 後由 softmax 正規化，更穩定。

- Q: `learnable_tau` 有風險嗎？
  - A: 會。若 $\tau\to 0$，權重集中在極少數鄰居；$\tau$ 太大則趨近均勻。實作會以 clamp/初始化避免退化，建議配合學習率與權重衰減做監控。

- Q: `knn_k` 如何選？
  - A: 小數值（8–16）通常足夠；過大會拖慢計算、且引入遠距離噪音。

- Q: `time_window` 多大合適？
  - A: 3–6 常見；太大會稀釋近期動態、太小又容易受噪音影響。

- Q: 能否加入道路拓撲或相關性圖？
  - A: 可以。可把拓撲當遮罩／偏置與 kNN 聯合，也可用近期互相關建圖後以 gate 與距離核混合（進階版設計）。

---

## 10. 範例（最小可執行）

```python
import torch
from social_xlstm.pooling.st_attention_pooling import STAttentionPooling

B,N,T,E = 2, 10, 12, 128
hidden = torch.randn(B, N, T, E)
xy = torch.randn(N, 2)

pool = STAttentionPooling(hidden_dim=E, knn_k=4, time_window=3, heads=4, learnable_tau=True)
context = pool(hidden, xy)  # [B, N, E]
```

---

## 11. 延伸方向

- 注意力偏置加入相對方向（$\Delta x, \Delta y$）或小型 MLP，提升空間結構感知。
- 把鄰居序列壓縮由「平均」改為 1D 卷積或小型 TCN，抓取更細緻時間紋理。
- 多尺度鄰域：local kNN + 道路拓撲鄰居，雙分支注意力後融合。

---

## 12. 參考與對照

- 實作檔案：`src/social_xlstm/pooling/st_attention_pooling.py`
- 整合位置：`src/social_xlstm/models/shared_social_xlstm.py`
- 配置檔案：`cfgs/social_pooling/st_attention_basic.yaml`

若你對任何段落仍有疑問，建議對照原始程式中的張量維度註解與錯誤訊息，逐步印出中間形狀（例如 `print(tensor.shape)`）即可定位問題。

