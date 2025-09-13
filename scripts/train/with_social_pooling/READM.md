## 說明
* Independent xLSTM
每一個 VD 都會有一個獨立的 xLSTM

* Shared Encoder 
所有 VD 的點都會經過一個 xLSTM 處理

## Summary 
Independent xLSTM  每個 VD 之間會完全獨立，資料不會進行如何交互，壞處是訓練很慢
Shared Encoder 每個 VD 不完全獨立，好處是訓練比較快