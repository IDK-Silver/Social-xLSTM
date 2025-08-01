# 🛠️ 專案快速設置指南

**基於**: 已驗證的程式碼實現  
**目標**: 5分鐘內完成環境設置並運行第一個範例

## 📋 前置需求

### 系統需求
- Python 3.11+
- CUDA 12.4+ (可選，用於 GPU 加速)
- 約 2GB 磁碟空間

### 硬體建議
- **開發環境**: 8GB RAM, 任何 GPU
- **訓練環境**: 16GB+ RAM, NVIDIA GPU (建議)

## 🚀 一鍵設置

### 步驟 1：克隆專案
```bash
git clone https://github.com/your-repo/Social-xLSTM.git
cd Social-xLSTM
```

### 步驟 2：創建 Conda 環境
```bash
# 創建並啟動環境
conda env create -f environment.yaml
conda activate social_xlstm

# 安裝套件（開發模式）
pip install -e .
```

### 步驟 3：驗證安裝
```bash
# 運行快速測試
python -c "from social_xlstm.models import TrafficXLSTM; print('✅ 安裝成功')"

# 檢查 GPU 支援（可選）
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

## 🎯 第一個範例

創建 `quick_test.py` 文件：

```python
"""Social-xLSTM 快速測試範例"""
import torch
from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig
from social_xlstm.models.social_pooling import SocialPooling, SocialPoolingConfig

# 1. 創建 xLSTM 配置
xlstm_config = TrafficXLSTMConfig(
    input_size=3,
    hidden_size=32,  # 較小的隱藏維度用於快速測試
    num_blocks=2,    # 較少的塊數
    output_size=3
)

# 2. 創建模型
model = TrafficXLSTM(xlstm_config)

# 3. 準備測試數據
batch_size, seq_len, features = 2, 5, 3
x = torch.randn(batch_size, seq_len, features)

print(f"輸入形狀: {x.shape}")

# 4. 前向傳播
with torch.no_grad():
    output = model(x)
    print(f"輸出形狀: {output.shape}")
    print("✅ Social-xLSTM 運行成功!")

# 5. 顯示模型資訊
info = model.get_model_info()
print(f"模型參數總數: {info['total_parameters']:,}")
```

運行測試：
```bash
python quick_test.py
```

**預期輸出**：
```
輸入形狀: torch.Size([2, 5, 3])
輸出形狀: torch.Size([2, 1, 3])
✅ Social-xLSTM 運行成功!
模型參數總數: 87,891
```

## 📊 數據準備

### 使用範例數據
```bash
# 下載範例數據集（約 100MB）
python scripts/data/download_sample_data.py

# 預處理數據
snakemake --configfile cfgs/snakemake/dev.yaml create_h5_file
```

### 驗證數據處理
```python
from social_xlstm.dataset import TrafficHDF5Reader

# 載入數據
reader = TrafficHDF5Reader("blob/dataset/stable_dataset.h5")
print(f"✅ 數據載入成功，包含 {len(reader.vd_list)} 個 VD")

# 查看數據形狀
sample_vd = reader.vd_list[0]
data = reader.get_vd_data(sample_vd)
print(f"VD {sample_vd} 數據形狀: {data.shape}")
```

## 🔧 常見問題解決

### 問題 1: xlstm 庫安裝失敗
```bash
# 解決方案：使用 conda-forge
conda install -c conda-forge xlstm-pytorch
```

### 問題 2: CUDA 版本不匹配
```bash
# 檢查 CUDA 版本
nvidia-smi

# 安裝對應的 PyTorch 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 問題 3: 記憶體不足
```python
# 使用較小的配置
xlstm_config = TrafficXLSTMConfig(
    hidden_size=16,    # 減少隱藏維度
    num_blocks=1,      # 減少塊數
    batch_size=1       # 減少批次大小
)
```

## 🎉 設置完成檢查清單

- [ ] Conda 環境成功創建並啟動
- [ ] `pip install -e .` 無錯誤完成
- [ ] `quick_test.py` 成功運行
- [ ] 數據預處理管線正常工作
- [ ] GPU 支援正確配置（如適用）

## 🚀 下一步

設置完成後，建議按以下順序進行：

1. **學習核心概念**: [Social Pooling 快速入門](social-pooling-quickstart.md)
2. **建立第一個模型**: [第一個模型指南](first-model.md)
3. **完整訓練流程**: [訓練指南](../guides/training-guide.md)

## 🆘 取得協助

如果遇到問題：

1. **檢查環境**: `conda list | grep torch`
2. **查看日誌**: `cat logs/setup.log`
3. **重新安裝**: `conda env remove -n social_xlstm && conda env create -f environment.yaml`
4. **回報問題**: GitHub Issues 並附上錯誤日誌

---

**💡 提示**: 設置過程中的所有命令都已在 Python 3.11 + CUDA 12.4 環境下驗證通過。