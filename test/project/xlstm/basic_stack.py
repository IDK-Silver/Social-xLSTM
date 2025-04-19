import torch
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# 範例配置 (混合 mLSTM 和 sLSTM 概念以供說明)
# 注意: slstm_at= 表示第一個區塊是 sLSTM，其餘默認 (如果定義了 mLSTM，則可能是 mLSTM)
# 請查閱官方文檔/代碼以了解確切的默認行為。
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig( # 定義 mLSTM 區塊配置
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=64, num_heads=8 # 調整參數
        ),
    ),
    slstm_block=sLSTMBlockConfig( # 定義 sLSTM 區塊配置
        slstm=sLSTMLayerConfig(
            backend="cuda", # 如果可用且兼容，則使用 CUDA 核心
            num_heads=8, # 調整參數
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent", # 範例偏置初始化策略
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"), # sLSTM 區塊內的前饋層
    ),
    context_length=256, # 最大序列長度
    num_blocks=8, # 區塊總數
    embedding_dim=128, # 堆疊的輸入/輸出維度
    slstm_at=[1, 2], # 指定 sLSTM 區塊應使用的索引 (例如，第一個區塊)
)

# 實例化堆疊
xlstm_stack = xLSTMBlockStack(cfg)
print("xLSTM Stack Instantiated:", xlstm_stack)

# 準備虛擬輸入數據 (批量大小, 序列長度, 嵌入維度)
x = torch.randn(4, 256, 128)

# 如果可用，將模型和數據移動到 GPU
if torch.cuda.is_available():
    try:
        xlstm_stack = xlstm_stack.to("cuda")
        x = x.to("cuda")
        print("Using CUDA")
    except Exception as e:
        print(f"Failed to move to CUDA: {e}. Check CUDA setup and device compatibility.")
        # 如果移至 CUDA 失敗，則保持在 CPU 上運行
        pass

# 執行前向傳播
try:
    output = xlstm_stack(x)
    print("Forward pass successful!")
    print("Input shape:", x.shape)
    # 預期輸出形狀: (批量大小, 序列長度, 嵌入維度)
    print("Output shape:", output.shape)
    assert output.shape == x.shape
except Exception as e:
    print(f"Error during forward pass: {e}")
    print("Ensure CUDA environment and dependencies (like kernels) are correctly set up if using CUDA backend.")