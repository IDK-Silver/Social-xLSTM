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


class Traffic_xLSTM(torch.nn.Module):
    def __init__(self, cfg: xLSTMBlockStackConfig):
        super(Traffic_xLSTM, self).__init__()
        self.cfg = cfg
        self.xlstm_stack = xLSTMBlockStack(cfg)

    def forward(self, x):
        return self.xlstm_stack(x)

