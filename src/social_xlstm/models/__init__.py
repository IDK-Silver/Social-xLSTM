"""
Social-xLSTM Models Module

This module provides different LSTM architectures for traffic prediction:
- TrafficLSTM: Traditional LSTM implementation (~226K parameters)
- TrafficXLSTM: Extended LSTM implementation (~655K parameters) ✅ 已實現

🚀 快速使用:
```python
# Traditional LSTM
from social_xlstm.models import TrafficLSTM, TrafficLSTMConfig
lstm_model = TrafficLSTM(TrafficLSTMConfig())

# Extended LSTM (xLSTM)
from social_xlstm.models import TrafficXLSTM, TrafficXLSTMConfig  
xlstm_model = TrafficXLSTM(TrafficXLSTMConfig())
```

📚 詳細文檔:
- LSTM 指南: docs/guides/lstm_usage_guide.md
- xLSTM 指南: docs/guides/xlstm_usage_guide.md
"""

from .lstm import TrafficLSTM, TrafficLSTMConfig
from .xlstm import TrafficXLSTM, TrafficXLSTMConfig

__all__ = [
    # LSTM models
    'TrafficLSTM',
    'TrafficLSTMConfig', 
    
    # xLSTM models
    'TrafficXLSTM',
    'TrafficXLSTMConfig',
]