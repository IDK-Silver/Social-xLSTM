"""
Social-xLSTM Models Module

This module provides different LSTM architectures for traffic prediction:
- TrafficLSTM: Traditional LSTM implementation
- TrafficXLSTM: Extended LSTM implementation (planned)
"""

from .lstm import TrafficLSTM, TrafficLSTMConfig

__all__ = [
    # LSTM models
    'TrafficLSTM',
    'TrafficLSTMConfig', 
    
    # xLSTM models (planned)
    # 'TrafficXLSTM',
    # 'TrafficXLSTMConfig',
]