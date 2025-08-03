"""
Distributed data loading for Social-xLSTM.

This package provides distributed data loading capabilities that transform
centralized tensor format into per-VD dictionary format for xLSTM processing.
"""

from .distributed_datamodule import DistributedTrafficDataModule, create_distributed_datamodule

__all__ = [
    'DistributedTrafficDataModule',
    'create_distributed_datamodule'
]