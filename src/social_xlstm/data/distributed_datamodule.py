"""
Distributed Traffic Data Module for Social-xLSTM.

This module implements distributed data loading that transforms centralized
[B, T, N, F] tensors into per-VD dictionary format {"VD_001": [B, T, F], ...}.

Key Features:
- Custom collate_fn for distributed batch format
- OrderedDict for deterministic VD ordering (multi-worker reproducibility)
- Tensor specification validation
- AMP-ready data pipeline

Usage:
```python
from social_xlstm.data.distributed_datamodule import DistributedTrafficDataModule

config = TrafficDatasetConfig(...)
datamodule = DistributedTrafficDataModule(config)
dataloader = datamodule.train_dataloader()

# Batch format: {"VD_001": tensor[B,T,F], "VD_002": tensor[B,T,F], ...}
batch = next(iter(dataloader))
```
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Union
from collections import OrderedDict
import logging

from ..dataset.core.datamodule import TrafficDataModule
from ..dataset.core.timeseries import TrafficTimeSeries
from ..dataset.config import TrafficDatasetConfig
from ..interfaces.tensor_spec import TensorSpec, ensure_deterministic_vd_order

logger = logging.getLogger(__name__)


class DistributedTrafficDataModule(pl.LightningDataModule):
    """
    Distributed Traffic Data Module for per-VD xLSTM architecture.
    
    This module extends the base TrafficDataModule to provide distributed
    batch format where each VD gets its own tensor in a dictionary structure.
    """
    
    def __init__(self, config: TrafficDatasetConfig):
        super().__init__()
        self.config = config
        
        # Initialize base data module for setup
        self.base_datamodule = TrafficDataModule(config)
        
        # Distributed-specific attributes
        self.vd_ids: List[str] = []
        self.input_tensor_spec: TensorSpec = None
        self.target_tensor_spec: TensorSpec = None
        
        logger.info(f"Initialized DistributedTrafficDataModule with config: {config}")
    
    def setup(self, stage: str = None):
        """Setup datasets and prepare for distributed format."""
        # Setup base datasets
        self.base_datamodule.setup(stage)
        
        # Get data info and extract VD IDs
        data_info = self.base_datamodule.get_data_info()
        self.vd_ids = data_info['vdids']
        
        # Create tensor specifications for validation
        # Input tensor spec (uses sequence_length)
        self.input_tensor_spec = TensorSpec(
            batch_size=self.config.batch_size,
            time_steps=self.config.sequence_length,
            num_vds=len(self.vd_ids),
            feature_dim=data_info['num_features']
        )
        
        # Target tensor spec (uses prediction_length)
        self.target_tensor_spec = TensorSpec(
            batch_size=self.config.batch_size,
            time_steps=self.config.prediction_length,
            num_vds=len(self.vd_ids),
            feature_dim=data_info['num_features']
        )
        
        logger.info(
            f"Setup complete. VDs: {len(self.vd_ids)}, "
            f"Features: {data_info['num_features']}, "
            f"Sequence length: {self.config.sequence_length}"
        )
    
    def distributed_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for distributed batch format.
        
        Transforms centralized batch tensors [B, T, N, F] into distributed
        format {"VD_001": [B, T, F], "VD_002": [B, T, F], ...}.
        
        Args:
            batch: List of samples from TrafficTimeSeries.__getitem__
            
        Returns:
            Dictionary with distributed format:
            {
                'features': OrderedDict[VD_ID, Tensor[B, T, F]],
                'targets': OrderedDict[VD_ID, Tensor[B, T, F]], 
                'input_time_feat': Tensor[B, T, time_feat_dim],
                'target_time_feat': Tensor[B, T, time_feat_dim],
                'input_mask': OrderedDict[VD_ID, Tensor[B, T]],
                'target_mask': OrderedDict[VD_ID, Tensor[B, T]],
                'vd_ids': List[str],
                'batch_size': int
            }
        """
        if not batch:
            raise ValueError("Empty batch provided to collate_fn")
        
        batch_size = len(batch)
        
        # Stack all samples into centralized tensors [B, T, N, F]
        input_seqs = torch.stack([sample['input_seq'] for sample in batch])
        target_seqs = torch.stack([sample['target_seq'] for sample in batch])
        input_time_feats = torch.stack([sample['input_time_feat'] for sample in batch])
        target_time_feats = torch.stack([sample['target_time_feat'] for sample in batch])
        input_masks = torch.stack([sample['input_mask'] for sample in batch])
        target_masks = torch.stack([sample['target_mask'] for sample in batch])
        
        # Validate centralized tensor format (allow smaller batches for last batch)
        if self.input_tensor_spec is not None and self.target_tensor_spec is not None:
            self.input_tensor_spec.validate_centralized_tensor(input_seqs, "input_seqs", allow_smaller_batch=True)
            self.target_tensor_spec.validate_centralized_tensor(target_seqs, "target_seqs", allow_smaller_batch=True)
        
        # Transform to distributed format: [B, T, N, F] → {"VD_ID": [B, T, F], ...}
        distributed_features = OrderedDict()
        distributed_targets = OrderedDict()
        distributed_input_masks = OrderedDict()
        distributed_target_masks = OrderedDict()
        
        for i, vd_id in enumerate(self.vd_ids):
            # Extract per-VD tensors [B, T, F]
            distributed_features[vd_id] = input_seqs[:, :, i, :]  # [B, T, F]
            distributed_targets[vd_id] = target_seqs[:, :, i, :]   # [B, T, F]
            distributed_input_masks[vd_id] = input_masks[:, :, i]  # [B, T]
            distributed_target_masks[vd_id] = target_masks[:, :, i] # [B, T]
        
        # Ensure deterministic ordering for multi-worker reproducibility
        distributed_features = ensure_deterministic_vd_order(distributed_features)
        distributed_targets = ensure_deterministic_vd_order(distributed_targets)
        distributed_input_masks = ensure_deterministic_vd_order(distributed_input_masks)
        distributed_target_masks = ensure_deterministic_vd_order(distributed_target_masks)
        
        # Validate distributed format
        if self.input_tensor_spec is not None:
            # Update tensor spec for per-VD validation
            vd_spec = TensorSpec(
                batch_size=batch_size,
                time_steps=self.config.sequence_length,
                num_vds=len(self.vd_ids),
                feature_dim=self.input_tensor_spec.feature_dim
            )
            vd_spec.validate_distributed_batch(distributed_features, self.vd_ids, allow_smaller_batch=True)
        
        return {
            'features': distributed_features,
            'targets': distributed_targets,
            'input_time_feat': input_time_feats,   # [B, T, time_feat_dim]
            'target_time_feat': target_time_feats, # [B, T, time_feat_dim]
            'input_mask': distributed_input_masks,
            'target_mask': distributed_target_masks,
            'vd_ids': self.vd_ids,
            'batch_size': batch_size,
            'timestamps': [sample['timestamps'] for sample in batch]
        }
    
    def train_dataloader(self) -> DataLoader:
        """Create distributed training dataloader."""
        return DataLoader(
            self.base_datamodule.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            collate_fn=self.distributed_collate_fn,
            persistent_workers=self.config.num_workers > 0  # Improve multi-worker performance
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create distributed validation dataloader."""
        return DataLoader(
            self.base_datamodule.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            collate_fn=self.distributed_collate_fn,
            persistent_workers=self.config.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create distributed test dataloader."""
        return DataLoader(
            self.base_datamodule.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            collate_fn=self.distributed_collate_fn,
            persistent_workers=self.config.num_workers > 0
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get distributed dataset information."""
        base_info = self.base_datamodule.get_data_info()
        
        # Add distributed-specific information
        base_info.update({
            'distributed_format': True,
            'vd_tensor_shape': (self.config.batch_size, self.config.sequence_length, base_info['num_features']),
            'input_tensor_spec': self.input_tensor_spec,
            'target_tensor_spec': self.target_tensor_spec
        })
        
        return base_info


def create_distributed_datamodule(config: TrafficDatasetConfig) -> DistributedTrafficDataModule:
    """
    Factory function for creating distributed data module.
    
    Args:
        config: Traffic dataset configuration
        
    Returns:
        Configured DistributedTrafficDataModule
    """
    datamodule = DistributedTrafficDataModule(config)
    return datamodule