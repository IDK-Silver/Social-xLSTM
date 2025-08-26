"""
Custom collate functions for distributed training.

This module provides collate functions that transform standard centralized
[B, T, N, F] tensors into distributed format {"VD_ID": [B, T, F], ...}.
"""

import torch
from torch.utils.data import default_collate
from typing import Dict, List, Any, Union
from collections import OrderedDict
import logging

try:
    from ...utils.tensor_checks import assert_shape, assert_dict_tensor_shapes, ensure_deterministic_vd_order
except ImportError:
    # Fallback for standalone testing
    def assert_shape(tensor, expected_shape, name):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} != expected {expected_shape}")
    
    def assert_dict_tensor_shapes(tensor_dict, expected_shapes):
        for key, tensor in tensor_dict.items():
            if key in expected_shapes:
                expected = expected_shapes[key]
                if tensor.shape != expected:
                    raise ValueError(f"{key} shape {tensor.shape} != expected {expected}")
    
    def ensure_deterministic_vd_order(ordered_dict):
        """Ensure deterministic ordering (OrderedDict should already be ordered)."""
        return ordered_dict

logger = logging.getLogger(__name__)


class DistributedCollator:
    """
    Collate function that transforms centralized batches to distributed format.
    
    This class is designed to be pickle-safe for multi-process DataLoaders.
    """
    
    def __init__(self, vd_ids: List[str], num_features: int, sequence_length: int, prediction_length: int):
        """
        Initialize the distributed collator.
        
        Args:
            vd_ids: List of VD identifiers in the order they appear in the N dimension
            num_features: Number of features per VD (F dimension)
            sequence_length: Input sequence length (T dimension)
            prediction_length: Prediction sequence length
        """
        self.vd_ids = list(vd_ids)  # Ensure immutable copy
        self.num_features = int(num_features)
        self.sequence_length = int(sequence_length)
        self.prediction_length = int(prediction_length)
        self.num_vds = len(self.vd_ids)
        
        logger.debug(
            f"DistributedCollator initialized: "
            f"VDs={self.num_vds}, Features={self.num_features}, "
            f"SeqLen={self.sequence_length}, PredLen={self.prediction_length}"
        )
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        
        # Validate centralized tensor format
        expected_input_shape = (batch_size, self.sequence_length, self.num_vds, self.num_features)
        expected_target_shape = (batch_size, self.prediction_length, self.num_vds, self.num_features)
        
        assert_shape(input_seqs, expected_input_shape, "input_seqs")
        assert_shape(target_seqs, expected_target_shape, "target_seqs")
        
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
        expected_vd_shape = (batch_size, self.sequence_length, self.num_features)
        expected_shapes = {vd_id: expected_vd_shape for vd_id in self.vd_ids}
        assert_dict_tensor_shapes(distributed_features, expected_shapes)
        
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


def create_collate_fn(batch_format: str, **kwargs) -> callable:
    """
    Factory function for creating appropriate collate function.
    
    Args:
        batch_format: 'centralized' or 'distributed'
        **kwargs: Additional arguments for collate function
        
    Returns:
        Appropriate collate function
    """
    if batch_format == 'distributed':
        return DistributedCollator(**kwargs)
    elif batch_format == 'centralized':
        return None  # Use PyTorch default collate
    else:
        raise ValueError(f"Unknown batch_format: {batch_format}")