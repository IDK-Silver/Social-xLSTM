"""
Custom collate functions for Social-xLSTM.

Provides:
- DistributedCollator: 將 centralized [B,T,N,F] 轉成 {vd_id: [B,T,F]}
- CentralizedCollator: 保持 [B,T,N,F]，並附上 vd_ids 與 positions（若可得）
"""

import torch
from torch.utils.data import default_collate
from typing import Dict, List, Any, Union, Optional
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
    
    def __init__(self, vd_ids: List[str], num_features: int, sequence_length: int, prediction_length: int, vd_positions_xy: Dict[str, tuple] | None = None):
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
        # Optional: precomputed static XY positions per VD (meters)
        # Example: {"400201": (x, y), ...}
        self.vd_positions_xy = vd_positions_xy or {}
        
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
        distributed_positions = OrderedDict() if self.vd_positions_xy else None
        
        for i, vd_id in enumerate(self.vd_ids):
            # Extract per-VD tensors [B, T, F]
            distributed_features[vd_id] = input_seqs[:, :, i, :]  # [B, T, F]
            distributed_targets[vd_id] = target_seqs[:, :, i, :]   # [B, T, F]
            distributed_input_masks[vd_id] = input_masks[:, :, i]  # [B, T]
            distributed_target_masks[vd_id] = target_masks[:, :, i] # [B, T]
            # Build per-VD positions [B, T, 2] if XY available (static sensor positions)
            if distributed_positions is not None and vd_id in self.vd_positions_xy:
                x, y = self.vd_positions_xy[vd_id]
                base = torch.tensor([x, y], dtype=torch.float32)
                # shape: [B, T, 2]
                pos = base.view(1, 1, 2).expand(batch_size, self.sequence_length, 2).clone()
                distributed_positions[vd_id] = pos
        
        # Ensure deterministic ordering for multi-worker reproducibility
        distributed_features = ensure_deterministic_vd_order(distributed_features)
        distributed_targets = ensure_deterministic_vd_order(distributed_targets)
        distributed_input_masks = ensure_deterministic_vd_order(distributed_input_masks)
        distributed_target_masks = ensure_deterministic_vd_order(distributed_target_masks)
        if distributed_positions is not None:
            distributed_positions = ensure_deterministic_vd_order(distributed_positions)
        
        # Validate distributed format
        expected_vd_shape = (batch_size, self.sequence_length, self.num_features)
        expected_shapes = {vd_id: expected_vd_shape for vd_id in self.vd_ids}
        assert_dict_tensor_shapes(distributed_features, expected_shapes)
        
        result = {
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
        if distributed_positions is not None:
            result['positions'] = distributed_positions
        return result


class CentralizedCollator:
    """
    保持 centralized 格式，並附加 vd_ids 與 positions_xy（若可得）。
    產出關鍵鍵值：
    - 'features': Tensor[B, T, N, F]
    - 'targets': Tensor[B, P, N, F]
    - 'vd_ids': List[str]
    - 'positions_xy': Tensor[N, 2]（若提供）
    - 其他：時間特徵與遮罩按原樣返回
    """

    def __init__(
        self,
        vd_ids: List[str],
        sequence_length: int,
        prediction_length: int,
        vd_positions_xy: Optional[Dict[str, tuple]] = None,
    ):
        self.vd_ids = list(vd_ids)
        self.sequence_length = int(sequence_length)
        self.prediction_length = int(prediction_length)
        self.vd_positions_xy = vd_positions_xy or {}

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch provided to collate_fn")

        batch_size = len(batch)
        # Use default collate to stack tensors
        stacked = default_collate(batch)

        # Map canonical keys
        features = stacked['input_seq']            # [B, T, N, F]
        targets = stacked['target_seq']            # [B, P, N, F]

        result: Dict[str, Any] = {
            'features': features,
            'targets': targets,
            'input_time_feat': stacked.get('input_time_feat'),
            'target_time_feat': stacked.get('target_time_feat'),
            'input_mask': stacked.get('input_mask'),
            'target_mask': stacked.get('target_mask'),
            'vd_ids': self.vd_ids,
            'batch_size': batch_size,
            'timestamps': [sample['timestamps'] for sample in batch],
        }

        # Attach positions_xy as [N,2] tensor if available
        if self.vd_positions_xy:
            xy = [self.vd_positions_xy.get(vd, (float('nan'), float('nan'))) for vd in self.vd_ids]
            result['positions_xy'] = torch.tensor(xy, dtype=torch.float32)

        return result


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
        # 一律回傳 centralized collator（是否附加 positions 取決於 vd_positions_xy 是否提供）
        return CentralizedCollator(
            vd_ids=kwargs['vd_ids'],
            sequence_length=kwargs['sequence_length'],
            prediction_length=kwargs['prediction_length'],
            vd_positions_xy=kwargs.get('vd_positions_xy')
        )
    else:
        raise ValueError(f"Unknown batch_format: {batch_format}")
