"""
Tensor specification validation for distributed xLSTM architecture.

This module provides strict B T N F tensor specification validation to prevent
>50% of runtime bugs according to OpenAI o3-pro recommendations.

Usage:
```python
from social_xlstm.interfaces.tensor_spec import TensorSpec, validate_distributed_batch

spec = TensorSpec(batch_size=4, time_steps=12, num_vds=3, feature_dim=5)
batch_dict = {"VD_001": torch.randn(4, 12, 5), "VD_002": torch.randn(4, 12, 5)}
spec.validate_distributed_batch(batch_dict)
```
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import OrderedDict


@dataclass
class TensorSpec:
    """
    B T N F tensor specification validator.
    
    Dimensions:
    - B: Batch Size (批次大小)
    - T: Time Steps (時間步長) 
    - N: Number of Nodes/VDs (虛擬偵測器數量)
    - F: Feature Dimension (特徵維度)
    """
    batch_size: int
    time_steps: int
    num_vds: int
    feature_dim: int
    
    def validate_centralized_tensor(self, tensor: torch.Tensor, name: str = "tensor", allow_smaller_batch: bool = True) -> None:
        """
        Validate centralized tensor format [B, T, N, F].
        
        Args:
            tensor: Input tensor to validate
            name: Tensor name for error messages
            allow_smaller_batch: If True, allows batch size smaller than expected (for last batches)
            
        Raises:
            ValueError: If tensor shape doesn't match B T N F spec
        """
        expected_shape = (self.batch_size, self.time_steps, self.num_vds, self.feature_dim)
        actual_batch_size = tensor.shape[0] if len(tensor.shape) > 0 else 0
        
        # Check batch dimension with flexibility for smaller batches
        if allow_smaller_batch:
            if actual_batch_size > self.batch_size or actual_batch_size <= 0:
                raise ValueError(
                    f"Tensor '{name}' batch size out of range. "
                    f"Expected: 1 <= batch_size <= {self.batch_size}, "
                    f"Got: {actual_batch_size}"
                )
        else:
            if actual_batch_size != self.batch_size:
                raise ValueError(
                    f"Tensor '{name}' batch size mismatch. "
                    f"Expected: {self.batch_size}, Got: {actual_batch_size}"
                )
        
        # Check other dimensions strictly
        if len(tensor.shape) != 4:
            raise ValueError(
                f"Tensor '{name}' must be 4D [B, T, N, F]. Got {len(tensor.shape)}D: {tensor.shape}"
            )
            
        if tensor.shape[1:] != (self.time_steps, self.num_vds, self.feature_dim):
            raise ValueError(
                f"Tensor '{name}' dimensions mismatch. "
                f"Expected: (?, {self.time_steps}, {self.num_vds}, {self.feature_dim}) "
                f"(B=flexible, T={self.time_steps}, N={self.num_vds}, F={self.feature_dim}), "
                f"Got: {tensor.shape}"
            )
    
    def validate_distributed_batch(
        self, 
        batch_dict: Dict[str, torch.Tensor], 
        expected_vd_ids: Optional[List[str]] = None,
        allow_smaller_batch: bool = True
    ) -> None:
        """
        Validate distributed batch format {"VD_001": [B, T, F], ...}.
        
        Args:
            batch_dict: Dictionary mapping VD_ID to tensors
            expected_vd_ids: Expected VD IDs (optional)
            allow_smaller_batch: If True, allows batch size smaller than expected (for last batches)
            
        Raises:
            ValueError: If batch format doesn't match distributed spec
        """
        if not isinstance(batch_dict, (dict, OrderedDict)):
            raise ValueError(f"batch_dict must be dict or OrderedDict, got {type(batch_dict)}")
        
        if len(batch_dict) != self.num_vds:
            raise ValueError(
                f"VD count mismatch. Expected: {self.num_vds}, Got: {len(batch_dict)}"
            )
        
        # Validate expected VD IDs if provided
        if expected_vd_ids is not None:
            missing_vds = set(expected_vd_ids) - set(batch_dict.keys())
            if missing_vds:
                raise ValueError(f"Missing VD IDs: {missing_vds}")
            
            extra_vds = set(batch_dict.keys()) - set(expected_vd_ids)
            if extra_vds:
                raise ValueError(f"Unexpected VD IDs: {extra_vds}")
        
        # Get the actual batch size from the first tensor
        if batch_dict:
            first_tensor = next(iter(batch_dict.values()))
            actual_batch_size = first_tensor.shape[0] if len(first_tensor.shape) > 0 else 0
        else:
            actual_batch_size = 0
        
        # Validate batch size with flexibility for smaller batches
        if allow_smaller_batch:
            if actual_batch_size > self.batch_size or actual_batch_size <= 0:
                raise ValueError(
                    f"Batch size out of range. "
                    f"Expected: 1 <= batch_size <= {self.batch_size}, "
                    f"Got: {actual_batch_size}"
                )
        else:
            if actual_batch_size != self.batch_size:
                raise ValueError(
                    f"Batch size mismatch. "
                    f"Expected: {self.batch_size}, Got: {actual_batch_size}"
                )
        
        # Validate each VD tensor shape [B, T, F] with flexible batch size
        expected_vd_shape = (actual_batch_size, self.time_steps, self.feature_dim)
        
        for vd_id, tensor in batch_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"VD '{vd_id}' must be torch.Tensor, got {type(tensor)}")
            
            if tensor.shape != expected_vd_shape:
                raise ValueError(
                    f"VD '{vd_id}' tensor shape mismatch. "
                    f"Expected: {expected_vd_shape} (B={actual_batch_size}, T={self.time_steps}, "
                    f"F={self.feature_dim}), "
                    f"Got: {tensor.shape}"
                )
    
    def validate_hidden_states(self, hidden_states: Dict[str, torch.Tensor]) -> None:
        """
        Validate hidden states format for Social Pooling.
        
        Args:
            hidden_states: Mapping from VD_ID to hidden state tensors [T, H]
            
        Raises:
            ValueError: If hidden states format is invalid
        """
        if not isinstance(hidden_states, (dict, OrderedDict)):
            raise ValueError(f"hidden_states must be dict or OrderedDict, got {type(hidden_states)}")
        
        if not hidden_states:
            raise ValueError("hidden_states cannot be empty")
        
        # Get reference tensor to check consistency
        first_vd = next(iter(hidden_states.keys()))
        first_tensor = hidden_states[first_vd]
        
        if len(first_tensor.shape) != 2:
            raise ValueError(f"Hidden state tensors must be 2D [T, H], got shape {first_tensor.shape}")
        
        if first_tensor.shape[0] != self.time_steps:
            raise ValueError(
                f"Hidden state time dimension mismatch. "
                f"Expected T={self.time_steps}, got {first_tensor.shape[0]}"
            )
        
        hidden_dim = first_tensor.shape[1]
        
        # Validate all VD hidden states have consistent shapes
        for vd_id, tensor in hidden_states.items():
            expected_shape = (self.time_steps, hidden_dim)
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"VD '{vd_id}' hidden state shape mismatch. "
                    f"Expected: {expected_shape}, Got: {tensor.shape}"
                )


def validate_distributed_batch(
    batch_dict: Dict[str, torch.Tensor],
    batch_size: int,
    time_steps: int, 
    feature_dim: int,
    expected_vd_ids: Optional[List[str]] = None
) -> None:
    """
    Convenience function for validating distributed batch format.
    
    Args:
        batch_dict: Dictionary mapping VD_ID to tensors [B, T, F]
        batch_size: Expected batch size
        time_steps: Expected time steps
        feature_dim: Expected feature dimension
        expected_vd_ids: Expected VD IDs (optional)
    """
    num_vds = len(expected_vd_ids) if expected_vd_ids else len(batch_dict)
    spec = TensorSpec(
        batch_size=batch_size,
        time_steps=time_steps,
        num_vds=num_vds,
        feature_dim=feature_dim
    )
    spec.validate_distributed_batch(batch_dict, expected_vd_ids)


def ensure_deterministic_vd_order(batch_dict: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """
    Ensure deterministic VD ordering for distributed workers.
    
    According to OpenAI o3-pro recommendations, use OrderedDict with 
    explicit sorting to maintain reproducibility across ranks.
    
    Args:
        batch_dict: Input VD batch dictionary
        
    Returns:
        OrderedDict with sorted VD keys
    """
    return OrderedDict(sorted(batch_dict.items()))