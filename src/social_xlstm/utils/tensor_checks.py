"""
Minimal tensor validation utilities.

Replaces the complex TensorSpec system with simple, direct shape checking.
"""

from typing import Dict
import torch
from collections import OrderedDict


def assert_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """
    Assert tensor has expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        name: Tensor name for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"{name}: expected {expected_shape}, got {tensor.shape}")


def assert_dict_tensor_shapes(tensor_dict: Dict[str, torch.Tensor], expected_shapes: Dict[str, tuple]):
    """
    Assert all tensors in dict have expected shapes.
    
    Args:
        tensor_dict: Dictionary of tensors to validate
        expected_shapes: Dictionary mapping keys to expected shapes
        
    Raises:
        ValueError: If any tensor shape doesn't match
    """
    for key, tensor in tensor_dict.items():
        if key in expected_shapes:
            assert_shape(tensor, expected_shapes[key], name=key)


def ensure_deterministic_vd_order(batch_dict: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """
    Ensure deterministic VD ordering for distributed workers.
    
    Args:
        batch_dict: Input VD batch dictionary
        
    Returns:
        OrderedDict with sorted VD keys
    """
    return OrderedDict(sorted(batch_dict.items()))