"""
VD xLSTM Manager for distributed Social-xLSTM architecture.

This module manages per-VD xLSTM instances using torch.nn.ModuleDict,
implementing lazy initialization and proper device management according
to OpenAI o3-pro recommendations.

Key Features:
- Lazy initialization for memory efficiency
- Automatic parameter registration via ModuleDict
- Device management with unified .to(device) handling
- Support for dynamic VD addition/removal
- Mixed precision (AMP) and gradient checkpointing ready

Usage:
```python
from social_xlstm.models.vd_xlstm_manager import VDXLSTMManager

manager = VDXLSTMManager(xlstm_config, vd_ids=["VD_001", "VD_002"])
batch_dict = {"VD_001": tensor1, "VD_002": tensor2}
hidden_states = manager(batch_dict)  # {"VD_001": hidden1, "VD_002": hidden2}
```
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from collections import OrderedDict
import logging

from .xlstm import TrafficXLSTM, TrafficXLSTMConfig
from ..interfaces.tensor_spec import TensorSpec, ensure_deterministic_vd_order

logger = logging.getLogger(__name__)


class VDXLSTMManager(nn.Module):
    """
    Manager for per-VD xLSTM instances in distributed Social-xLSTM architecture.
    
    This class manages individual xLSTM models for each VD using torch.nn.ModuleDict,
    ensuring proper parameter registration, device handling, and memory efficiency
    through lazy initialization.
    """
    
    def __init__(
        self,
        xlstm_config: TrafficXLSTMConfig,
        vd_ids: Optional[List[str]] = None,
        lazy_init: bool = True,
        enable_gradient_checkpointing: bool = True
    ):
        """
        Initialize VD xLSTM Manager.
        
        Args:
            xlstm_config: Configuration for individual xLSTM models
            vd_ids: List of VD IDs to initialize (optional for lazy init)
            lazy_init: Whether to use lazy initialization for memory efficiency
            enable_gradient_checkpointing: Enable gradient checkpointing for memory savings
        """
        super().__init__()
        
        self.xlstm_config = xlstm_config
        self.lazy_init = lazy_init
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        
        # Use ModuleDict for automatic parameter registration
        self.vd_models: nn.ModuleDict = nn.ModuleDict()
        
        # Track initialized VDs
        self.initialized_vds: set = set()
        
        # Pre-initialize VDs if provided and not using lazy init
        if vd_ids is not None and not lazy_init:
            self._initialize_vds(vd_ids)
        
        logger.info(
            f"VDXLSTMManager initialized. "
            f"Lazy init: {lazy_init}, "
            f"Gradient checkpointing: {enable_gradient_checkpointing}, "
            f"Pre-initialized VDs: {len(self.vd_models)}"
        )
    
    def _create_xlstm_model(self, vd_id: str) -> TrafficXLSTM:
        """
        Create a new xLSTM model instance for a VD.
        
        Args:
            vd_id: VD identifier
            
        Returns:
            Configured TrafficXLSTM model
        """
        model = TrafficXLSTM(self.xlstm_config)
        
        # Enable gradient checkpointing for memory efficiency
        if self.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
        
        logger.debug(f"Created xLSTM model for VD: {vd_id}")
        return model
    
    def _initialize_vd(self, vd_id: str) -> None:
        """
        Initialize a single VD model (lazy initialization).
        
        Args:
            vd_id: VD identifier to initialize
        """
        if vd_id not in self.vd_models:
            self.vd_models[vd_id] = self._create_xlstm_model(vd_id)
            self.initialized_vds.add(vd_id)
            logger.debug(f"Lazy initialized VD: {vd_id}")
    
    def _initialize_vds(self, vd_ids: List[str]) -> None:
        """
        Initialize multiple VD models.
        
        Args:
            vd_ids: List of VD identifiers to initialize
        """
        for vd_id in vd_ids:
            if vd_id not in self.vd_models:
                self.vd_models[vd_id] = self._create_xlstm_model(vd_id)
                self.initialized_vds.add(vd_id)
        
        logger.info(f"Initialized {len(vd_ids)} VD models")
    
    def add_vd(self, vd_id: str) -> None:
        """
        Dynamically add a new VD model.
        
        Args:
            vd_id: VD identifier to add
        """
        if vd_id in self.vd_models:
            logger.warning(f"VD {vd_id} already exists, skipping")
            return
        
        self._initialize_vd(vd_id)
        logger.info(f"Added new VD: {vd_id}")
    
    def remove_vd(self, vd_id: str) -> None:
        """
        Remove a VD model.
        
        Args:
            vd_id: VD identifier to remove
        """
        if vd_id in self.vd_models:
            del self.vd_models[vd_id]
            self.initialized_vds.discard(vd_id)
            logger.info(f"Removed VD: {vd_id}")
        else:
            logger.warning(f"VD {vd_id} not found, cannot remove")
    
    def get_vd_ids(self) -> List[str]:
        """Get list of all initialized VD IDs."""
        return list(self.vd_models.keys())
    
    def get_model_count(self) -> int:
        """Get number of initialized VD models."""
        return len(self.vd_models)
    
    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all VD models.
        
        Args:
            batch_dict: Dictionary mapping VD_ID to input tensors [B, T, F]
            
        Returns:
            Dictionary mapping VD_ID to hidden states [B, T, H]
            
        Raises:
            KeyError: If VD model not found and lazy_init is False
            ValueError: If batch_dict is empty or contains invalid tensors
        """
        if not batch_dict:
            raise ValueError("batch_dict cannot be empty")
        
        # Ensure deterministic processing order
        if not isinstance(batch_dict, OrderedDict):
            batch_dict = ensure_deterministic_vd_order(batch_dict)
        
        hidden_states = OrderedDict()
        
        for vd_id, input_tensor in batch_dict.items():
            # Validate input tensor
            if not isinstance(input_tensor, torch.Tensor):
                raise ValueError(f"Input for VD '{vd_id}' must be torch.Tensor, got {type(input_tensor)}")
            
            # Lazy initialization if needed
            if self.lazy_init and vd_id not in self.vd_models:
                self._initialize_vd(vd_id)
            
            # Get VD model
            if vd_id not in self.vd_models:
                raise KeyError(f"VD model '{vd_id}' not found and lazy_init is disabled")
            
            vd_model = self.vd_models[vd_id]
            
            # Ensure model is on same device as input tensor
            if input_tensor.device != next(vd_model.parameters()).device:
                vd_model = vd_model.to(input_tensor.device)
            
            # Forward pass through VD model to get hidden states
            try:
                hidden_state = vd_model.get_hidden_states(input_tensor)  # [B, T, H]
                hidden_states[vd_id] = hidden_state
            except Exception as e:
                logger.error(f"Forward pass failed for VD '{vd_id}': {e}")
                raise RuntimeError(f"VD '{vd_id}' forward pass failed") from e
        
        return hidden_states
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for all VD models.
        
        Returns:
            Dictionary with memory usage information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_params = {}
        for vd_id, model in self.vd_models.items():
            model_params[vd_id] = sum(p.numel() for p in model.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_models': len(self.vd_models),
            'per_model_parameters': model_params,
            'initialized_vds': list(self.initialized_vds)
        }
    
    def enable_gradient_checkpointing_all(self) -> None:
        """Enable gradient checkpointing for all VD models."""
        for vd_id, model in self.vd_models.items():
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
        
        self.enable_gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for all VD models")
    
    def disable_gradient_checkpointing_all(self) -> None:
        """Disable gradient checkpointing for all VD models."""
        for vd_id, model in self.vd_models.items():
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            elif hasattr(model, 'disable_gradient_checkpointing'):
                model.disable_gradient_checkpointing()
        
        self.enable_gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for all VD models")
    
    def __repr__(self) -> str:
        return (
            f"VDXLSTMManager("
            f"models={len(self.vd_models)}, "
            f"lazy_init={self.lazy_init}, "
            f"gradient_checkpointing={self.enable_gradient_checkpointing})"
        )