"""
Social Traffic Model - Unified Integration of TrafficLSTM and Social Pooling

This module implements the SocialTrafficModel wrapper that integrates the existing
TrafficLSTM with Social Pooling using an optimized Post-Fusion strategy with 
gated fusion mechanisms.

Based on:
- ADR-0100: Social Pooling vs Graph Networks decision
- ADR-0101: xLSTM vs Traditional LSTM decision  
- Phase 1 Social Pooling implementation

Author: Social-xLSTM Team
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Union, Tuple
import warnings
import logging

from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig
from social_xlstm.models.social_pooling import SocialPooling
from social_xlstm.models.social_pooling_config import SocialPoolingConfig
from social_xlstm.utils.spatial_coords import CoordinateSystem

logger = logging.getLogger(__name__)


class SocialTrafficModelError(Exception):
    """Base exception for Social Traffic Model operations."""
    pass


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining base temporal features with social spatial features.
    
    This implements a learnable gating mechanism that dynamically balances between
    temporal (base) and spatial (social) features, avoiding the dimension bottleneck
    of simple concatenation approaches.
    """
    
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        """
        Args:
            feature_dim: Dimension of both base and social features
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Gate computation network
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        # Optional feature transformation before gating
        self.base_transform = nn.Linear(feature_dim, feature_dim)
        self.social_transform = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, base_features: torch.Tensor, social_features: torch.Tensor) -> torch.Tensor:
        """
        Apply gated fusion to combine base and social features.
        
        Args:
            base_features: Temporal features from base model [batch_size, num_nodes, feature_dim]
            social_features: Spatial features from social pooling [batch_size, num_nodes, feature_dim]
            
        Returns:
            Fused features [batch_size, num_nodes, feature_dim]
        """
        # Transform features
        base_transformed = self.base_transform(base_features)
        social_transformed = self.social_transform(social_features)
        
        # Compute gate weights
        gate_input = torch.cat([base_transformed, social_transformed], dim=-1)
        gate_weights = self.gate_network(gate_input)
        
        # Apply gating: gate controls balance between base and social
        fused = base_transformed * gate_weights + social_transformed * (1 - gate_weights)
        
        # Apply layer normalization
        return self.layer_norm(fused)


class SocialTrafficModel(nn.Module):
    """
    Social Traffic Model - Integration of TrafficLSTM with Social Pooling
    
    This model wraps the existing TrafficLSTM and enhances it with Social Pooling
    for spatial feature aggregation. It uses a Post-Fusion strategy with gated
    fusion mechanisms for optimal performance.
    
    Architecture:
        Input → TrafficLSTM (temporal) → Social Pooling (spatial) → Gated Fusion → Output
    
    Key Features:
    - Preserves existing TrafficLSTM interface
    - Adds minimal computational overhead
    - Uses gated fusion to avoid dimension bottlenecks
    - Supports both single-VD and multi-VD modes
    - Learnable weighting of social influence
    """
    
    def __init__(
        self,
        base_model_config: TrafficLSTMConfig,
        social_pooling_config: SocialPoolingConfig,
        coord_system: Optional[CoordinateSystem] = None,
        fusion_dropout: float = 0.1,
        social_influence_weight: float = 0.3
    ):
        """
        Initialize Social Traffic Model.
        
        Args:
            base_model_config: Configuration for the base TrafficLSTM model
            social_pooling_config: Configuration for Social Pooling
            coord_system: Coordinate system for spatial calculations
            fusion_dropout: Dropout rate in fusion layers
            social_influence_weight: Initial weight for social influence (0-1)
        """
        super().__init__()
        
        # Store configurations
        self.base_config = base_model_config
        self.social_config = social_pooling_config
        self.coord_system = coord_system
        
        # Initialize base TrafficLSTM model
        self.base_model = TrafficLSTM(base_model_config)
        
        # Initialize Social Pooling module
        # Feature dimension matches base model's hidden size
        self.social_pooling = SocialPooling(
            config=social_pooling_config,
            feature_dim=base_model_config.hidden_size,
            coord_system=coord_system
        )
        
        # Initialize gated fusion mechanism
        self.fusion = GatedFusion(
            feature_dim=base_model_config.hidden_size,
            dropout=fusion_dropout
        )
        
        # Learnable social influence weighting
        self.social_weight = nn.Parameter(
            torch.tensor(social_influence_weight, dtype=torch.float32)
        )
        
        # Output projection (matches base model output size)
        self.output_projection = nn.Sequential(
            nn.Linear(base_model_config.hidden_size, base_model_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(base_model_config.hidden_size // 2, base_model_config.output_size)
        )
        
        # Track forward calls for monitoring
        self._forward_calls = 0
        self._social_pooling_enabled = True
        
        logger.info(f"Initialized SocialTrafficModel with social_weight={social_influence_weight}")
    
    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        vd_ids: List[str],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Social Traffic Model.
        
        Args:
            x: Input tensor (same format as TrafficLSTM)
               - Single VD: [batch_size, seq_len, num_features]
               - Multi VD: [batch_size, seq_len, num_vds, num_features]
            coordinates: Node coordinates [num_nodes, 2]
            vd_ids: List of VD identifiers [num_nodes]
            hidden: Optional initial hidden state for LSTM
            return_attention: Whether to return attention weights from social pooling
            
        Returns:
            If return_attention=False: Predictions [batch_size, prediction_length, output_size]
            If return_attention=True: (predictions, attention_weights)
        """
        self._forward_calls += 1
        
        # Step 1: Process through base TrafficLSTM model
        # We need to extract hidden states for social pooling
        batch_size = x.size(0)
        
        if self.base_config.multi_vd_mode:
            # Handle multi-VD mode
            if x.dim() == 4:
                seq_len, num_vds, num_features = x.size(1), x.size(2), x.size(3)
                x_flat = x.view(batch_size, seq_len, num_vds * num_features)
            else:
                x_flat = x
                seq_len = x.size(1)
        else:
            # Single VD mode
            x_flat = x
            seq_len = x.size(1)
        
        # Get LSTM hidden states
        lstm_output, (h_n, c_n) = self.base_model.lstm(x_flat, hidden)
        
        # Use last timestep hidden state for social pooling
        # h_n shape: [num_layers, batch_size, hidden_size]
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # Step 2: Apply Social Pooling if enabled
        if self._social_pooling_enabled and coordinates is not None:
            # Expand hidden states to match spatial dimensions
            if self.base_config.multi_vd_mode:
                # For multi-VD, we need to map hidden states to VD level
                num_nodes = len(vd_ids)
                
                # Create node-level features by averaging or projecting
                # Simple approach: expand the hidden state for each VD
                node_features = last_hidden.unsqueeze(1).expand(
                    batch_size, num_nodes, self.base_config.hidden_size
                )
            else:
                # Single VD: expand to match coordinate dimensions
                num_nodes = len(vd_ids)
                node_features = last_hidden.unsqueeze(1).expand(
                    batch_size, num_nodes, self.base_config.hidden_size
                )
            
            # Apply social pooling with optional attention return
            if return_attention:
                social_features, attention_weights = self.social_pooling(
                    node_features, coordinates, vd_ids, return_weights=True
                )
            else:
                social_features = self.social_pooling(
                    node_features, coordinates, vd_ids, return_weights=False
                )
                attention_weights = None
            
            # Apply social influence weighting
            social_features = social_features * torch.sigmoid(self.social_weight)
            
            # Step 3: Fuse temporal and spatial features
            fused_features = self.fusion(node_features, social_features)
            
            # Aggregate back to sequence level for final prediction
            # Take mean across spatial dimension
            final_features = fused_features.mean(dim=1)  # [batch_size, hidden_size]
            
        else:
            # No social pooling - use base features only
            final_features = last_hidden
            attention_weights = None
        
        # Step 4: Generate final predictions
        predictions = self.output_projection(final_features)  # [batch_size, output_size]
        
        # Expand to prediction_length if needed
        if self.base_config.prediction_length > 1:
            predictions = predictions.unsqueeze(1).repeat(1, self.base_config.prediction_length, 1)
        else:
            predictions = predictions.unsqueeze(1)  # [batch_size, 1, output_size]
        
        if return_attention and attention_weights is not None:
            return predictions, attention_weights
        return predictions
    
    def forward_base_only(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass using only the base model (for comparison/ablation studies).
        
        Args:
            x: Input tensor
            hidden: Optional initial hidden state
            
        Returns:
            Base model predictions
        """
        return self.base_model(x, hidden)
    
    def enable_social_pooling(self, enabled: bool = True):
        """Enable or disable social pooling."""
        self._social_pooling_enabled = enabled
        logger.info(f"Social pooling {'enabled' if enabled else 'disabled'}")
    
    def get_social_influence_weight(self) -> float:
        """Get current social influence weight."""
        return torch.sigmoid(self.social_weight).item()
    
    def set_social_influence_weight(self, weight: float):
        """Set social influence weight (0-1 range)."""
        if not 0 <= weight <= 1:
            raise ValueError(f"Social influence weight must be in [0, 1], got {weight}")
        
        # Convert to logit space for unconstrained optimization
        logit_weight = torch.log(torch.tensor(weight) / (1 - weight + 1e-8))
        self.social_weight.data = logit_weight
        
    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        base_info = self.base_model.get_model_info()
        social_stats = self.social_pooling.get_performance_stats()
        
        total_params = sum(p.numel() for p in self.parameters())
        social_params = sum(p.numel() for p in self.social_pooling.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        
        return {
            'model_type': 'SocialTrafficModel',
            'base_model_info': base_info,
            'social_config': self.social_config.__dict__,
            'total_parameters': total_params,
            'social_pooling_parameters': social_params,
            'fusion_parameters': fusion_params,
            'parameter_overhead': (total_params - base_info['total_parameters']) / base_info['total_parameters'],
            'social_influence_weight': self.get_social_influence_weight(),
            'forward_calls': self._forward_calls,
            'social_pooling_stats': social_stats,
            'social_pooling_enabled': self._social_pooling_enabled
        }
    
    @classmethod
    def create_from_base_model(
        cls,
        base_model: TrafficLSTM,
        social_pooling_config: SocialPoolingConfig,
        coord_system: Optional[CoordinateSystem] = None,
        **kwargs
    ) -> 'SocialTrafficModel':
        """
        Create SocialTrafficModel from existing TrafficLSTM instance.
        
        Args:
            base_model: Existing TrafficLSTM model
            social_pooling_config: Social pooling configuration
            coord_system: Coordinate system
            **kwargs: Additional arguments for SocialTrafficModel
            
        Returns:
            SocialTrafficModel wrapping the base model
        """
        # Create new instance with base model's config
        social_model = cls(
            base_model_config=base_model.config,
            social_pooling_config=social_pooling_config,
            coord_system=coord_system,
            **kwargs
        )
        
        # Copy base model parameters
        social_model.base_model.load_state_dict(base_model.state_dict())
        
        logger.info("Created SocialTrafficModel from existing base model")
        return social_model
    
    @classmethod
    def create_single_vd_social_model(
        cls,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        social_config: Optional[SocialPoolingConfig] = None,
        **kwargs
    ) -> 'SocialTrafficModel':
        """
        Factory method to create single VD social model with defaults.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            social_config: Social pooling config (uses urban preset if None)
            **kwargs: Additional arguments
            
        Returns:
            Configured SocialTrafficModel for single VD prediction
        """
        base_config = TrafficLSTMConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            multi_vd_mode=False
        )
        
        if social_config is None:
            social_config = SocialPoolingConfig.urban_preset()
        
        return cls(
            base_model_config=base_config,
            social_pooling_config=social_config,
            **kwargs
        )


# Factory functions for easy model creation

def create_social_traffic_model(
    scenario: str = "urban",
    base_hidden_size: int = 128,
    base_num_layers: int = 2,
    social_config_overrides: Optional[Dict] = None,
    **kwargs
) -> SocialTrafficModel:
    """
    Factory function to create Social Traffic Model with preset configurations.
    
    Args:
        scenario: Preset scenario ("urban", "highway", "mixed")
        base_hidden_size: Base model hidden size
        base_num_layers: Base model number of layers
        social_config_overrides: Overrides for social pooling config
        **kwargs: Additional model arguments
        
    Returns:
        Configured SocialTrafficModel
    """
    # Create base model config
    base_config = TrafficLSTMConfig(
        hidden_size=base_hidden_size,
        num_layers=base_num_layers,
        multi_vd_mode=False
    )
    
    # Create social pooling config with scenario preset
    social_overrides = social_config_overrides or {}
    
    if scenario == "urban":
        social_config = SocialPoolingConfig.urban_preset(**social_overrides)
    elif scenario == "highway":
        social_config = SocialPoolingConfig.highway_preset(**social_overrides)
    elif scenario == "mixed":
        social_config = SocialPoolingConfig.mixed_preset(**social_overrides)
    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: urban, highway, mixed")
    
    return SocialTrafficModel(
        base_model_config=base_config,
        social_pooling_config=social_config,
        **kwargs
    )


# Export public interface
__all__ = [
    "SocialTrafficModel",
    "GatedFusion", 
    "SocialTrafficModelError",
    "create_social_traffic_model",
]