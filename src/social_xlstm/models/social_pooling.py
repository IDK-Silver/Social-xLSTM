"""
Social Pooling Module

This module implements the core Social Pooling mechanism for Social-xLSTM.
It provides coordinate-driven spatial aggregation without requiring predefined
graph topology, making it suitable for irregular traffic detector networks.

Author: Social-xLSTM Team
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import warnings

from social_xlstm.models.social_pooling_config import SocialPoolingConfig
from social_xlstm.models.distance_functions import SpatialCalculator
from social_xlstm.utils.spatial_coords import CoordinateSystem


class SocialPoolingError(Exception):
    """Base exception for Social Pooling operations."""
    pass


class InvalidInputError(SocialPoolingError):
    """Raised when input tensors have invalid shapes or values."""
    pass


class SocialPooling(nn.Module):
    """
    Social Pooling layer for spatial feature aggregation in traffic prediction.
    
    This module implements coordinate-driven spatial pooling that aggregates
    features from neighboring nodes based on their geographical proximity,
    without requiring predefined graph topology.
    
    The implementation follows the Post-Fusion strategy outlined in the design
    documents, providing a clean separation between individual node processing
    and spatial aggregation.
    
    Architecture:
        Input Features -> Spatial Weight Computation -> Neighbor Aggregation -> Output
    
    Attributes:
        config (SocialPoolingConfig): Configuration for spatial pooling
        spatial_calculator (SpatialCalculator): Handles distance and weight calculations
        aggregation_layer (nn.Module): Optional learned aggregation layer
    """
    
    def __init__(
        self, 
        config: SocialPoolingConfig,
        feature_dim: int,
        coord_system: Optional[CoordinateSystem] = None,
        learnable_aggregation: bool = False
    ):
        """
        Initialize Social Pooling layer.
        
        Args:
            config: Social pooling configuration
            feature_dim: Dimension of input features
            coord_system: Coordinate system for spatial calculations
            learnable_aggregation: Whether to use learnable aggregation weights
            
        Raises:
            ValueError: If feature_dim is not positive
        """
        super().__init__()
        
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        
        self.config = config
        self.feature_dim = feature_dim
        
        # Initialize spatial calculator
        self.spatial_calculator = SpatialCalculator(config, coord_system)
        
        # Optional learnable aggregation layer
        if learnable_aggregation:
            self.aggregation_layer = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            self.aggregation_layer = nn.Identity()
        
        # Track statistics for monitoring
        self._forward_calls = 0
        self._total_aggregation_time = 0.0
    
    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        vd_ids: List[str],
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Social Pooling layer.
        
        Args:
            features: Node features [batch_size, num_nodes, feature_dim] or [num_nodes, feature_dim]
            coordinates: Node coordinates [num_nodes, 2] in format specified by config.coordinate_system
            vd_ids: List of vehicle detector IDs [num_nodes]
            return_weights: Whether to return spatial weights for analysis
            
        Returns:
            If return_weights=False: Pooled features with same shape as input features
            If return_weights=True: Tuple of (pooled_features, spatial_weights)
            
        Raises:
            InvalidInputError: If input shapes are incompatible
        """
        self._forward_calls += 1
        
        # Validate inputs
        self._validate_inputs(features, coordinates, vd_ids)
        
        # Handle batch dimension
        original_shape = features.shape
        if features.dim() == 3:
            # Batch processing: [batch_size, num_nodes, feature_dim]
            batch_size, num_nodes, feature_dim = features.shape
            features_2d = features.view(-1, feature_dim)  # [batch_size * num_nodes, feature_dim]
            process_batched = True
        else:
            # Single sequence: [num_nodes, feature_dim]
            num_nodes, feature_dim = features.shape
            features_2d = features
            batch_size = 1
            process_batched = False
        
        # Compute spatial weights
        spatial_weights = self._compute_spatial_weights(coordinates, vd_ids)
        
        # Perform spatial aggregation
        if process_batched:
            pooled_features = self._aggregate_features_batched(
                features_2d, spatial_weights, batch_size, num_nodes
            )
            pooled_features = pooled_features.view(original_shape)
        else:
            pooled_features = self._aggregate_features_single(features_2d, spatial_weights)
        
        # Apply optional learnable aggregation
        pooled_features = self.aggregation_layer(pooled_features)
        
        if return_weights:
            return pooled_features, spatial_weights
        return pooled_features
    
    def _validate_inputs(
        self, 
        features: torch.Tensor, 
        coordinates: torch.Tensor, 
        vd_ids: List[str]
    ) -> None:
        """Validate input tensor shapes and values."""
        
        # Check feature tensor
        if features.dim() not in [2, 3]:
            raise InvalidInputError(
                f"features must be 2D [num_nodes, feature_dim] or "
                f"3D [batch_size, num_nodes, feature_dim], got {features.dim()}D"
            )
        
        # Check coordinate tensor
        if coordinates.dim() != 2 or coordinates.shape[1] != 2:
            raise InvalidInputError(
                f"coordinates must be [num_nodes, 2], got {coordinates.shape}"
            )
        
        # Check VD IDs
        num_nodes = coordinates.shape[0]
        if len(vd_ids) != num_nodes:
            raise InvalidInputError(
                f"vd_ids length ({len(vd_ids)}) must match num_nodes ({num_nodes})"
            )
        
        # Check feature dimension compatibility
        expected_feature_dim = features.shape[-1]
        if expected_feature_dim != self.feature_dim:
            raise InvalidInputError(
                f"Expected feature_dim {self.feature_dim}, got {expected_feature_dim}"
            )
        
        # Check for invalid values
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            raise InvalidInputError("features contains NaN or infinite values")
        
        if torch.any(torch.isnan(coordinates)) or torch.any(torch.isinf(coordinates)):
            raise InvalidInputError("coordinates contains NaN or infinite values")
    
    def _compute_spatial_weights(
        self, 
        coordinates: torch.Tensor, 
        vd_ids: List[str]
    ) -> torch.Tensor:
        """
        Compute spatial weights from coordinates.
        
        Args:
            coordinates: Node coordinates [num_nodes, 2]
            vd_ids: Vehicle detector IDs
            
        Returns:
            Spatial weight matrix [num_nodes, num_nodes]
        """
        # Compute distance matrix (with caching)
        distance_matrix = self.spatial_calculator.compute_distance_matrix(
            coordinates, vd_ids
        )
        
        # Compute spatial weights from distances
        spatial_weights = self.spatial_calculator.compute_spatial_weights(distance_matrix)
        
        # Apply neighbor limit constraint
        spatial_weights = self.spatial_calculator.apply_neighbor_limit(spatial_weights)
        
        # Normalize weights (row-wise normalization for proper averaging)
        spatial_weights = self._normalize_weights(spatial_weights)
        
        return spatial_weights
    
    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize spatial weights based on aggregation method.
        
        Args:
            weights: Raw spatial weights [num_nodes, num_nodes]
            
        Returns:
            Normalized weights [num_nodes, num_nodes]
        """
        if self.config.aggregation_method == "weighted_mean":
            # Row-wise normalization (each row sums to 1)
            row_sums = weights.sum(dim=1, keepdim=True)
            # Avoid division by zero
            row_sums = torch.clamp(row_sums, min=1e-8)
            normalized_weights = weights / row_sums
            
        elif self.config.aggregation_method == "weighted_sum":
            # No normalization for weighted sum
            normalized_weights = weights
            
        elif self.config.aggregation_method == "attention":
            # Softmax normalization (attention mechanism)
            normalized_weights = F.softmax(weights, dim=1)
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
        
        return normalized_weights
    
    def _aggregate_features_single(
        self, 
        features: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate features for single sequence (no batch dimension).
        
        Args:
            features: Node features [num_nodes, feature_dim]
            weights: Normalized spatial weights [num_nodes, num_nodes]
            
        Returns:
            Aggregated features [num_nodes, feature_dim]
        """
        # Matrix multiplication: [num_nodes, num_nodes] @ [num_nodes, feature_dim]
        # -> [num_nodes, feature_dim]
        aggregated = torch.matmul(weights, features)
        return aggregated
    
    def _aggregate_features_batched(
        self, 
        features: torch.Tensor, 
        weights: torch.Tensor,
        batch_size: int,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate features for batched input.
        
        Args:
            features: Flattened features [batch_size * num_nodes, feature_dim]
            weights: Spatial weights [num_nodes, num_nodes] 
            batch_size: Number of sequences in batch
            num_nodes: Number of nodes per sequence
            
        Returns:
            Aggregated features [batch_size * num_nodes, feature_dim]
        """
        feature_dim = features.shape[1]
        
        # Reshape features to [batch_size, num_nodes, feature_dim]
        features_3d = features.view(batch_size, num_nodes, feature_dim)
        
        # Apply spatial weights to each sequence in the batch
        # weights: [num_nodes, num_nodes]
        # features_3d: [batch_size, num_nodes, feature_dim]
        # Result: [batch_size, num_nodes, feature_dim]
        aggregated_3d = torch.matmul(weights.unsqueeze(0), features_3d)
        
        # Flatten back to [batch_size * num_nodes, feature_dim]
        aggregated = aggregated_3d.view(-1, feature_dim)
        
        return aggregated
    
    def get_spatial_weights(
        self, 
        coordinates: torch.Tensor, 
        vd_ids: List[str]
    ) -> torch.Tensor:
        """
        Get spatial weights without processing features (useful for analysis).
        
        Args:
            coordinates: Node coordinates [num_nodes, 2]
            vd_ids: Vehicle detector IDs
            
        Returns:
            Spatial weight matrix [num_nodes, num_nodes]
        """
        return self._compute_spatial_weights(coordinates, vd_ids)
    
    def get_neighbor_info(
        self, 
        coordinates: torch.Tensor, 
        vd_ids: List[str],
        node_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed neighbor information for a specific node (useful for debugging).
        
        Args:
            coordinates: Node coordinates [num_nodes, 2]
            vd_ids: Vehicle detector IDs
            node_idx: Index of the node to analyze
            
        Returns:
            Dictionary with neighbor distances, weights, and indices
        """
        if node_idx >= len(vd_ids):
            raise ValueError(f"node_idx {node_idx} >= num_nodes {len(vd_ids)}")
        
        # Compute distance matrix and weights
        distance_matrix = self.spatial_calculator.compute_distance_matrix(coordinates, vd_ids)
        spatial_weights = self._compute_spatial_weights(coordinates, vd_ids)
        
        # Extract information for the specified node
        distances = distance_matrix[node_idx]
        weights = spatial_weights[node_idx]
        
        # Find active neighbors (non-zero weights)
        active_mask = weights > 1e-8
        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        
        return {
            "node_id": vd_ids[node_idx],
            "node_coordinates": coordinates[node_idx],
            "all_distances": distances,
            "all_weights": weights,
            "neighbor_indices": active_indices,
            "neighbor_distances": distances[active_indices],
            "neighbor_weights": weights[active_indices],
            "neighbor_ids": [vd_ids[i] for i in active_indices.tolist()],
            "num_active_neighbors": len(active_indices),
        }
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get performance statistics for monitoring.
        
        Returns:
            Dictionary with performance metrics
        """
        cache_stats = self.spatial_calculator.get_cache_stats()
        
        stats = {
            "forward_calls": self._forward_calls,
            "avg_aggregation_time": self._total_aggregation_time / max(self._forward_calls, 1),
            **cache_stats,
        }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._forward_calls = 0
        self._total_aggregation_time = 0.0
        self.spatial_calculator.clear_cache()
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"feature_dim={self.feature_dim}, "
            f"pooling_radius={self.config.pooling_radius}m, "
            f"max_neighbors={self.config.max_neighbors}, "
            f"distance_metric={self.config.distance_metric}, "
            f"weighting_function={self.config.weighting_function}, "
            f"aggregation_method={self.config.aggregation_method}"
        )


# === Factory Functions ===

def create_social_pooling_layer(
    config: SocialPoolingConfig,
    feature_dim: int,
    coord_system: Optional[CoordinateSystem] = None,
    **kwargs
) -> SocialPooling:
    """
    Factory function to create a Social Pooling layer.
    
    Args:
        config: Social pooling configuration
        feature_dim: Dimension of input features
        coord_system: Optional coordinate system
        **kwargs: Additional arguments for SocialPooling
        
    Returns:
        Configured SocialPooling layer
    """
    return SocialPooling(config, feature_dim, coord_system, **kwargs)


def create_urban_social_pooling(
    feature_dim: int,
    **config_overrides
) -> SocialPooling:
    """
    Create Social Pooling layer with urban preset configuration.
    
    Args:
        feature_dim: Dimension of input features
        **config_overrides: Configuration overrides
        
    Returns:
        SocialPooling layer configured for urban environments
    """
    config = SocialPoolingConfig.urban_preset(**config_overrides)
    return SocialPooling(config, feature_dim)


def create_highway_social_pooling(
    feature_dim: int,
    **config_overrides  
) -> SocialPooling:
    """
    Create Social Pooling layer with highway preset configuration.
    
    Args:
        feature_dim: Dimension of input features
        **config_overrides: Configuration overrides
        
    Returns:
        SocialPooling layer configured for highway environments
    """
    config = SocialPoolingConfig.highway_preset(**config_overrides)
    return SocialPooling(config, feature_dim)


# Export public interface
__all__ = [
    "SocialPooling",
    "SocialPoolingError", 
    "InvalidInputError",
    "create_social_pooling_layer",
    "create_urban_social_pooling",
    "create_highway_social_pooling",
]