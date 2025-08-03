"""
Distance-Based Social Pooling for Social-xLSTM

This module implements distance-based continuous spatial pooling for the 
Social-xLSTM architecture, which DIFFERS from the original Social LSTM 
paper's grid-based discretization approach.

KEY ARCHITECTURAL DECISION:
This implementation uses continuous distance-based pooling rather than the 
grid-based approach from Alahi et al. (2016). This decision was made for 
improved traffic modeling and computational efficiency.

CORE DIFFERENCES FROM ORIGINAL SOCIAL LSTM:
- Uses Euclidean distance for neighbor selection (not grid cells)
- Applies distance-based weighting for aggregation
- No spatial discretization or grid resolution parameters
- Operates on continuous coordinate space

RATIONALE:
- Traffic scenarios benefit from continuous spatial representation
- Avoids discretization artifacts at grid boundaries  
- More computationally efficient for typical traffic densities
- Aligns with modern trajectory prediction approaches (Social-GAN, Trajectron++)

MATHEMATICAL APPROACH:
Original (Grid-Based):
    H^i_t(m,n,:) = Σ_{j∈N_i} 1_{mn}[x^j_t - x^i_t, y^j_t - y^i_t] h^j_{t-1}

Our Implementation (Distance-Based):
    distance = ||pos_i - pos_j||_2
    neighbors = {j : distance ≤ radius}
    social_context = weighted_mean(neighbor_hidden_states)

PERFORMANCE:
- Complexity: O(N×k) where k is avg neighbors (≪ N²)
- Memory: More efficient for sparse traffic scenarios
- Training: Smoother gradients from continuous functions

REFERENCES:
- Architecture Design: docs/architecture/social_pooling.md
- Decision Record: docs/decisions/adr-001-distance-based-social-pooling.md
- Mathematical Specs: docs/technical/mathematical-specifications.md

Based on:
- Alahi et al. (2016) Social LSTM: Human Trajectory Prediction in Crowded Spaces
- Beck et al. (2024) xLSTM: Extended Long Short-Term Memory  
- Modern trajectory prediction best practices
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def xlstm_hidden_states_aggregation(
    agent_hidden_states: Dict[str, torch.Tensor],
    agent_positions: Dict[str, torch.Tensor],
    target_agent_id: str,
    radius: float = 2.0,
    pool_type: str = "mean"
) -> torch.Tensor:
    """
    Aggregate neighboring hidden states for a target agent using spatial proximity.
    
    This is the core algorithm for Task 2.1: xlstm_hidden_states_aggregation.
    
    Args:
        agent_hidden_states: Dictionary mapping agent_id to hidden states [B, T, H]
        agent_positions: Dictionary mapping agent_id to positions [B, T, 2] (x, y coordinates)
        target_agent_id: ID of the target agent for which to compute social context
        radius: Spatial radius for neighbor selection (meters)
        pool_type: Aggregation method ("mean", "max", "weighted_mean")
        
    Returns:
        Aggregated social context tensor [B, H] for the target agent
        
    Raises:
        ValueError: If target_agent_id is not found in the input dictionaries
        KeyError: If agent_positions and agent_hidden_states have mismatched keys
    """
    if target_agent_id not in agent_hidden_states:
        raise ValueError(f"Target agent '{target_agent_id}' not found in hidden states")
    
    if target_agent_id not in agent_positions:
        raise ValueError(f"Target agent '{target_agent_id}' not found in positions")
    
    # Get target agent's information
    target_hidden = agent_hidden_states[target_agent_id]  # [B, T, H]
    target_position = agent_positions[target_agent_id]    # [B, T, 2]
    
    batch_size, seq_len, hidden_dim = target_hidden.shape
    device = target_hidden.device
    
    # Use last timestep for spatial aggregation
    target_pos_last = target_position[:, -1, :]  # [B, 2]
    
    # Collect neighbor hidden states within radius
    neighbor_states = []
    neighbor_weights = []
    
    for agent_id, hidden_states in agent_hidden_states.items():
        if agent_id == target_agent_id:
            continue  # Skip self
            
        if agent_id not in agent_positions:
            logger.warning(f"Agent '{agent_id}' missing position data, skipping")
            continue
            
        # Get neighbor position (last timestep)
        neighbor_pos = agent_positions[agent_id][:, -1, :]  # [B, 2]
        
        # Compute Euclidean distance for each batch item
        distance = torch.norm(target_pos_last - neighbor_pos, p=2, dim=-1)  # [B]
        
        # Check which batch items have this neighbor within radius
        within_radius = distance <= radius  # [B]
        
        if within_radius.any():
            # Get neighbor hidden state (last timestep)
            neighbor_hidden = hidden_states[:, -1, :]  # [B, H]
            
            # Create mask for valid neighbors in this batch
            masked_hidden = torch.where(
                within_radius.unsqueeze(-1).expand(-1, hidden_dim),
                neighbor_hidden,
                torch.zeros_like(neighbor_hidden)
            )
            
            neighbor_states.append(masked_hidden)
            
            if pool_type == "weighted_mean":
                # Inverse distance weighting (add small epsilon to avoid division by zero)
                weights = 1.0 / (distance + 1e-6)  # [B]
                # Zero out weights for agents outside radius
                weights = torch.where(within_radius, weights, torch.zeros_like(weights))
                neighbor_weights.append(weights)
    
    # Aggregate neighbor states
    if not neighbor_states:
        # No neighbors found - return zero context
        logger.debug(f"No neighbors found for agent '{target_agent_id}' within radius {radius}")
        return torch.zeros(batch_size, hidden_dim, device=device, dtype=target_hidden.dtype)
    
    # Stack neighbor states: [num_neighbors, B, H]
    stacked_neighbors = torch.stack(neighbor_states, dim=0)
    
    if pool_type == "mean":
        # Simple mean aggregation
        # Count non-zero neighbors for each batch item
        neighbor_mask = torch.any(stacked_neighbors != 0, dim=-1)  # [num_neighbors, B]
        neighbor_count = neighbor_mask.sum(dim=0).float()  # [B]
        neighbor_count = torch.clamp(neighbor_count, min=1.0)  # Avoid division by zero
        
        social_context = stacked_neighbors.sum(dim=0) / neighbor_count.unsqueeze(-1)  # [B, H]
        
    elif pool_type == "max":
        # Max pooling
        social_context = torch.max(stacked_neighbors, dim=0)[0]  # [B, H]
        
    elif pool_type == "weighted_mean":
        # Weighted mean using inverse distance
        if not neighbor_weights:
            # Fallback to regular mean
            neighbor_mask = torch.any(stacked_neighbors != 0, dim=-1)
            neighbor_count = neighbor_mask.sum(dim=0).float()  # [B]
            neighbor_count = torch.clamp(neighbor_count, min=1.0)
            social_context = stacked_neighbors.sum(dim=0) / neighbor_count.unsqueeze(-1)
        else:
            # Stack weights: [num_neighbors, B]
            stacked_weights = torch.stack(neighbor_weights, dim=0)
            
            # Normalize weights for each batch item
            total_weights = stacked_weights.sum(dim=0, keepdim=True)  # [1, B]
            total_weights = torch.clamp(total_weights, min=1e-6)  # Avoid division by zero
            normalized_weights = stacked_weights / total_weights  # [num_neighbors, B]
            
            # Weighted aggregation
            weighted_neighbors = stacked_neighbors * normalized_weights.unsqueeze(-1)  # [num_neighbors, B, H]
            social_context = weighted_neighbors.sum(dim=0)  # [B, H]
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}. Supported: 'mean', 'max', 'weighted_mean'")
    
    return social_context


class XLSTMSocialPoolingLayer(nn.Module):
    """
    Neural network module wrapper for xLSTM hidden states aggregation.
    
    This class provides a PyTorch nn.Module interface for the 
    xlstm_hidden_states_aggregation function, making it easy to integrate
    into larger neural network architectures.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        radius: float = 2.0,
        pool_type: str = "mean",
        learnable_radius: bool = False
    ):
        """
        Initialize the social pooling layer.
        
        Args:
            hidden_dim: Hidden dimension of xLSTM states
            radius: Spatial radius for neighbor selection (meters)
            pool_type: Aggregation method ("mean", "max", "weighted_mean")
            learnable_radius: Whether to make radius a learnable parameter
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        
        if learnable_radius:
            # Make radius a learnable parameter
            self.radius = nn.Parameter(torch.tensor(radius, dtype=torch.float32))
        else:
            # Fixed radius
            self.register_buffer('radius', torch.tensor(radius, dtype=torch.float32))
        
        logger.info(f"XLSTMSocialPoolingLayer initialized: hidden_dim={hidden_dim}, "
                   f"radius={radius}, pool_type={pool_type}, learnable_radius={learnable_radius}")
    
    def forward(
        self,
        agent_hidden_states: Dict[str, torch.Tensor],
        agent_positions: Dict[str, torch.Tensor],
        target_agent_ids: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batch social pooling.
        
        Args:
            agent_hidden_states: Dictionary mapping agent_id to hidden states [B, T, H]
            agent_positions: Dictionary mapping agent_id to positions [B, T, 2]
            target_agent_ids: Optional list of specific agents to process.
                              If None, processes all agents in agent_hidden_states.
        
        Returns:
            Dictionary mapping agent_id to aggregated social context [B, H]
        """
        if target_agent_ids is None:
            target_agent_ids = list(agent_hidden_states.keys())
        
        social_contexts = OrderedDict()
        
        for target_id in target_agent_ids:
            try:
                social_context = xlstm_hidden_states_aggregation(
                    agent_hidden_states=agent_hidden_states,
                    agent_positions=agent_positions,
                    target_agent_id=target_id,
                    radius=float(self.radius),
                    pool_type=self.pool_type
                )
                social_contexts[target_id] = social_context
                
            except (ValueError, KeyError) as e:
                logger.error(f"Social pooling failed for agent '{target_id}': {e}")
                # Return zero context for failed agents
                batch_size = next(iter(agent_hidden_states.values())).shape[0]
                device = next(iter(agent_hidden_states.values())).device
                social_contexts[target_id] = torch.zeros(
                    batch_size, self.hidden_dim, device=device
                )
        
        return social_contexts
    
    def get_info(self) -> Dict[str, any]:
        """Get layer information."""
        return {
            "layer_type": "XLSTMSocialPoolingLayer",
            "hidden_dim": self.hidden_dim,
            "radius": float(self.radius),
            "pool_type": self.pool_type,
            "learnable_radius": isinstance(self.radius, nn.Parameter)
        }


# Helper functions for coordinate systems and spatial utilities

def create_mock_positions(
    vd_ids: List[str],
    batch_size: int,
    seq_len: int,
    spatial_range: float = 10.0,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    Create mock position data for testing and development.
    
    Args:
        vd_ids: List of VD/agent identifiers
        batch_size: Batch size
        seq_len: Sequence length
        spatial_range: Range of spatial coordinates (meters)
        device: Torch device
        
    Returns:
        Dictionary mapping vd_id to position tensors [B, T, 2]
    """
    positions = OrderedDict()
    
    for i, vd_id in enumerate(vd_ids):
        # Create deterministic but varied positions for each VD
        base_x = (i % 5) * spatial_range / 5  # Grid pattern
        base_y = (i // 5) * spatial_range / 5
        
        # Add some random movement over time
        torch.manual_seed(hash(vd_id) % 2**32)  # Deterministic based on vd_id
        positions[vd_id] = torch.randn(batch_size, seq_len, 2, device=device) * 0.5 + \
                          torch.tensor([base_x, base_y], device=device)
    
    return positions


def validate_spatial_inputs(
    agent_hidden_states: Dict[str, torch.Tensor],
    agent_positions: Dict[str, torch.Tensor]
) -> None:
    """
    Validate that hidden states and positions have compatible formats.
    
    Args:
        agent_hidden_states: Dictionary mapping agent_id to hidden states [B, T, H]
        agent_positions: Dictionary mapping agent_id to positions [B, T, 2]
        
    Raises:
        ValueError: If inputs are incompatible
    """
    if not agent_hidden_states:
        raise ValueError("agent_hidden_states cannot be empty")
    
    if not agent_positions:
        raise ValueError("agent_positions cannot be empty")
    
    # Check that all agents in hidden_states have positions
    missing_positions = set(agent_hidden_states.keys()) - set(agent_positions.keys())
    if missing_positions:
        raise ValueError(f"Missing positions for agents: {missing_positions}")
    
    # Check tensor shapes consistency
    for agent_id in agent_hidden_states.keys():
        if agent_id not in agent_positions:
            continue
            
        hidden_shape = agent_hidden_states[agent_id].shape
        position_shape = agent_positions[agent_id].shape
        
        # Check batch and time dimensions match
        if hidden_shape[:2] != position_shape[:2]:
            raise ValueError(
                f"Agent '{agent_id}': hidden states {hidden_shape} and "
                f"positions {position_shape} have incompatible batch/time dimensions"
            )
        
        # Check position has correct spatial dimension
        if position_shape[-1] != 2:
            raise ValueError(
                f"Agent '{agent_id}': positions must have shape [B, T, 2], "
                f"got {position_shape}"
            )