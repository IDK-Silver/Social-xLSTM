"""
Model Factory - Centralized Model Creation and Configuration

This module provides factory functions and utilities for creating and configuring
various models in the Social-xLSTM project, including TrafficLSTM, SocialPooling,
and SocialTrafficModel combinations.

Author: Social-xLSTM Team
Version: 1.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from dataclasses import asdict
import logging

from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig
from social_xlstm.models.social_pooling import SocialPooling
from social_xlstm.models.social_pooling_config import SocialPoolingConfig
from social_xlstm.models.social_traffic_model import SocialTrafficModel
from social_xlstm.utils.spatial_coords import CoordinateSystem

logger = logging.getLogger(__name__)


class ModelFactoryError(Exception):
    """Base exception for model factory operations."""
    pass


class UnsupportedModelTypeError(ModelFactoryError):
    """Raised when an unsupported model type is requested."""
    pass


class ConfigurationError(ModelFactoryError):
    """Raised when model configuration is invalid."""
    pass


# Model type constants
MODEL_TYPES = {
    "traffic_lstm": TrafficLSTM,
    "social_pooling": SocialPooling,
    "social_traffic_model": SocialTrafficModel,
}

# Preset configurations for different scenarios
SCENARIO_PRESETS = {
    "urban": {
        "base_model": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "social_pooling": {
            "pooling_radius": 500.0,
            "max_neighbors": 12,
            "distance_metric": "euclidean",
            "weighting_function": "gaussian",
            "aggregation_method": "weighted_mean",
        },
        "social_model": {
            "fusion_dropout": 0.1,
            "social_influence_weight": 0.3,
        }
    },
    "highway": {
        "base_model": {
            "hidden_size": 96,
            "num_layers": 2,
            "dropout": 0.15,
        },
        "social_pooling": {
            "pooling_radius": 2000.0,
            "max_neighbors": 5,
            "distance_metric": "euclidean",
            "weighting_function": "exponential",
            "aggregation_method": "weighted_mean",
        },
        "social_model": {
            "fusion_dropout": 0.1,
            "social_influence_weight": 0.2,
        }
    },
    "mixed": {
        "base_model": {
            "hidden_size": 112,
            "num_layers": 2,
            "dropout": 0.18,
        },
        "social_pooling": {
            "pooling_radius": 1200.0,
            "max_neighbors": 8,
            "distance_metric": "euclidean",
            "weighting_function": "linear",
            "aggregation_method": "weighted_mean",
        },
        "social_model": {
            "fusion_dropout": 0.1,
            "social_influence_weight": 0.25,
        }
    }
}


def create_model(
    model_type: str,
    scenario: str = "urban",
    config_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> nn.Module:
    """
    Universal model factory function.
    
    Args:
        model_type: Type of model to create ("traffic_lstm", "social_pooling", "social_traffic_model")
        scenario: Preset scenario ("urban", "highway", "mixed")
        config_overrides: Dictionary of configuration overrides
        **kwargs: Additional model-specific arguments
        
    Returns:
        Configured model instance
        
    Raises:
        UnsupportedModelTypeError: If model_type is not supported
        ConfigurationError: If configuration is invalid
    """
    if model_type not in MODEL_TYPES:
        raise UnsupportedModelTypeError(
            f"Unsupported model type '{model_type}'. Available: {list(MODEL_TYPES.keys())}"
        )
    
    if scenario not in SCENARIO_PRESETS:
        raise ConfigurationError(
            f"Unknown scenario '{scenario}'. Available: {list(SCENARIO_PRESETS.keys())}"
        )
    
    config_overrides = config_overrides or {}
    
    if model_type == "traffic_lstm":
        return _create_traffic_lstm(scenario, config_overrides, **kwargs)
    elif model_type == "social_pooling":
        return _create_social_pooling(scenario, config_overrides, **kwargs)
    elif model_type == "social_traffic_model":
        return _create_social_traffic_model(scenario, config_overrides, **kwargs)


def _create_traffic_lstm(
    scenario: str,
    config_overrides: Dict[str, Any],
    **kwargs
) -> TrafficLSTM:
    """Create TrafficLSTM model with scenario presets."""
    preset = SCENARIO_PRESETS[scenario]["base_model"]
    
    # Merge preset with overrides
    config_dict = {**preset, **config_overrides}
    
    # Create configuration
    config = TrafficLSTMConfig(**config_dict)
    
    logger.info(f"Created TrafficLSTM for scenario '{scenario}' with config: {config}")
    return TrafficLSTM(config)


def _create_social_pooling(
    scenario: str,
    config_overrides: Dict[str, Any],
    feature_dim: int = 128,
    coord_system: Optional[CoordinateSystem] = None,
    **kwargs
) -> SocialPooling:
    """Create SocialPooling model with scenario presets."""
    preset = SCENARIO_PRESETS[scenario]["social_pooling"]
    
    # Merge preset with overrides
    config_dict = {**preset, **config_overrides}
    
    # Create configuration
    config = SocialPoolingConfig(**config_dict)
    
    logger.info(f"Created SocialPooling for scenario '{scenario}' with config: {config}")
    return SocialPooling(config, feature_dim, coord_system, **kwargs)


def _create_social_traffic_model(
    scenario: str,
    config_overrides: Dict[str, Any],
    **kwargs
) -> SocialTrafficModel:
    """Create SocialTrafficModel with scenario presets."""
    preset = SCENARIO_PRESETS[scenario]
    
    # Extract component configurations
    base_preset = preset["base_model"]
    social_preset = preset["social_pooling"]
    model_preset = preset["social_model"]
    
    # Apply overrides
    base_config_dict = {**base_preset, **config_overrides.get("base_model", {})}
    social_config_dict = {**social_preset, **config_overrides.get("social_pooling", {})}
    model_config_dict = {**model_preset, **config_overrides.get("social_model", {})}
    
    # Create configurations
    base_config = TrafficLSTMConfig(**base_config_dict)
    social_config = SocialPoolingConfig(**social_config_dict)
    
    # Merge additional kwargs
    model_kwargs = {**model_config_dict, **kwargs}
    
    logger.info(f"Created SocialTrafficModel for scenario '{scenario}'")
    return SocialTrafficModel(
        base_model_config=base_config,
        social_pooling_config=social_config,
        **model_kwargs
    )


def create_model_ensemble(
    model_configs: List[Dict[str, Any]],
    ensemble_method: str = "averaging"
) -> nn.Module:
    """
    Create an ensemble of models.
    
    Args:
        model_configs: List of model configuration dictionaries
        ensemble_method: Ensemble method ("averaging", "voting", "stacking")
        
    Returns:
        Model ensemble
        
    Note:
        This is a placeholder for future ensemble implementation
    """
    if ensemble_method != "averaging":
        raise NotImplementedError("Only averaging ensemble is currently supported")
    
    models = []
    for config in model_configs:
        model = create_model(**config)
        models.append(model)
    
    return ModelEnsemble(models, ensemble_method)


class ModelEnsemble(nn.Module):
    """
    Simple model ensemble for averaging predictions.
    
    This is a basic implementation that averages predictions from multiple models.
    """
    
    def __init__(self, models: List[nn.Module], method: str = "averaging"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = []
        
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred


def compare_model_sizes(models: Dict[str, nn.Module]) -> Dict[str, Dict[str, Any]]:
    """
    Compare the sizes and parameters of multiple models.
    
    Args:
        models: Dictionary of model name -> model instance
        
    Returns:
        Dictionary with size comparison information
    """
    comparison = {}
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        comparison[name] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,  # float32
            "model_type": model.__class__.__name__
        }
    
    return comparison


def save_model_config(model: nn.Module, filepath: str):
    """
    Save model configuration to file.
    
    Args:
        model: Model instance
        filepath: Path to save configuration
    """
    if hasattr(model, 'get_model_info'):
        config = model.get_model_info()
    elif hasattr(model, 'config'):
        config = asdict(model.config)
    else:
        config = {"model_type": model.__class__.__name__}
    
    torch.save(config, filepath)
    logger.info(f"Saved model configuration to {filepath}")


def load_model_from_config(config_path: str, checkpoint_path: Optional[str] = None) -> nn.Module:
    """
    Load model from configuration file.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Loaded model instance
    """
    config = torch.load(config_path, map_location='cpu')
    
    # Determine model type and create model
    model_type = config.get('model_type')
    
    if model_type == 'TrafficLSTM':
        model_config = TrafficLSTMConfig(**config['config'])
        model = TrafficLSTM(model_config)
    elif model_type == 'SocialTrafficModel':
        # More complex reconstruction for social model
        base_config = TrafficLSTMConfig(**config['base_model_info']['config'])
        social_config = SocialPoolingConfig(**config['social_config'])
        model = SocialTrafficModel(base_config, social_config)
    else:
        raise UnsupportedModelTypeError(f"Cannot reconstruct model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from config: {config_path}")
    return model


# Utility functions for model validation

def validate_model_compatibility(
    base_model: TrafficLSTM,
    social_config: SocialPoolingConfig
) -> bool:
    """
    Validate that base model and social config are compatible.
    
    Args:
        base_model: Base TrafficLSTM model
        social_config: Social pooling configuration
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check hidden size compatibility
        if base_model.config.hidden_size <= 0:
            return False
        
        # Check social config validity
        if social_config.pooling_radius <= 0 or social_config.max_neighbors <= 0:
            return False
        
        # More comprehensive validation could be added here
        return True
        
    except Exception as e:
        logger.warning(f"Model compatibility check failed: {e}")
        return False


def get_recommended_config(
    data_characteristics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get recommended model configuration based on data characteristics.
    
    Args:
        data_characteristics: Dictionary describing the data
            - scenario: "urban", "highway", "mixed"
            - num_vds: number of VDs
            - sequence_length: input sequence length
            - feature_count: number of input features
            - spatial_density: "high", "medium", "low"
            
    Returns:
        Recommended configuration dictionary
    """
    scenario = data_characteristics.get("scenario", "urban")
    num_vds = data_characteristics.get("num_vds", 1)
    spatial_density = data_characteristics.get("spatial_density", "medium")
    
    # Base recommendation from scenario
    base_config = SCENARIO_PRESETS[scenario].copy()
    
    # Adjust based on number of VDs
    if num_vds > 10:
        base_config["base_model"]["hidden_size"] = min(
            base_config["base_model"]["hidden_size"] * 1.5, 256
)
    
    # Adjust based on spatial density
    if spatial_density == "high":
        base_config["social_pooling"]["max_neighbors"] = min(
            base_config["social_pooling"]["max_neighbors"] * 1.5, 20
        )
    elif spatial_density == "low":
        base_config["social_pooling"]["max_neighbors"] = max(
            base_config["social_pooling"]["max_neighbors"] // 2, 3
        )
    
    return base_config


# Export public interface
__all__ = [
    "create_model",
    "create_model_ensemble",
    "ModelEnsemble",
    "compare_model_sizes",
    "save_model_config",
    "load_model_from_config",
    "validate_model_compatibility",
    "get_recommended_config",
    "MODEL_TYPES",
    "SCENARIO_PRESETS",
    "ModelFactoryError",
    "UnsupportedModelTypeError",
    "ConfigurationError",
]