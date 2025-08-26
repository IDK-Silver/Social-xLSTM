"""
Model Registry System

Centralizes model type information and provides lookup functionality
for dynamic model configuration and instantiation.
"""

from typing import Dict, Any, Type, Optional
from dataclasses import dataclass


@dataclass
class ModelRegistryInfo:
    """Information about a registered model type."""
    config_class: Type
    model_class: Type
    config_key: str
    description: str


# Central model registry mapping model names to their implementations
MODEL_REGISTRY: Dict[str, ModelRegistryInfo] = {}


def register_model(
    name: str,
    config_class: Type,
    model_class: Type, 
    config_key: str,
    description: str = ""
) -> None:
    """Register a model type in the global registry."""
    MODEL_REGISTRY[name] = ModelRegistryInfo(
        config_class=config_class,
        model_class=model_class,
        config_key=config_key,
        description=description
    )


def get_model_info(name: str) -> Optional[ModelRegistryInfo]:
    """Get registry information for a model by name."""
    return MODEL_REGISTRY.get(name)


def list_available_models() -> Dict[str, str]:
    """List all available model types with descriptions."""
    return {name: info.description for name, info in MODEL_REGISTRY.items()}


# Register standard models (imported lazily to avoid circular dependencies)
def _register_standard_models():
    """Register the standard traffic prediction models."""
    try:
        from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig
        register_model(
            name="TrafficLSTM",
            config_class=TrafficLSTMConfig,
            model_class=TrafficLSTM,
            config_key="lstm",
            description="Traditional LSTM model for traffic prediction"
        )
    except ImportError as e:
        print(f"Warning: Could not register TrafficLSTM: {e}")
    
    try:
        from social_xlstm.models.xlstm import TrafficXLSTM, TrafficXLSTMConfig  
        register_model(
            name="TrafficXLSTM",
            config_class=TrafficXLSTMConfig,
            model_class=TrafficXLSTM,
            config_key="xlstm",
            description="Extended LSTM model with enhanced capabilities"
        )
    except ImportError as e:
        print(f"Warning: Could not register TrafficXLSTM: {e}")


# Auto-register standard models when module is imported
_register_standard_models()