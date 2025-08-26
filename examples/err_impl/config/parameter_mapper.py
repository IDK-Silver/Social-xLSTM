"""
Parameter Mapping Utilities

Handles inconsistencies between training script parameters and underlying
implementation parameters, providing bidirectional mapping and validation.
"""

from typing import Dict, Any, Optional, List
import warnings


class ParameterMapper:
    """
    Maps between training script parameters and underlying implementation parameters.
    
    Handles the inconsistency between:
    - Training script: pool_type with choices ['mean', 'max', 'weighted_mean'] 
    - Social pooling implementation: aggregation_method with ['weighted_mean', 'weighted_sum', 'attention']
    """
    
    # Mapping from training script pool_type to implementation aggregation_method
    POOL_TYPE_TO_AGGREGATION_METHOD = {
        'mean': 'weighted_mean',         # Basic mean pooling -> weighted mean
        'max': 'weighted_mean',          # Max pooling -> weighted mean (closest equivalent)  
        'weighted_mean': 'weighted_mean', # Direct mapping
    }
    
    # Reverse mapping from implementation to training script (for backward compatibility)
    AGGREGATION_METHOD_TO_POOL_TYPE = {
        'weighted_mean': 'weighted_mean',
        'weighted_sum': 'weighted_mean',   # Map to closest training script equivalent
        'attention': 'weighted_mean',      # Map to closest training script equivalent  
    }
    
    # Extended mapping to support all implementation methods in new system
    YAML_AGGREGATION_TO_POOL_TYPE = {
        'weighted_mean': 'weighted_mean',
        'weighted_sum': 'weighted_mean',   # Training script limitation
        'attention': 'weighted_mean',      # Training script limitation
    }
    
    @classmethod
    def map_social_config_to_training_args(
        cls, 
        social_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map YAML social configuration to training script arguments.
        
        DEPRECATED: This method is deprecated with the removal of enable_spatial_pooling.
        New training scripts should use DistributedSocialXLSTMConfig directly.
        
        Args:
            social_config: Social configuration from YAML
            
        Returns:
            Dictionary with simplified social pooling configuration
        """
        warnings.warn(
            "map_social_config_to_training_args is deprecated. "
            "Use DistributedSocialXLSTMConfig with load_distributed_config_from_dict instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not social_config.get('enabled', False):
            return {
                'social_enabled': False,
                'social_radius': 2.0,
                'social_aggregation': 'mean',
                'social_hidden_dim': 128
            }
        
        # Extract spatial-only parameters
        radius = social_config.get('radius', 2.0)  # meters
        aggregation = social_config.get('aggregation', 'weighted_mean')
        hidden_dim = social_config.get('hidden_dim', 128)
        
        # Validate aggregation method
        from ...models.distributed_config import ALLOWED_POOL_TYPES
        if aggregation not in ALLOWED_POOL_TYPES:
            warnings.warn(
                f"Social pooling aggregation '{aggregation}' not in {ALLOWED_POOL_TYPES}. "
                f"Using 'weighted_mean' as fallback.",
                UserWarning
            )
            aggregation = 'weighted_mean'
        
        return {
            'social_enabled': True,
            'social_radius': radius,
            'social_aggregation': aggregation,
            'social_hidden_dim': hidden_dim
        }
    
    @classmethod  
    def map_training_args_to_social_config(
        cls,
        social_enabled: bool,
        social_radius: float,
        social_aggregation: str,
        social_hidden_dim: int = 128
    ) -> Dict[str, Any]:
        """
        Map training script arguments to social configuration format.
        
        DEPRECATED: This method is deprecated with the new configuration system.
        
        Args:
            social_enabled: Whether spatial pooling is enabled
            social_radius: Spatial radius in meters
            social_aggregation: Aggregation method 
            social_hidden_dim: Hidden dimension for social pooling
            
        Returns:
            Social configuration dictionary
        """
        warnings.warn(
            "map_training_args_to_social_config is deprecated. "
            "Use DistributedSocialXLSTMConfig directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not social_enabled:
            return {
                'enabled': False,
                'radius': 2.0,
                'aggregation': 'mean',
                'hidden_dim': 128
            }
        
        # Validate aggregation method
        from ...models.distributed_config import ALLOWED_POOL_TYPES
        if social_aggregation not in ALLOWED_POOL_TYPES:
            warnings.warn(
                f"Social aggregation '{social_aggregation}' not in {ALLOWED_POOL_TYPES}. "
                f"Using 'weighted_mean' as fallback.",
                UserWarning
            )
            social_aggregation = 'weighted_mean'
        
        return {
            'enabled': True,
            'radius': social_radius,
            'aggregation': social_aggregation,
            'hidden_dim': social_hidden_dim
        }
    
    @classmethod
    def validate_parameter_consistency(
        cls,
        training_args: Dict[str, Any],
        social_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Validate parameter consistency and return list of warnings.
        
        Args:
            training_args: Training script arguments
            social_config: Optional social configuration for cross-validation
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check pool_type support
        pool_type = training_args.get('pool_type', 'mean')
        if pool_type not in cls.POOL_TYPE_TO_AGGREGATION_METHOD:
            warnings.append(f"Unknown pool_type '{pool_type}', using 'weighted_mean'")
        
        # Check spatial radius range
        spatial_radius = training_args.get('spatial_radius', 2.0)
        if spatial_radius > 10.0:  # Assuming kilometers
            warnings.append(f"Large spatial_radius {spatial_radius}km may affect performance")
        
        # Cross-validate with social config if provided
        if social_config:
            yaml_enabled = social_config.get('enabled', False)
            script_enabled = training_args.get('social_enabled', training_args.get('enable_spatial_pooling', False))
            
            if yaml_enabled != script_enabled:
                warnings.append(
                    f"Inconsistent pooling enabled state: "
                    f"YAML={yaml_enabled}, script={script_enabled}"
                )
        
        return warnings


def create_parameter_mapper() -> ParameterMapper:
    """Factory function to create a parameter mapper instance."""
    return ParameterMapper()