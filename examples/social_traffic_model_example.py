"""
Social Traffic Model Usage Example

This script demonstrates how to use the new SocialTrafficModel for traffic prediction
with spatial-temporal features. It shows basic usage, configuration, and integration
with existing training workflows.

Author: Social-xLSTM Team
Version: 1.0
"""

import torch
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our models
from social_xlstm.models.social_traffic_model import SocialTrafficModel, create_social_traffic_model
from social_xlstm.models.model_factory import create_model, compare_model_sizes
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.models.social_pooling_config import SocialPoolingConfig
from social_xlstm.utils.spatial_coords import CoordinateSystem

def generate_dummy_data(
    batch_size: int = 32,
    seq_len: int = 12,
    num_nodes: int = 10,
    num_features: int = 3
) -> tuple:
    """Generate dummy traffic data for testing."""
    
    # Generate temporal data
    temporal_data = torch.randn(batch_size, seq_len, num_features)
    
    # Generate spatial coordinates (in meters, projected coordinates)
    # Simulate a small urban network
    coords = torch.randn(num_nodes, 2) * 1000  # Within 1km radius
    
    # Generate VD IDs
    vd_ids = [f"VD_{i:03d}" for i in range(num_nodes)]
    
    # Generate targets
    targets = torch.randn(batch_size, 1, num_features)
    
    return temporal_data, coords, vd_ids, targets


def example_basic_usage():
    """Example 1: Basic SocialTrafficModel usage."""
    print("\n=== Example 1: Basic Usage ===")
    
    # Generate dummy data
    batch_size, seq_len, num_nodes, num_features = 32, 12, 8, 3
    x_temporal, coordinates, vd_ids, targets = generate_dummy_data(
        batch_size, seq_len, num_nodes, num_features
    )
    
    # Create model using factory function
    model = create_social_traffic_model(
        scenario="urban",
        base_hidden_size=64,
        base_num_layers=2
    )
    
    print(f"Created model: {model.__class__.__name__}")
    print(f"Model info: {model.get_model_info()}")
    
    # Forward pass
    try:
        predictions = model(x_temporal, coordinates, vd_ids)
        print(f"Input shape: {x_temporal.shape}")
        print(f"Coordinates shape: {coordinates.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print("✅ Forward pass successful!")
        
        # Test with attention weights
        predictions_with_attention, attention_weights = model(
            x_temporal, coordinates, vd_ids, return_attention=True
        )
        print(f"Attention weights shape: {attention_weights.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")


def example_model_comparison():
    """Example 2: Compare base LSTM vs Social Traffic Model."""
    print("\n=== Example 2: Model Comparison ===")
    
    # Generate data
    x_temporal, coordinates, vd_ids, targets = generate_dummy_data(16, 12, 5, 3)
    
    # Create base LSTM model
    base_model = create_model(
        model_type="traffic_lstm",
        scenario="urban",
        config_overrides={"hidden_size": 64, "num_layers": 2}
    )
    
    # Create social traffic model
    social_model = create_model(
        model_type="social_traffic_model",
        scenario="urban",
        config_overrides={
            "base_model": {"hidden_size": 64, "num_layers": 2}
        }
    )
    
    # Compare model sizes
    models = {
        "Base LSTM": base_model,
        "Social Traffic Model": social_model
    }
    
    size_comparison = compare_model_sizes(models)
    
    print("Model Size Comparison:")
    for name, info in size_comparison.items():
        print(f"  {name}:")
        print(f"    Parameters: {info['total_parameters']:,}")
        print(f"    Size: {info['model_size_mb']:.2f} MB")
    
    # Compare predictions
    with torch.no_grad():
        base_pred = base_model(x_temporal)
        social_pred = social_model(x_temporal, coordinates, vd_ids)
        
        print(f"\nPrediction Shapes:")
        print(f"  Base LSTM: {base_pred.shape}")
        print(f"  Social Model: {social_pred.shape}")
        
        # Calculate prediction difference
        pred_diff = torch.abs(social_pred - base_pred).mean()
        print(f"  Mean prediction difference: {pred_diff:.4f}")


def example_different_scenarios():
    """Example 3: Test different scenario configurations."""
    print("\n=== Example 3: Different Scenarios ===")
    
    scenarios = ["urban", "highway", "mixed"]
    x_temporal, coordinates, vd_ids, targets = generate_dummy_data(8, 12, 6, 3)
    
    models = {}
    
    for scenario in scenarios:
        try:
            model = create_social_traffic_model(scenario=scenario)
            models[scenario] = model
            
            # Test forward pass
            with torch.no_grad():
                pred = model(x_temporal, coordinates, vd_ids)
                social_weight = model.get_social_influence_weight()
                
            print(f"{scenario.capitalize()} scenario:")
            print(f"  Social influence weight: {social_weight:.3f}")
            print(f"  Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
            
        except Exception as e:
            print(f"❌ {scenario} scenario failed: {e}")
    
    # Compare model configurations
    if models:
        size_comparison = compare_model_sizes(models)
        print("\nScenario Size Comparison:")
        for scenario, info in size_comparison.items():
            print(f"  {scenario}: {info['total_parameters']:,} parameters")


def example_social_pooling_ablation():
    """Example 4: Test with and without social pooling."""
    print("\n=== Example 4: Social Pooling Ablation ===")
    
    x_temporal, coordinates, vd_ids, targets = generate_dummy_data(16, 12, 8, 3)
    
    # Create social traffic model
    model = create_social_traffic_model(scenario="urban")
    
    with torch.no_grad():
        # Prediction with social pooling
        model.enable_social_pooling(True)
        pred_with_social = model(x_temporal, coordinates, vd_ids)
        
        # Prediction without social pooling
        model.enable_social_pooling(False)
        pred_without_social = model(x_temporal, coordinates, vd_ids)
        
        # Base model only prediction
        pred_base_only = model.forward_base_only(x_temporal)
    
    print("Ablation Study Results:")
    print(f"  With social pooling: mean={pred_with_social.mean():.4f}, std={pred_with_social.std():.4f}")
    print(f"  Without social pooling: mean={pred_without_social.mean():.4f}, std={pred_without_social.std():.4f}")
    print(f"  Base model only: mean={pred_base_only.mean():.4f}, std={pred_base_only.std():.4f}")
    
    # Calculate differences
    social_effect = torch.abs(pred_with_social - pred_without_social).mean()
    wrapper_effect = torch.abs(pred_without_social - pred_base_only).mean()
    
    print(f"  Social pooling effect: {social_effect:.4f}")
    print(f"  Wrapper overhead effect: {wrapper_effect:.4f}")


def example_coordinate_systems():
    """Example 5: Test different coordinate systems."""
    print("\n=== Example 5: Coordinate Systems ===")
    
    x_temporal, _, vd_ids, targets = generate_dummy_data(8, 12, 5, 3)
    
    # Test with different coordinate formats
    coord_systems = {
        "projected": torch.randn(5, 2) * 1000,  # Projected coordinates in meters
        "geographic": torch.rand(5, 2) * torch.tensor([0.1, 0.1]) + torch.tensor([23.9, 120.6])  # Taiwan area
    }
    
    for coord_type, coordinates in coord_systems.items():
        try:
            # Create coordinate system
            if coord_type == "geographic":
                coord_system = CoordinateSystem(lat_origin=23.9150, lon_origin=120.6846)
                social_config = SocialPoolingConfig(coordinate_system="geographic")
            else:
                coord_system = CoordinateSystem()
                social_config = SocialPoolingConfig(coordinate_system="projected")
            
            # Update config for appropriate distance metric
            if coord_type == "geographic":
                social_config = SocialPoolingConfig(
                    coordinate_system="geographic",
                    distance_metric="haversine"
                )
            
            model = create_social_traffic_model(
                scenario="urban",
                coord_system=coord_system
            )
            
            with torch.no_grad():
                pred = model(x_temporal, coordinates, vd_ids)
            
            print(f"{coord_type.capitalize()} coordinates:")
            print(f"  Coordinate range: [{coordinates.min():.3f}, {coordinates.max():.3f}]")
            print(f"  Prediction successful: ✅")
            
        except Exception as e:
            print(f"❌ {coord_type} coordinates failed: {e}")


def example_performance_monitoring():
    """Example 6: Monitor model performance and statistics."""
    print("\n=== Example 6: Performance Monitoring ===")
    
    model = create_social_traffic_model(scenario="urban")
    x_temporal, coordinates, vd_ids, targets = generate_dummy_data(32, 12, 10, 3)
    
    # Run multiple forward passes to collect statistics
    num_runs = 5
    print(f"Running {num_runs} forward passes...")
    
    for i in range(num_runs):
        with torch.no_grad():
            pred = model(x_temporal, coordinates, vd_ids)
    
    # Get model statistics
    model_info = model.get_model_info()
    
    print("Model Statistics:")
    print(f"  Forward calls: {model_info['forward_calls']}")
    print(f"  Parameter overhead: {model_info['parameter_overhead']:.2%}")
    print(f"  Social influence weight: {model_info['social_influence_weight']:.3f}")
    
    # Social pooling statistics
    social_stats = model_info['social_pooling_stats']
    if 'cache_hits' in social_stats:
        print("  Social Pooling Cache:")
        print(f"    Hit rate: {social_stats['hit_rate']:.2%}")
        print(f"    Cache size: {social_stats['cache_size']}")


def main():
    """Run all examples."""
    print("🚀 Social Traffic Model Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_model_comparison()
        example_different_scenarios()
        example_social_pooling_ablation()
        example_coordinate_systems()
        example_performance_monitoring()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()