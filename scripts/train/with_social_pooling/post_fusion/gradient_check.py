#!/usr/bin/env python3
"""
Post-Fusion Social Pooling Gradient Flow Validation

This script performs detailed gradient flow analysis to ensure that Social Pooling
correctly propagates gradients back to the base model components.

Key validations:
1. Base model gradients exist and are non-zero
2. Social pooling gradients flow correctly
3. Gated fusion mechanism maintains gradient flow
4. No gradient vanishing or explosion issues

Usage:
    python gradient_check.py [--model_type lstm|xlstm] [--debug]

Author: Social-xLSTM Project Team
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import Social Pooling components
from social_xlstm.models.social_pooling_config import SocialPoolingConfig
from social_xlstm.models.social_traffic_model import create_social_traffic_model
from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.models.xlstm import TrafficXLSTM

# Import Post-Fusion utilities
from common import setup_logging, create_post_fusion_model


class GradientValidator:
    """Validates gradient flow in Post-Fusion Social Pooling models."""
    
    def __init__(self, model_type: str = "lstm", debug: bool = False):
        self.model_type = model_type
        self.debug = debug
        self.logger = setup_logging()
        
        # Initialize components
        self.device = torch.device("cpu")  # Use CPU for gradient analysis
        self.model = None
        self.test_data = None
        self.criterion = nn.MSELoss()
        
        # Gradient tracking
        self.gradient_info = {}
        
    def create_test_model(self) -> nn.Module:
        """Create a small test model for gradient analysis."""
        self.logger.info(f"Creating {self.model_type} model for gradient validation...")
        
        # Small model parameters for fast testing
        input_size = 3
        output_size = 3
        hidden_size = 32
        num_layers = 1
        
        # Create base model
        if self.model_type == "lstm":
            base_model = TrafficLSTM.create_single_vd_model(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.0
            )
        elif self.model_type == "xlstm":
            base_model = TrafficXLSTM.create_single_vd_model(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.0
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create Social Pooling configuration
        social_config = SocialPoolingConfig(
            pooling_radius=500.0,
            max_neighbors=2,
            distance_metric="euclidean",
            weighting_function="gaussian",
            aggregation_method="weighted_mean"
        )
        
        # Create Social Traffic Model
        social_model = create_social_traffic_model(
            base_model=base_model,
            social_config=social_config,
            model_type=f"test_{self.model_type}",
            scenario="test"
        )
        
        self.model = social_model.to(self.device)
        self.logger.info(f"Model created: {self.model.get_model_info()['total_parameters']} parameters")
        
        return self.model
    
    def create_test_data(self, batch_size: int = 2, seq_len: int = 5) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Create synthetic test data for gradient validation."""
        self.logger.info(f"Creating test data: batch_size={batch_size}, seq_len={seq_len}")
        
        # Input sequences [batch_size, seq_len, features]
        input_seq = torch.randn(batch_size, seq_len, 3, device=self.device, requires_grad=False)
        
        # Target sequences [batch_size, 1, features] 
        target_seq = torch.randn(batch_size, 1, 3, device=self.device, requires_grad=False)
        
        # Coordinates for each sample in batch
        coordinates = torch.tensor([
            [0.0, 0.0],      # VD-01 at origin
            [100.0, 100.0]   # VD-02 nearby
        ], dtype=torch.float32, device=self.device)
        
        vd_ids = ["VD-01", "VD-02"]
        
        return input_seq, target_seq, coordinates, vd_ids
    
    def register_gradient_hooks(self) -> None:
        """Register hooks to track gradients in key components."""
        self.logger.info("Registering gradient tracking hooks...")
        
        def create_hook(name: str):
            def hook_fn(grad):
                if grad is not None:
                    self.gradient_info[name] = {
                        'shape': grad.shape,
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'norm': grad.norm().item(),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item()
                    }
                else:
                    self.gradient_info[name] = None
                return grad
            return hook_fn
        
        # Register hooks for key components
        hooks = []
        
        # Base model components
        for name, param in self.model.base_model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(create_hook(f"base_model.{name}"))
                hooks.append(hook)
        
        # Social pooling components  
        for name, param in self.model.social_pooling.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(create_hook(f"social_pooling.{name}"))
                hooks.append(hook)
        
        # Gated fusion components
        for name, param in self.model.gated_fusion.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(create_hook(f"gated_fusion.{name}"))
                hooks.append(hook)
        
        self.logger.info(f"Registered {len(hooks)} gradient hooks")
    
    def validate_gradient_flow(self) -> Dict:
        """Perform gradient flow validation."""
        self.logger.info("Starting gradient flow validation...")
        
        # Create test data
        input_seq, target_seq, coordinates, vd_ids = self.create_test_data()
        
        # Register gradient hooks
        self.register_gradient_hooks()
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Forward pass
        self.logger.info("Performing forward pass...")
        output = self.model(input_seq, coordinates, vd_ids)
        
        # Compute loss
        loss = self.criterion(output, target_seq)
        self.logger.info(f"Forward pass complete. Loss: {loss.item():.6f}")
        
        # Backward pass
        self.logger.info("Performing backward pass...")
        loss.backward()
        
        # Validate gradients
        validation_results = self._analyze_gradients()
        
        return validation_results
    
    def _analyze_gradients(self) -> Dict:
        """Analyze collected gradient information."""
        self.logger.info("Analyzing gradient information...")
        
        results = {
            'summary': {},
            'base_model': {},
            'social_pooling': {},
            'gated_fusion': {},
            'issues': []
        }
        
        # Categorize gradients
        base_grads = {k: v for k, v in self.gradient_info.items() if k.startswith('base_model.')}
        social_grads = {k: v for k, v in self.gradient_info.items() if k.startswith('social_pooling.')}
        fusion_grads = {k: v for k, v in self.gradient_info.items() if k.startswith('gated_fusion.')}
        
        # Analyze each category
        results['base_model'] = self._analyze_gradient_category("Base Model", base_grads)
        results['social_pooling'] = self._analyze_gradient_category("Social Pooling", social_grads)
        results['gated_fusion'] = self._analyze_gradient_category("Gated Fusion", fusion_grads)
        
        # Overall summary
        total_params = len(self.gradient_info)
        valid_grads = sum(1 for v in self.gradient_info.values() if v is not None)
        
        results['summary'] = {
            'total_parameters': total_params,
            'parameters_with_gradients': valid_grads,
            'gradient_coverage': valid_grads / total_params if total_params > 0 else 0,
            'all_components_have_gradients': all([
                results['base_model']['has_gradients'],
                results['social_pooling']['has_gradients'],
                results['gated_fusion']['has_gradients']
            ])
        }
        
        # Check for issues
        self._check_gradient_issues(results)
        
        return results
    
    def _analyze_gradient_category(self, category_name: str, grads: Dict) -> Dict:
        """Analyze gradients for a specific component category."""
        if not grads:
            return {
                'has_gradients': False,
                'parameter_count': 0,
                'issues': [f"No parameters found in {category_name}"]
            }
        
        valid_grads = [v for v in grads.values() if v is not None]
        
        if not valid_grads:
            return {
                'has_gradients': False,
                'parameter_count': len(grads),
                'issues': [f"No gradients computed for {category_name}"]
            }
        
        # Compute statistics
        grad_norms = [g['norm'] for g in valid_grads]
        grad_means = [g['mean'] for g in valid_grads]
        
        analysis = {
            'has_gradients': True,
            'parameter_count': len(grads),
            'valid_gradients': len(valid_grads),
            'avg_gradient_norm': np.mean(grad_norms),
            'max_gradient_norm': np.max(grad_norms),
            'min_gradient_norm': np.min(grad_norms),
            'avg_gradient_mean': np.mean(np.abs(grad_means)),
            'has_nan': any(g['has_nan'] for g in valid_grads),
            'has_inf': any(g['has_inf'] for g in valid_grads),
            'issues': []
        }
        
        # Check for issues
        if analysis['has_nan']:
            analysis['issues'].append(f"{category_name}: NaN gradients detected")
        if analysis['has_inf']:
            analysis['issues'].append(f"{category_name}: Infinite gradients detected")
        if analysis['max_gradient_norm'] > 10.0:
            analysis['issues'].append(f"{category_name}: Large gradients (max norm: {analysis['max_gradient_norm']:.3f})")
        if analysis['max_gradient_norm'] < 1e-6:
            analysis['issues'].append(f"{category_name}: Very small gradients (max norm: {analysis['max_gradient_norm']:.3e})")
        
        return analysis
    
    def _check_gradient_issues(self, results: Dict) -> None:
        """Check for overall gradient flow issues."""
        issues = []
        
        # Collect all issues
        for category in ['base_model', 'social_pooling', 'gated_fusion']:
            issues.extend(results[category].get('issues', []))
        
        # Check overall gradient flow
        if not results['summary']['all_components_have_gradients']:
            issues.append("Some components are not receiving gradients")
        
        if results['summary']['gradient_coverage'] < 0.9:
            issues.append(f"Low gradient coverage: {results['summary']['gradient_coverage']:.1%}")
        
        results['issues'] = issues
    
    def print_results(self, results: Dict) -> None:
        """Print validation results in a readable format."""
        print("\n" + "="*60)
        print("POST-FUSION GRADIENT FLOW VALIDATION RESULTS")
        print("="*60)
        
        # Summary
        summary = results['summary']
        print(f"\n📊 SUMMARY:")
        print(f"   Total Parameters: {summary['total_parameters']}")
        print(f"   Parameters with Gradients: {summary['parameters_with_gradients']}")
        print(f"   Gradient Coverage: {summary['gradient_coverage']:.1%}")
        print(f"   All Components Active: {'✅' if summary['all_components_have_gradients'] else '❌'}")
        
        # Component analysis
        for component, data in [
            ("Base Model", results['base_model']),
            ("Social Pooling", results['social_pooling']),
            ("Gated Fusion", results['gated_fusion'])
        ]:
            print(f"\n🔍 {component.upper()}:")
            if data['has_gradients']:
                print(f"   Status: ✅ Active")
                print(f"   Parameters: {data['parameter_count']}")
                print(f"   Avg Gradient Norm: {data['avg_gradient_norm']:.6f}")
                print(f"   Max Gradient Norm: {data['max_gradient_norm']:.6f}")
                print(f"   Min Gradient Norm: {data['min_gradient_norm']:.6f}")
            else:
                print(f"   Status: ❌ Inactive")
                print(f"   Parameters: {data['parameter_count']}")
        
        # Issues
        if results['issues']:
            print(f"\n⚠️  ISSUES DETECTED:")
            for issue in results['issues']:
                print(f"   • {issue}")
        else:
            print(f"\n✅ NO ISSUES DETECTED")
        
        # Overall status
        success = len(results['issues']) == 0 and summary['all_components_have_gradients']
        print(f"\n🎯 OVERALL STATUS: {'✅ PASSED' if success else '❌ FAILED'}")
        
        if success:
            print("   Gradient flow is working correctly!")
            print("   Ready for end-to-end training.")
        else:
            print("   Gradient flow issues detected.")
            print("   Review issues before proceeding to training.")
        
        print("="*60)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-Fusion Social Pooling Gradient Flow Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model_type", type=str, choices=["lstm", "xlstm"], 
                        default="lstm", help="Base model type to test")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug output")
    
    return parser.parse_args()


def main():
    """Main gradient validation function."""
    args = parse_arguments()
    
    print("="*60)
    print("POST-FUSION GRADIENT FLOW VALIDATION")
    print("="*60)
    print(f"Model Type: {args.model_type}")
    print(f"Debug Mode: {args.debug}")
    print("")
    
    # Create validator
    validator = GradientValidator(model_type=args.model_type, debug=args.debug)
    
    try:
        # Create test model
        validator.create_test_model()
        
        # Run gradient validation
        results = validator.validate_gradient_flow()
        
        # Print results
        validator.print_results(results)
        
        # Save results for analysis
        results_path = Path(__file__).parent / f"gradient_validation_{args.model_type}.json"
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        print(f"\nDetailed results saved to: {results_path}")
        
        # Exit with appropriate code
        success = len(results['issues']) == 0 and results['summary']['all_components_have_gradients']
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Gradient validation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()