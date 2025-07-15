#!/usr/bin/env python3
"""
Data stability tools - unified interface for data quality and stabilization.

This module provides a unified interface to the data stability functions
that have been integrated into the main codebase.

Usage:
    python scripts/utils/data_stability_tools.py --validate dataset.h5
    python scripts/utils/data_stability_tools.py --stabilize input.h5 output.h5
"""

import argparse
import sys
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from social_xlstm.dataset.storage.h5_converter import TrafficFeatureExtractor


def validate_dataset(dataset_path: str) -> None:
    """Validate dataset quality using the integrated method."""
    print(f"🧪 VALIDATING DATASET QUALITY")
    print(f"{'='*50}")
    
    quality_good, metrics = TrafficFeatureExtractor.validate_dataset_quality(dataset_path)
    
    print(f"\n📊 Dataset: {dataset_path}")
    print(f"📈 Quality Results:")
    
    # Display feature metrics
    for feat_name, feat_metrics in metrics.items():
        if feat_name == 'overall':
            continue
        mean_diff = feat_metrics['mean_diff']
        std_diff = feat_metrics['std_diff']
        
        mean_ok = mean_diff <= 0.10
        std_ok = std_diff <= 0.10
        overall_ok = mean_ok and std_ok
        
        status = "✅" if overall_ok else "⚠️"
        print(f"   {status} {feat_name:15}: mean_diff={mean_diff:.3f} {'✅' if mean_ok else '❌'}, std_diff={std_diff:.3f} {'✅' if std_ok else '❌'}")
    
    # Display overall results
    overall = metrics['overall']
    print(f"\n🎯 Overall Assessment:")
    print(f"   Max mean diff: {overall['max_mean_diff']:.3f}")
    print(f"   Max std diff:  {overall['max_std_diff']:.3f}")
    print(f"   Data quality:  {'✅ GOOD' if quality_good else '⚠️ NEEDS IMPROVEMENT'}")


def stabilize_dataset(input_path: str, output_path: str, start_ratio: float = 0.3) -> None:
    """Stabilize dataset using the integrated method."""
    print(f"🔧 STABILIZING DATASET")
    print(f"{'='*50}")
    
    result_path = TrafficFeatureExtractor.stabilize_dataset(input_path, output_path, start_ratio)
    
    print(f"\n✅ Dataset stabilization complete!")
    print(f"📄 Input:  {input_path}")
    print(f"💾 Output: {result_path}")
    print(f"📊 Removed: {start_ratio*100:.0f}% of early data")


def main():
    parser = argparse.ArgumentParser(
        description="Data stability tools for Social-xLSTM dataset management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--validate", metavar="DATASET", 
                       help="Validate dataset quality")
    parser.add_argument("--stabilize", nargs=2, metavar=("INPUT", "OUTPUT"),
                       help="Stabilize dataset by removing early problematic data")
    parser.add_argument("--start-ratio", type=float, default=0.3,
                       help="Ratio of data to skip from beginning (default: 0.3)")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset(args.validate)
    elif args.stabilize:
        input_path, output_path = args.stabilize
        stabilize_dataset(input_path, output_path, args.start_ratio)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()