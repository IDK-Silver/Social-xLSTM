#!/usr/bin/env python3
"""
Configuration generator - unified interface for creating optimized model configurations.

This module provides a unified interface to the configuration generation functions
that have been integrated into the main codebase.

Usage:
    python scripts/utils/config_generator.py --type optimized --h5_path stable_dataset.h5
    python scripts/utils/config_generator.py --type development --h5_path dataset.h5
    python scripts/utils/config_generator.py --type production --h5_path dataset.h5
"""

import argparse
import sys
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from social_xlstm.dataset.core.processor import TrafficConfigGenerator


def create_optimized_configs(h5_path: str, output_dir: str = "cfgs/fixed", 
                           vd_ids: str = None) -> None:
    """Create optimized configurations using the integrated method."""
    print(f"🔧 CREATING OPTIMIZED CONFIGURATIONS")
    print(f"{'='*50}")
    
    # Parse VD IDs if provided
    vd_id_list = vd_ids.split(',') if vd_ids else None
    
    saved_configs = TrafficConfigGenerator.create_optimized_configs(h5_path, output_dir, vd_id_list)
    
    print(f"\n✅ Configuration generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Configurations created:")
    for config_path in saved_configs:
        print(f"   • {config_path}")


def create_development_configs(h5_path: str, output_dir: str = "cfgs/dev_fixed",
                             vd_ids: str = None) -> None:
    """Create development configurations using the integrated method."""
    print(f"🚀 CREATING DEVELOPMENT CONFIGURATIONS")
    print(f"{'='*50}")
    
    # Parse VD IDs if provided
    vd_id_list = vd_ids.split(',') if vd_ids else None
    
    saved_configs = TrafficConfigGenerator.create_development_configs(h5_path, output_dir, vd_id_list)
    
    print(f"\n✅ Development configuration generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Configurations created:")
    for config_path in saved_configs:
        print(f"   • {config_path}")


def create_production_configs(h5_path: str, output_dir: str = "cfgs/production",
                            vd_ids: str = None) -> None:
    """Create production configurations using the integrated method."""
    print(f"🏭 CREATING PRODUCTION CONFIGURATIONS")
    print(f"{'='*50}")
    
    # Parse VD IDs if provided
    vd_id_list = vd_ids.split(',') if vd_ids else None
    
    saved_configs = TrafficConfigGenerator.create_production_configs(h5_path, output_dir, vd_id_list)
    
    print(f"\n✅ Production configuration generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Configurations created:")
    for config_path in saved_configs:
        print(f"   • {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Configuration generator for Social-xLSTM model training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--h5_path", required=True,
                       help="Path to H5 dataset file")
    parser.add_argument("--type", required=True,
                       choices=['optimized', 'development', 'production'],
                       help="Type of configuration to generate")
    parser.add_argument("--output_dir", 
                       help="Output directory for configurations")
    parser.add_argument("--vd_ids", 
                       help="Comma-separated list of VD IDs to use")
    
    args = parser.parse_args()
    
    try:
        if args.type == 'optimized':
            output_dir = args.output_dir or "cfgs/fixed"
            create_optimized_configs(args.h5_path, output_dir, args.vd_ids)
        elif args.type == 'development':
            output_dir = args.output_dir or "cfgs/dev_fixed"
            create_development_configs(args.h5_path, output_dir, args.vd_ids)
        elif args.type == 'production':
            output_dir = args.output_dir or "cfgs/production"
            create_production_configs(args.h5_path, output_dir, args.vd_ids)
    
    except Exception as e:
        print(f"❌ Error generating configurations: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())