#!/usr/bin/env python
"""
HDF5 creation script with improved performance and error handling.

Usage:
    python create_h5_file.py \
        --source_dir blob/dataset/pre-processed/unzip_to_json \
        --output_path blob/dataset/pre-processed/h5/traffic_features.h5 \
        --time_range 2025-03-18_00-00-00 2025-03-18_12-00-00 \
        --verbose
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from social_xlstm.dataset.config import TrafficHDF5Config
from social_xlstm.dataset.storage.h5_converter import TrafficHDF5Converter


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create HDF5 file from traffic data with improved performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion (all VDs, all timesteps)
    %(prog)s --source_dir data/json --output_path data/traffic.h5
    
    # Convert specific time range
    %(prog)s --source_dir data/json --output_path data/traffic.h5 \\
        --time_range 2025-03-18_00-00-00 2025-03-18_06-00-00
    
    # Convert specific VDs only
    %(prog)s --source_dir data/json --output_path data/traffic.h5 \\
        --selected_vdids VD-11-0020-002-001 VD-11-0020-008-001
    
    # Verbose mode with detailed warnings
    %(prog)s --source_dir data/json --output_path data/traffic.h5 \\
        --verbose --show_warnings
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Source directory containing time-stamped JSON subdirectories"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output HDF5 file path"
    )
    
    # Optional data selection
    parser.add_argument(
        "--selected_vdids",
        nargs="*",
        default=None,
        help="Specific VD IDs to process (default: all VDs)"
    )
    parser.add_argument(
        "--time_range",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Time range to process (format: YYYY-MM-DD_HH-MM-SS)"
    )
    
    # Processing options
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 file"
    )
    parser.add_argument(
        "--no_consistency_check",
        action="store_true",
        help="Skip consistency check with existing file"
    )
    
    # Performance options
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--show_warnings",
        action="store_true",
        help="Show detailed warnings (can be noisy with many missing VDs)"
    )
    parser.add_argument(
        "--max_missing_report",
        type=int,
        default=10,
        help="Maximum number of missing VD IDs to report per timestep (default: 10)"
    )
    
    # Compression options
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "szip", None],
        default="gzip",
        help="HDF5 compression algorithm (default: gzip)"
    )
    parser.add_argument(
        "--compression_level",
        type=int,
        default=6,
        choices=range(10),
        help="Compression level 0-9 (default: 6)"
    )
    
    # Features to extract
    parser.add_argument(
        "--features",
        nargs="+",
        default=["avg_speed", "total_volume", "avg_occupancy", "speed_std", "lane_count"],
        help="Features to extract (default: all standard features)"
    )
    
    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = TrafficHDF5Config(
            source_dir=Path(args.source_dir),
            output_path=Path(args.output_path),
            selected_vdids=args.selected_vdids,
            feature_names=args.features,
            time_range=tuple(args.time_range) if args.time_range else None,
            compression=args.compression if args.compression else None,
            compression_opts=args.compression_level,
            overwrite=args.overwrite,
            check_consistency=not args.no_consistency_check,
            show_progress=not args.no_progress,
            verbose_warnings=args.show_warnings,
            max_missing_report=args.max_missing_report
        )
        
        # Log configuration
        logger.info("Starting HDF5 conversion with configuration:")
        logger.info(f"  Source: {config.source_dir}")
        logger.info(f"  Output: {config.output_path}")
        logger.info(f"  Time range: {config.time_range or 'All timesteps'}")
        logger.info(f"  VDs: {len(config.selected_vdids) if config.selected_vdids else 'All VDs'}")
        logger.info(f"  Features: {config.feature_names}")
        logger.info(f"  Compression: {config.compression or 'None'}")
        
        # Create converter and run
        converter = TrafficHDF5Converter(config)
        converter.convert()
        
        logger.info("Conversion completed successfully!")
        
        # Print summary statistics
        if config.output_path.exists():
            import h5py
            with h5py.File(config.output_path, 'r') as h5file:
                num_timesteps = h5file.attrs.get('num_timesteps', 0)
                num_locations = h5file.attrs.get('num_locations', 0)
                num_features = h5file.attrs.get('num_features', 0)
                
                logger.info(f"\nOutput file summary:")
                logger.info(f"  Timesteps: {num_timesteps}")
                logger.info(f"  Locations (VDs): {num_locations}")
                logger.info(f"  Features: {num_features}")
                logger.info(f"  File size: {config.output_path.stat().st_size / 1024 / 1024:.2f} MB")
                
                # Check data quality
                features = h5file['data/features']
                total_values = features.size
                # Sample to check NaN ratio (don't load all data)
                sample_size = min(1000, features.shape[0])
                sample_data = features[:sample_size, :, :]
                nan_count = np.isnan(sample_data).sum()
                nan_ratio = nan_count / sample_data.size
                
                logger.info(f"  Data quality (sample):")
                logger.info(f"    NaN ratio: {nan_ratio:.2%}")
        
    except KeyboardInterrupt:
        logger.warning("Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues if not needed
    main()