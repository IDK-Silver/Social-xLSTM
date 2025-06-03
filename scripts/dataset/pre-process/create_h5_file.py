import os
import argparse
from pathlib import Path
from typing import Optional, List

from social_xlstm.dataset.h5_utils import create_traffic_hdf5

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Create HDF5 file from traffic data JSON files."
    )
    
    parser.add_argument(
        "--source_dir",
        help="Source directory containing JSON files (e.g., 'blob/dataset/pre-processed/unzip_to_json')",
        required=True
    )
    parser.add_argument(
        "--output_path",
        help="Output HDF5 file path (e.g., 'blob/dataset/pre-processed/h5/traffic_features.h5')",
        required=True
    )
    parser.add_argument(
        "--selected_vdids",
        help="Selected VD IDs (comma-separated list, optional)",
        nargs="*",
        default=None
    )
    
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing HDF5 file if it exists",
        action="store_true",
        default=False
    )

    return parser.parse_args()


def create_h5_file(
    source_dir: str, output_path: str,
    selected_vdids: Optional[List[str]] = None,
    overwrite: bool = False
    ) -> None:
    """
    Create HDF5 file from traffic data.
    
    Args:
        source_dir: Source directory containing JSON files
        output_path: Output HDF5 file path
        selected_vdids: Optional list of selected VD IDs
    """
    
    # Convert paths to Path objects for validation
    try:
        source_dir_path = Path(source_dir).absolute()
        output_path_path = Path(output_path).absolute()
    except Exception as e:
        print(f"Error converting paths: {e}")
        raise
    
    print(f"Source directory: {source_dir_path}")
    print(f"Output HDF5 path: {output_path_path}")
    print(f"Selected VD IDs: {selected_vdids}")
    
    # Check if source directory exists
    if not source_dir_path.exists():
        print(f"Error: Source directory not found: {source_dir_path}")
        raise FileNotFoundError(f"Source directory not found: {source_dir_path}")
    
    # Create output directory if it doesn't exist
    output_path_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create the HDF5 file
        reader = create_traffic_hdf5(
            source_dir=str(source_dir_path),
            output_path=str(output_path_path),
            selected_vdids=selected_vdids,
            overwrite=overwrite
        )
        
        print(f"Successfully created HDF5 file: {output_path_path}")
        
    except Exception as e:
        print(f"Error creating HDF5 file: {e}")
        raise


def main():
    args = parse_arguments()
    
    # Process selected_vdids if provided
    selected_vdids = None
    if args.selected_vdids:
        # If it's a single string with commas, split it
        if len(args.selected_vdids) == 1 and ',' in args.selected_vdids[0]:
            selected_vdids = args.selected_vdids[0].split(',')
        else:
            selected_vdids = args.selected_vdids
        # Strip whitespace from each ID
        selected_vdids = [vdid.strip() for vdid in selected_vdids]
    
    create_h5_file(
        source_dir=args.source_dir,
        output_path=args.output_path,
        selected_vdids=selected_vdids,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()