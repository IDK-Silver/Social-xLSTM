#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from typing import List

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Read a list of folders, and list all zip files in each folder."
    )
    
    parser.add_argument(
        "--input_folder_list",
        help="Input folder list file path (e.g., 'path/folder/')",
        nargs="+",
        required=True
    )
    parser.add_argument(
        "--output_file_path",
        help="Output file path (e.g., 'path/to/output.txt')",
        required=True
    )

    return parser.parse_args()


def list_all_zips(input_folder_list: List[str], output_file_path: str) -> None:
    """
    List all zip files in the provided folders and write them to an output file.
    
    Args:
        input_folder_list: List of folder paths to search for zip files
        output_file_path: Path where the list of zip files will be written
    """

    # Convert input folder list and output path to Path objects
    try:
        input_folder_list = [Path(folder).absolute() for folder in input_folder_list]
        output_file_path = Path(output_file_path).absolute()
    except Exception as e:
        print(f"Error converting paths: {e}")
        raise
    
    print(f"Input folder list: {input_folder_list}")
    print(f"Output file path: {output_file_path}")
    
    
    # Create output directory if it doesn't exist
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize list to store all zip files
    all_zip_files = []
    
    # Search for zip files in each input folder
    for folder in input_folder_list:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder}")
            continue
            
        # Find all zip files in the current folder
        zip_files = [
            str(f) 
            for f in folder_path.rglob("*")
            if f.is_file() and f.suffix.lower() in ('.zip', '.ZIP')
        ]
        
        all_zip_files.extend(zip_files)
    
    # Sort the list for consistency
    all_zip_files.sort()
    
    # Write results to output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for zip_file in all_zip_files:
                f.write(f"{zip_file}\n")
        print(f"Successfully wrote {len(all_zip_files)} zip files to {output_file_path}")
    
    except Exception as e:
        print(f"Error writing to output file: {e}")
        raise
    
   

def main():
    args = parse_arguments()
    list_all_zips(
        input_folder_list=args.input_folder_list,
        output_file_path=args.output_file_path
    )

if __name__ == '__main__':
    main()