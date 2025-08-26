#!/usr/bin/env python 
# Standard library imports
import os
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Union
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local application imports
from social_xlstm.dataset.zip_utils import extract_archive
from social_xlstm.dataset.xml_utils import raw_xml_to_json, DatasetType
from social_xlstm.utils.pure_text import load_txt_file


class Status:
    def __init__(self, processed_files=None):
        """
        Initialize a Status object to track processed files.
        
        Args:
            processed_files: List of already processed files (optional)
        """
        self.processed_files = processed_files or []
    
    def add_processed(self, file_name):
        """
        Mark a file as processed.
        
        Args:
            file_name: Name of the file that has been processed
        """
        if file_name not in self.processed_files:
            self.processed_files.append(file_name)
    
    def is_processed(self, file_name):
        """
        Check if a file has already been processed.
        
        Args:
            file_name: Name of the file to check
        
        Returns:
            bool: True if the file has been processed, False otherwise
        """
        return file_name in self.processed_files
    
    def save_to_json(self, file_path):
        """
        Save the current status to a JSON file.
        
        Args:
            file_path: Path where the status will be saved
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'processed_files': self.processed_files}, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_json(cls, file_path):
        """
        Load status from a JSON file.
        
        Args:
            file_path: Path to the JSON file to load from
            
        Returns:
            Status: A new Status instance with loaded data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls(processed_files=data.get('processed_files', []))
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, return a new empty Status
            return cls()

# Helper function for multithreaded XML to JSON conversion
def _process_xml_to_json_worker(xml_file_path: Path, target_output_dir: Path):
    """
    Converts a single XML file to JSON.
    Helper for multithreaded processing.
    """
    # Use with_suffix for robust extension replacement
    output_json_filename = xml_file_path.with_suffix('.json').name
    file_output_path = target_output_dir / output_json_filename

    raw_xml_to_json(
        input_file_path=xml_file_path.absolute(),
        output_file_path=file_output_path.absolute(),
        dataset_type=None,  # As per original logic
        unknown_ignore=True, # As per original logic
    )

def parse_arguments():
    """
    Parse command-line arguments for the unzip_and_to_json step.
    """
    parser = argparse.ArgumentParser(
        description="Unzip archives and convert their contents to JSON."
    )
    parser.add_argument(
        "--input_zip_list_path",
        help="Path to the file containing the list of zip files to process.",
        required=True
    )
    parser.add_argument(
        "--output_folder_path",
        help="Directory where the unzipped files will be stored.",
        required=True
    )
    parser.add_argument(
        "--status_file",
        help="Path to the status file to write process results.",
        required=True
    )

    return parser.parse_args()


def unzip_and_to_json(
    input_zip_list_path: str,
    output_folder_path: str,
    status_file: str,
) -> None:
    
    # Convert paths to Path objects and validate existence
    input_zip_list_path = Path(input_zip_list_path)
    status_file_path = Path(status_file)
    output_folder_path = Path(output_folder_path)

    if not input_zip_list_path.exists():
        raise FileNotFoundError(f"Input zip list file not found: {input_zip_list_path}")

    # Load existing status or create new if file doesn't exist
    status: Union[Status, None] = None
    try:
        status = Status.load_from_json(status_file_path)
    except FileNotFoundError:
        status = Status()
    
    # Read the list of zip files
    zip_files = load_txt_file(input_zip_list_path)
    
    # Get list of unprocessed files by filtering out already processed ones
    unprocessed_zip_files = [
        f 
        for f in zip_files 
        if str(Path(f).absolute()) not in status.processed_files
    ]
    
    print(f"Unprocessed files: {unprocessed_zip_files}")
    
    if not unprocessed_zip_files:
        print("All files have already been processed.")
        return
    
    for zip_file_path in unprocessed_zip_files:
        zip_file_path = Path(zip_file_path)
        if not zip_file_path.exists():
            print(f"Warning: File not found: {zip_file_path}")
            continue
        
        # Create a temporary folder for unzipping
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Unzip the file to temp directory
            extract_archive(
                archive_filepath=zip_file_path,
                extract_to_dir=temp_dir,
                flatten_single_root_folder=True
            )
            
            # Process files from temp directory
            temp_path = Path(temp_dir)
            
            # Note : all unconverted files are stored in the temp_path
            #        and it from same zip file
            unconverted_files: list[Path] = []
            
            
            # Print the structure of the temp directory for debugging
            for folder in temp_path.iterdir():
                unconverted_files.extend([
                    f for f in folder.rglob("*") 
                    if f.is_file() and f.suffix.lower() in ('.xml', '.XML')
                ])
            
            with ThreadPoolExecutor() as executor:
                    futures = []
                    for f_xml in unconverted_files:
                        futures.append(executor.submit(
                            _process_xml_to_json_worker,
                            f_xml,
                            output_folder_path / f_xml.parent.name
                        ))
                    
                    for future in as_completed(futures):
                        try:
                            future.result()  # Retrieve result or raise exception from thread
                        except Exception as e:
                            # Handle exceptions from threads, e.g., log them
                            print(f"Error converting file in thread: {e}")
                            # Depending on requirements, you might want to collect errors
                            # or stop processing.
            
            status.add_processed(str(
                zip_file_path.absolute()
            ))
    
        status.save_to_json(status_file_path)            

def main():
    args = parse_arguments()
    unzip_and_to_json(
        input_zip_list_path=args.input_zip_list_path,
        output_folder_path=args.output_folder_path,
        status_file=args.status_file,
    )

if __name__ == '__main__':
    main()