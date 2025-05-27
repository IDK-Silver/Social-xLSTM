# -*- coding: utf-8 -*-
# File location: src/social_xlstm/dataset/zip_utils.py
import os
import re
from datetime import datetime
import zipfile
from typing import Optional
import py7zr
import shutil  # Added
import tempfile  # Added

ZIP_MAGIC = b'PK\x03\x04'
SEVEN_ZIP_MAGIC = b'7z\xBC\xAF\x27\x1C' # 7z header identifier


def parse_filename_timerange(filename: str) -> tuple[datetime, datetime]:
    """
    Parses a filename string to extract start and end timestamps.

    The expected filename format is:
    'YYYY-MM-DD_HH:MM:SS_to_YYYY-MM-DD_HH:MM:SS.zip'

    Args:
        filename (str): The filename string to parse.

    Returns:
        tuple[datetime, datetime]: A tuple containing two datetime objects:
                                   (start_datetime, end_datetime).

    Raises:
        TypeError: If the input filename is not a string.
        ValueError: If the filename does not match the expected format
                    or contains invalid date/time values.
    """
    # Check if the input is a string
    if not isinstance(filename, str):
        raise TypeError(f"Input must be a string, but got {type(filename)}")

    # Regex to capture the two timestamp parts, matching the full string.
    # It expects YYYY-MM-DD_HH:MM:SS format strictly.
    match = re.match(
        r"^(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})_to_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\.zip$",
        filename
    )

    # If the regex does not match, the format is invalid.
    if not match:
        raise ValueError(f"Invalid filename format: '{filename}'")

    # Extract the start and end timestamp strings from the matched groups.
    start_str = match.group(1)
    end_str = match.group(2)

    # Define the format code for datetime parsing.
    time_format = "%Y-%m-%d_%H:%M:%S"

    try:
        # Attempt to convert the extracted strings into datetime objects.
        start_datetime = datetime.strptime(start_str, time_format)
        end_datetime = datetime.strptime(end_str, time_format)
    except ValueError as e:
        # If strptime fails (e.g., invalid date like '2025-02-30'),
        # it raises a ValueError. Catch it and re-raise with more context.
        # 'from e' preserves the original exception's traceback.
        raise ValueError(f"Invalid date/time value in filename '{filename}': {e}") from e

    # If parsing is successful, return the tuple of datetime objects.
    return start_datetime, end_datetime

# --- Internal helper function for ZIP extraction ---
def _extract_zip_internal(zip_filepath: str, extract_to_dir: str, password: Optional[str] = None, flatten_single_root_folder: bool = False) -> None:
    """Internal helper to extract ZIP files using zipfile."""
    print(f"INFO: Detected ZIP format (using zipfile) for: {zip_filepath}")
    password_bytes: Optional[bytes] = None
    if password is not None:
        # Standard zipfile password needs bytes
        password_bytes = password.encode('utf-8')

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            if flatten_single_root_folder:
                namelist = zip_ref.namelist()
                if not namelist:
                    print(f"Successfully extracted ZIP (empty) to {extract_to_dir}")
                    return

                # Determine if there's a single root folder
                top_level_names = set()
                for name in namelist:
                    normalized_name = name.replace('\\', '/')
                    if normalized_name.strip() == "":
                        continue
                    parts = normalized_name.split('/', 1)
                    top_level_names.add(parts[0])

                if len(top_level_names) == 1:
                    single_root_candidate = list(top_level_names)[0]
                    prefix_to_strip_val = single_root_candidate + "/"
                    
                    is_genuine_single_root_folder = False
                    # Check if candidate acts as a directory and all items are under it
                    if any(name.replace('\\', '/').startswith(prefix_to_strip_val) for name in namelist) or \
                       (single_root_candidate + "/" in [n.replace('\\', '/') for n in namelist]):
                        all_conform = True
                        for name in namelist:
                            normalized_name = name.replace('\\', '/')
                            if not (normalized_name.startswith(prefix_to_strip_val) or \
                                    normalized_name == single_root_candidate or \
                                    normalized_name == single_root_candidate + "/"):
                                all_conform = False
                                break
                        if all_conform:
                            is_genuine_single_root_folder = True
                    
                    if is_genuine_single_root_folder:
                        print(f"INFO: Flattening single root folder '{single_root_candidate}' from ZIP: {zip_filepath}")
                        for member_info in zip_ref.infolist():
                            original_path = member_info.filename.replace('\\', '/')
                            
                            if original_path.startswith(prefix_to_strip_val):
                                member_info.filename = original_path[len(prefix_to_strip_val):]
                            elif original_path == single_root_candidate or original_path == single_root_candidate + "/":
                                # Skip extracting the root folder itself as a named entity
                                continue
                            # else: filename remains original_path (should not be hit if logic is correct)

                            if member_info.filename:  # Avoid extracting if filename becomes empty
                                zip_ref.extract(member_info, path=extract_to_dir, pwd=password_bytes)
                        print(f"Successfully extracted and flattened ZIP to {extract_to_dir}")
                        return

            # Default: extract all without flattening logic, or if flattening conditions not met
            zip_ref.extractall(path=extract_to_dir, pwd=password_bytes)
            print(f"Successfully extracted ZIP to {extract_to_dir}")
    except zipfile.BadZipFile as e:
        raise ValueError(f"Bad ZIP file '{zip_filepath}': {e}") from e
    except RuntimeError as e:
        if 'password' in str(e).lower() or 'passwd' in str(e).lower():
             raise ValueError(f"Password required or incorrect for ZIP file: {zip_filepath}") from e
        else:
             raise RuntimeError(f"Runtime error during ZIP extraction of '{zip_filepath}': {e}") from e
    except Exception as e:
        raise Exception(f"An error occurred during ZIP extraction of '{zip_filepath}': {e}") from e


# --- Internal helper function for 7z extraction (using py7zr) ---
def _extract_7z_internal(seven_zip_filepath: str, extract_to_dir: str, password: Optional[str] = None, flatten_single_root_folder: bool = False) -> None:
    """Internal helper to extract 7z files using py7zr."""
    print(f"INFO: Detected 7z format (using py7zr) for: {seven_zip_filepath}")
    try:
        if flatten_single_root_folder:
            single_root_candidate_name = None
            # Peek into the archive to identify a single root folder
            with py7zr.SevenZipFile(seven_zip_filepath, mode='r', password=password) as archive_peek:
                all_files_peek = archive_peek.getnames()
                if not all_files_peek:
                    print(f"Successfully extracted 7z (empty) to {extract_to_dir}")
                    return

                top_level_names = set()
                for name_orig in all_files_peek:
                    name = name_orig.replace(os.sep, '/')
                    if name.strip() == "":
                        continue
                    parts = name.split('/', 1)
                    top_level_names.add(parts[0])
                
                if len(top_level_names) == 1:
                    candidate = list(top_level_names)[0]
                    prefix_to_check = candidate + "/"
                    
                    is_dir_like_candidate = False
                    if any(n.replace(os.sep, '/').startswith(prefix_to_check) for n in all_files_peek) or \
                       (candidate + "/" in [n.replace(os.sep, '/') for n in all_files_peek]):
                        is_dir_like_candidate = True
                    
                    if is_dir_like_candidate:
                        all_under_candidate = True
                        for name_orig in all_files_peek:
                            name = name_orig.replace(os.sep, '/')
                            if not (name.startswith(prefix_to_check) or \
                                    name == candidate or \
                                    name == candidate + '/'):
                                all_under_candidate = False
                                break
                        if all_under_candidate:
                            single_root_candidate_name = candidate
            
            if single_root_candidate_name:
                print(f"INFO: Flattening single root folder '{single_root_candidate_name}' from 7z: {seven_zip_filepath}")
                with tempfile.TemporaryDirectory(prefix="7z_flatten_") as temp_full_extract_path:
                    with py7zr.SevenZipFile(seven_zip_filepath, mode='r', password=password) as archive_extract:
                        archive_extract.extractall(path=temp_full_extract_path)
                    
                    source_folder_to_flatten = os.path.join(temp_full_extract_path, single_root_candidate_name.replace('/', os.sep))

                    if os.path.isdir(source_folder_to_flatten):
                        os.makedirs(extract_to_dir, exist_ok=True)
                        for item_name in os.listdir(source_folder_to_flatten):
                            source_item_path = os.path.join(source_folder_to_flatten, item_name)
                            dest_item_path = os.path.join(extract_to_dir, item_name)
                            
                            if os.path.isdir(source_item_path) and os.path.isdir(dest_item_path):
                                # Merge contents if destination directory exists
                                for sub_item in os.listdir(source_item_path):
                                    shutil.move(os.path.join(source_item_path, sub_item), os.path.join(dest_item_path, sub_item))
                            else:
                                shutil.move(source_item_path, dest_item_path)
                        print(f"Successfully extracted and flattened 7z to {extract_to_dir}")
                    else:
                        print(f"WARN: Identified single root folder '{single_root_candidate_name}' not found after temp extraction for {seven_zip_filepath}. Extracting normally.")
                        # Fallback to normal extraction if the identified folder isn't there
                        with py7zr.SevenZipFile(seven_zip_filepath, mode='r', password=password) as archive_extract_fallback:
                            archive_extract_fallback.extractall(path=extract_to_dir)
                        print(f"Successfully extracted 7z to {extract_to_dir} (fallback).")
                return

        # Default extraction if not flattening or if flattening conditions not met/failed
        with py7zr.SevenZipFile(seven_zip_filepath, mode='r', password=password) as archive:
            archive.extractall(path=extract_to_dir)
        print(f"Successfully extracted 7z to {extract_to_dir}")
    except py7zr.exceptions.PasswordRequired:
         # If password was None but is required, or if it was wrong.
         # py7zr doesn't always distinguish between missing and wrong password easily.
         raise ValueError(f"Password required or incorrect for 7z file: {seven_zip_filepath}") from None
    except py7zr.exceptions.Bad7zFile as e:
        raise ValueError(f"Bad 7z file '{seven_zip_filepath}': {e}") from e
    except Exception as e:
        # Catch-all for other potential py7zr or OS errors during extraction
        raise Exception(f"An error occurred during 7z extraction of '{seven_zip_filepath}': {e}") from e

# --- Main extraction function (Dispatcher using Magic Numbers) ---
def extract_archive(
    archive_filepath: str,
    extract_to_dir: str,
    password: Optional[str] = None,
    flatten_single_root_folder: bool = False  # Added option
) -> None:
    """
    Extracts contents from ZIP or 7z archives using magic number detection.

    Handles cases where file extension might not match the actual format
    (e.g., a file named '.zip' containing 7z data).
    Creates the extraction directory if it doesn't exist. Uses py7zr for 7z format.
    Optionally, flattens the structure if the archive contains a single root folder.

    Args:
        archive_filepath (str): Path to the archive file.
        extract_to_dir (str): Directory where contents will be extracted.
        password (Optional[str]): Password for the archive, if encrypted.
        flatten_single_root_folder (bool): If True and the archive contains
                                           a single top-level folder, its contents
                                           are extracted directly into extract_to_dir,
                                           omitting the top-level folder itself.
                                           Defaults to False.

    Raises:
        FileNotFoundError: If the archive_filepath does not exist.
        IOError: If the archive file cannot be read.
        ValueError: If the file format is not recognized (ZIP or 7z),
                    or if the archive is invalid/corrupted,
                    or password incorrect/required.
        ImportError: If 7z format is detected but 'py7zr' package is not installed.
        OSError: If directory creation or file writing fails.
        Exception: For other unexpected errors.
    """
    if not os.path.isfile(archive_filepath):
        raise FileNotFoundError("Archive file not found: {}".format(archive_filepath))

    # Create the extraction directory safely
    try:
        os.makedirs(extract_to_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create extraction directory '{extract_to_dir}': {e}") from e

    # Read the first few bytes to detect format
    header = b''
    try:
        with open(archive_filepath, 'rb') as f:
            # Read enough bytes to identify both formats (6 bytes for 7z is longest)
            header = f.read(max(len(ZIP_MAGIC), len(SEVEN_ZIP_MAGIC)))
    except IOError as e:
        raise IOError(f"Could not read header from file {archive_filepath}: {e}") from e

    # Dispatch based on magic number
    if header.startswith(ZIP_MAGIC):
        _extract_zip_internal(archive_filepath, extract_to_dir, password, flatten_single_root_folder)
    elif header.startswith(SEVEN_ZIP_MAGIC):
        _extract_7z_internal(archive_filepath, extract_to_dir, password, flatten_single_root_folder)
    else:
        # Get extension only for the error message if format is unknown
        _, file_extension = os.path.splitext(archive_filepath)
        raise ValueError(
            f"Unrecognized or unsupported archive format for file: {archive_filepath} "
            f"(header: {header!r}, ext: {file_extension}). Only ZIP and 7z are supported."
        )
