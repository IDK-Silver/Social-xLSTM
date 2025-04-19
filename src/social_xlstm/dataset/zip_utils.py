# -*- coding: utf-8 -*-
# File location: src/social_xlstm/dataset/zip_utils.py
import os
import re
from datetime import datetime
import zipfile
from typing import Optional
import py7zr

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
def _extract_zip_internal(zip_filepath: str, extract_to_dir: str, password: Optional[str] = None) -> None:
    """Internal helper to extract ZIP files using zipfile."""
    print(f"INFO: Detected ZIP format (using zipfile) for: {zip_filepath}")
    password_bytes: Optional[bytes] = None
    if password is not None:
        # Standard zipfile password needs bytes
        password_bytes = password.encode('utf-8')

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
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
def _extract_7z_internal(seven_zip_filepath: str, extract_to_dir: str, password: Optional[str] = None) -> None:
    """Internal helper to extract 7z files using py7zr."""
    print(f"INFO: Detected 7z format (using py7zr) for: {seven_zip_filepath}")
    try:
        # Note: py7zr uses string password directly in constructor
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
    password: Optional[str] = None
) -> None:
    """
    Extracts contents from ZIP or 7z archives using magic number detection.

    Handles cases where file extension might not match the actual format
    (e.g., a file named '.zip' containing 7z data).
    Creates the extraction directory if it doesn't exist. Uses py7zr for 7z format.

    Args:
        archive_filepath (str): Path to the archive file.
        extract_to_dir (str): Directory where contents will be extracted.
        password (Optional[str]): Password for the archive, if encrypted.

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
        raise FileNotFoundError(f"Archive file not found: {archive_filepath}")

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
        raise IOError(f"Could not read header from file '{archive_filepath}': {e}") from e

    # Dispatch based on magic number
    if header.startswith(ZIP_MAGIC):
        _extract_zip_internal(archive_filepath, extract_to_dir, password)
    elif header.startswith(SEVEN_ZIP_MAGIC):
        _extract_7z_internal(archive_filepath, extract_to_dir, password)
    else:
        # Get extension only for the error message if format is unknown
        _, file_extension = os.path.splitext(archive_filepath)
        raise ValueError(
            f"Unrecognized or unsupported archive format for file: {archive_filepath} "
            f"(header: {header!r}, ext: {file_extension}). Only ZIP and 7z are supported."
        )
