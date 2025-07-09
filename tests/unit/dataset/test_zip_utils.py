# -*- coding: utf-8 -*-
# File location: test/dataset/test_zip_utils.py
import pytest
from datetime import datetime

# Import the function to be tested from your source code directory
# pytest usually handles the src path automatically when run from the project root.
# If import issues occur, PYTHONPATH might need configuration, or adjust pytest settings.
from social_xlstm.dataset.utils.zip_utils import parse_filename_timerange

# === Test cases for valid inputs ===

def test_valid_filename_standard():
    """Test a standard, valid filename."""
    filename = "2025-04-14_00:00:00_to_2025-04-15_00:00:00.zip"
    expected_start = datetime(2025, 4, 14, 0, 0, 0)
    expected_end = datetime(2025, 4, 15, 0, 0, 0)
    
    start_dt, end_dt = parse_filename_timerange(filename)
    
    assert start_dt == expected_start
    assert end_dt == expected_end
    assert isinstance(start_dt, datetime)
    assert isinstance(end_dt, datetime)

def test_valid_filename_crossing_boundaries():
    """Test a valid filename that crosses date, month, and year boundaries."""
    filename = "2024-12-31_23:59:58_to_2025-01-01_00:00:02.zip"
    expected_start = datetime(2024, 12, 31, 23, 59, 58)
    expected_end = datetime(2025, 1, 1, 0, 0, 2)
    
    start_dt, end_dt = parse_filename_timerange(filename)
    
    assert start_dt == expected_start
    assert end_dt == expected_end

# === Test cases for invalid filename formats (expecting ValueError) ===

@pytest.mark.parametrize("invalid_filename", [
    "2025-04-14_to_2025-04-15.zip",                     # Missing time parts
    "2025-04-14_00:00:00_2025-04-15_00:00:00.zip",     # Missing '_to_' separator
    "prefix_2025-04-14_00:00:00_to_2025-04-15_00:00:00.zip", # Has prefix before timestamp
    "2025-04-14_00:00:00_to_2025-04-15_00:00:00_suffix.zip", # Has suffix after timestamp (before .zip)
    "2025-04-14_00:00:00_to_2025-04-15_00:00:00.txt",     # Incorrect file extension
    "2025-04-14 00:00:00_to_2025-04-15 00:00:00.zip",    # Space instead of underscore between date/time
    "completely_wrong_filename.zip",                    # Completely wrong format
])
def test_invalid_format_raises_value_error(invalid_filename):
    """Test that various invalid filename formats raise ValueError."""
    # Check that the function raises ValueError and the message indicates invalid format
    with pytest.raises(ValueError, match="Invalid filename format"): 
        parse_filename_timerange(invalid_filename)

# === Test cases for invalid date/time values (expecting ValueError) ===

@pytest.mark.parametrize("invalid_datetime_filename", [
    "2025-02-30_10:00:00_to_2025-03-01_10:00:00.zip", # Invalid day (Feb 30)
    "2025-04-14_24:00:00_to_2025-04-15_00:00:00.zip", # Invalid hour (24)
    "2025-04-14_00:60:00_to_2025-04-15_00:00:00.zip", # Invalid minute (60)
    "2025-04-14_00:00:60_to_2025-04-15_00:00:00.zip", # Invalid second (60)
    "2025-13-01_00:00:00_to_2026-01-01_00:00:00.zip", # Invalid month (13)
])
def test_invalid_value_raises_value_error(invalid_datetime_filename):
    """Test that filenames containing invalid date or time values raise ValueError."""
    # Check for ValueError related to invalid date/time values within the filename
    # The 'match' argument checks if the error message contains the specified substring.
    with pytest.raises(ValueError, match="Invalid date/time value in filename"):
        parse_filename_timerange(invalid_datetime_filename)

# === Test cases for invalid input types (expecting TypeError) ===

@pytest.mark.parametrize("invalid_input", [
    None,       # NoneType
    12345,      # Integer
    ["2025-04-14_00:00:00_to_2025-04-15_00:00:00.zip"], # List
    {"filename": "2025-04-14_00:00:00_to_2025-04-15_00:00:00.zip"} # Dictionary
])
def test_invalid_type_raises_type_error(invalid_input):
    """Test that non-string inputs raise TypeError."""
    # Check that the function raises TypeError when the input is not a string.
    with pytest.raises(TypeError, match="Input must be a string"):
        parse_filename_timerange(invalid_input)