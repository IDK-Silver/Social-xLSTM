# -*- coding: utf-8 -*-
# File location: test/test_social_xlstm/dataset/test_extract_archive.py

import pytest
import zipfile
import shutil
import pathlib # Using pathlib for easier path manipulation in tests

# Conditionally import py7zr to check availability and mark tests
try:
    import py7zr
    PY7ZR_INSTALLED = True
except ImportError:
    PY7ZR_INSTALLED = False

# Import the function to be tested
from social_xlstm.dataset.utils.zip_utils import extract_archive

# --- Fixtures for creating test data ---

@pytest.fixture
def content_files(tmp_path: pathlib.Path) -> dict:
    """Creates dummy content files and returns their paths."""
    content_dir = tmp_path / "content_src"
    content_dir.mkdir()
    file1 = content_dir / "file_a.txt"
    file2_dir = content_dir / "subdir"
    file2_dir.mkdir()
    file2 = file2_dir / "file_b.xml"

    file1.write_text("Content of file A.", encoding='utf-8')
    file2.write_text("<data>Content of file B.</data>", encoding='utf-8')
    
    return {"file1": file1, "file2": file2, "base": content_dir}

@pytest.fixture
def sample_zip_file(tmp_path: pathlib.Path, content_files: dict) -> pathlib.Path:
    """Creates a standard ZIP file with dummy content."""
    zip_path = tmp_path / "test_archive.zip"
    file1_rel = content_files["file1"].relative_to(content_files["base"])
    file2_rel = content_files["file2"].relative_to(content_files["base"])

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(content_files["file1"], arcname=file1_rel)
        zf.write(content_files["file2"], arcname=file2_rel)
    return zip_path

@pytest.fixture
def sample_7z_file(tmp_path: pathlib.Path, content_files: dict) -> pathlib.Path:
    """Creates a standard 7z file with dummy content."""
    seven_zip_path = tmp_path / "test_archive.7z"
    file1_rel = content_files["file1"].relative_to(content_files["base"])
    file2_rel = content_files["file2"].relative_to(content_files["base"])

    with py7zr.SevenZipFile(seven_zip_path, 'w') as archive:
        # 使用絕對路徑而不是相對路徑
        archive.write(str(content_files["file1"]), arcname=str(file1_rel))
        archive.write(str(content_files["file2"]), arcname=str(file2_rel))
    
    return seven_zip_path

@pytest.fixture
def sample_7z_as_zip_file(tmp_path: pathlib.Path, sample_7z_file: pathlib.Path) -> pathlib.Path:
    """Creates a 7z file but names it with a .zip extension."""
    target_path = tmp_path / "7z_pretending_to_be.zip"
    shutil.copy(sample_7z_file, target_path)
    return target_path

@pytest.fixture
def non_archive_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a simple text file, not an archive."""
    txt_path = tmp_path / "not_an_archive.txt"
    txt_path.write_text("This is just text.", encoding='utf-8')
    return txt_path

# --- Test Functions ---

def test_extract_valid_zip(sample_zip_file: pathlib.Path, tmp_path: pathlib.Path):
    """Test extracting a standard ZIP file."""
    extract_dir = tmp_path / "extracted_zip"
    extract_archive(str(sample_zip_file), str(extract_dir))

    # Verify extracted content
    assert (extract_dir / "file_a.txt").is_file()
    assert (extract_dir / "file_a.txt").read_text(encoding='utf-8') == "Content of file A."
    assert (extract_dir / "subdir" / "file_b.xml").is_file()
    assert (extract_dir / "subdir" / "file_b.xml").read_text(encoding='utf-8') == "<data>Content of file B.</data>"

@pytest.mark.skipif(not PY7ZR_INSTALLED, reason="py7zr not installed")
@pytest.mark.usefixtures("sample_7z_file")
def test_extract_valid_7z(sample_7z_file: pathlib.Path, tmp_path: pathlib.Path):
    """Test extracting a standard 7z file."""
    extract_dir = tmp_path / "extracted_7z"
    extract_archive(str(sample_7z_file), str(extract_dir))

    # Verify extracted content
    assert (extract_dir / "file_a.txt").is_file()
    assert (extract_dir / "file_a.txt").read_text(encoding='utf-8') == "Content of file A."
    assert (extract_dir / "subdir" / "file_b.xml").is_file()
    assert (extract_dir / "subdir" / "file_b.xml").read_text(encoding='utf-8') == "<data>Content of file B.</data>"

@pytest.mark.skipif(not PY7ZR_INSTALLED, reason="py7zr not installed")
@pytest.mark.usefixtures("sample_7z_as_zip_file")
def test_extract_7z_as_zip(sample_7z_as_zip_file: pathlib.Path, tmp_path: pathlib.Path):
    """Test extracting a 7z file named with a .zip extension."""
    extract_dir = tmp_path / "extracted_7z_as_zip"
    # Should detect 7z based on header despite the .zip name
    extract_archive(str(sample_7z_as_zip_file), str(extract_dir))

    # Verify extracted content
    assert (extract_dir / "file_a.txt").is_file()
    assert (extract_dir / "file_a.txt").read_text(encoding='utf-8') == "Content of file A."
    assert (extract_dir / "subdir" / "file_b.xml").is_file()
    assert (extract_dir / "subdir" / "file_b.xml").read_text(encoding='utf-8') == "<data>Content of file B.</data>"

def test_extract_file_not_found(tmp_path: pathlib.Path):
    """Test extracting a non-existent file."""
    extract_dir = tmp_path / "extract_not_found"
    with pytest.raises(FileNotFoundError, match="Archive file not found"):
        extract_archive("non_existent_archive.zip", str(extract_dir))

def test_extract_unsupported_format(non_archive_file: pathlib.Path, tmp_path: pathlib.Path):
    """Test extracting a file that is not a supported archive format."""
    extract_dir = tmp_path / "extract_unsupported"
    with pytest.raises(ValueError, match="Unrecognized or unsupported archive format"):
        extract_archive(str(non_archive_file), str(extract_dir))

def test_extract_directory_creation(sample_zip_file: pathlib.Path, tmp_path: pathlib.Path):
    """Test that the extraction directory is created if it doesn't exist."""
    extract_dir = tmp_path / "newly_created_dir"
    assert not extract_dir.exists() # Ensure it doesn't exist initially
    extract_archive(str(sample_zip_file), str(extract_dir))
    assert extract_dir.is_dir() # Check it was created
    assert (extract_dir / "file_a.txt").is_file() # Check content was extracted

# --- Tests for password protected files (Optional - requires creating protected archives) ---
# Add tests here if you implement password creation in fixtures and test password handling

# Example placeholder for password test
# @pytest.mark.skip(reason="Password protected archive creation not implemented in fixtures yet")
# def test_extract_zip_with_password(password_protected_zip_file, tmp_path):
#     extract_dir = tmp_path / "extracted_pwd_zip"
#     extract_archive(str(password_protected_zip_file), str(extract_dir), password="correct_password")
#     # ... assertions ...

# @pytest.mark.skip(reason="Password protected archive creation not implemented in fixtures yet")
# def test_extract_zip_wrong_password(password_protected_zip_file, tmp_path):
#      extract_dir = tmp_path / "extract_pwd_zip_fail"
#      with pytest.raises(ValueError, match="Password required or incorrect"): # Or RuntimeError depending on zipfile version
#          extract_archive(str(password_protected_zip_file), str(extract_dir), password="wrong_password")