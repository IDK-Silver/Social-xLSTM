import os
import pathlib
from social_xlstm.dataset.zip_utils import extract_archive


if __name__ == "__main__":
    
    zip_folder = pathlib.Path(__file__).parent.parent.parent / "blob" / "dataset" / "raw"
    
    os.makedirs(zip_folder, exist_ok=True)
    
    unzip_folder = pathlib.Path(__file__).parent.parent.parent / "blob" / "lab" / "zip"
    os.makedirs(unzip_folder, exist_ok=True)
    
    all_items = os.listdir(zip_folder)
    zip_files = [
        f for f in all_items
        if f.lower().endswith(".zip") and (zip_folder / f).is_file()
    ]
    if not zip_files:
        raise ValueError("No zip files found in the specified directory.")
    
    print(f"Found {len(zip_files)} zip files in {zip_folder}")
    
    filename = zip_files[0]
    zip_file = zip_folder / filename
    
    print(f"Found zip file: {zip_file}")
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_file}")
    print(f"Zip file name: {zip_file.absolute()}")
    
    print(f"Extracting {zip_file} to {unzip_folder}")
    
    
    extract_archive(
        archive_filepath=zip_file.absolute(),
        extract_to_dir=unzip_folder.absolute(),
    )