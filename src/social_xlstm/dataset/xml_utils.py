from pathlib import Path
from enum import Enum
import xml.etree.ElementTree as ET
from .XML_to_Json import VDList_xml_to_Json
from .XML_to_Json import VDLiveList_xml_to_Json


class DatasetType(Enum):
    VDLIST = "VDList"
    VDLIVELIST = "VDLiveList"

class TrafficDatasetParser:
    @staticmethod
    def identify_dataset_type(input_path: str | Path) -> DatasetType:
        """
        Identifies the type of traffic dataset from the filename or XML content.
        
        Args:
            input_path (str | Path): Path to the XML file
            
        Returns:
            DatasetType: Enum indicating whether the dataset is VDList or VDLiveList
            
        Raises:
            ValueError: If the dataset type cannot be determined
        """
        input_path = Path(input_path)
        
        # First try to identify by filename
        filename = input_path.stem.lower()  # Get filename without extension
        if "vdlist" in filename:
            return DatasetType.VDLIST
        elif "vdlivelist" in filename:
            return DatasetType.VDLIVELIST
        
        # If filename doesn't help, check XML content
        try:
            tree = ET.parse(str(input_path))
            root = tree.getroot()
            
            if root.find('.//VDList') is not None:
                return DatasetType.VDLIST
            elif root.find('.//VDLiveList') is not None:
                return DatasetType.VDLIVELIST
            else:
                raise ValueError("Unknown dataset type: XML must contain either VDList or VDLiveList")
        except ET.ParseError:
            raise ValueError("Invalid XML file")

    @staticmethod
    def parse_dataset(input_path: str | Path, output_path: str | Path) -> None:
        """
        Parses the dataset and converts it to JSON based on its type.
        
        Args:
            input_path (str | Path): Path to the input XML file
            output_path (str | Path): Path to save the output JSON file
        """
        dataset_type = TrafficDatasetParser.identify_dataset_type(input_path)
        
        if dataset_type == DatasetType.VDLIST:
            VDList_xml_to_Json(input_path, output_path)
        else:
            VDLiveList_xml_to_Json(input_path, output_path)


def raw_xml_to_json(
    input_file_path:   str | Path,
    output_file_path : str | Path,
    dataset_type: DatasetType = None,
    unknown_ignore: bool = False,
):
    """
    Converts XML file to JSON format based on the dataset type.
    
    Args:
        input_file_path (str | Path): Path to the input XML file to convert
        output_file_path (str | Path): Path where the output JSON file will be saved
        dataset_type (DatasetType, optional): Type of dataset (VDList or VDLiveList). If None, will attempt to identify from file
    
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If dataset type is unknown or cannot be determined
    """
    
    input_file_path = Path(input_file_path)
    output_file_path = Path(output_file_path)
    
    # Check if input file exists
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Check if output folder exists, if not create it
    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset_type is None
    if dataset_type is None:
        # Identify the dataset type
        try:
            dataset_type = TrafficDatasetParser.identify_dataset_type(input_file_path)
        except ValueError as e:
            if unknown_ignore:
                print(f"Warning: {e}. Ignoring.")
                return
            else:
                raise ValueError(f"Could not identify dataset type: {e}")
        print(f"Identified dataset type: {dataset_type}")
    
    
    # Parse the dataset and convert to JSON
    if dataset_type == DatasetType.VDLIST:
        VDList_xml_to_Json(input_file_path, output_file_path)
    elif dataset_type == DatasetType.VDLIVELIST:
        VDLiveList_xml_to_Json(input_file_path, output_file_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")