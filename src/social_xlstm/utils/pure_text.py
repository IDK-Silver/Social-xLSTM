def load_txt_file(file_path: str, encoding: str = 'utf-8') -> list[str]:
    """
    Load a text file and return its contents as a list of strings.
    
    Args:
        file_path (str): Path to the text file
        encoding (str): Encoding of the text file (default: 'utf-8')
    
    Returns:
        list[str]: List of strings, each string is a line from the file
        
    Raises:
        IOError: If file cannot be opened or read
        UnicodeDecodeError: If file cannot be decoded with specified encoding
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            lines = [line.strip() for line in file.readlines()]
        return lines
    except (IOError, UnicodeDecodeError) as e:
        raise IOError(f"Error loading file {file_path}: {str(e)}")