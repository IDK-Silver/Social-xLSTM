import logging
import sys
import os
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

# Logger Configuration
LOG_FILENAME = 'logs/workflow.log'
LOG_LEVEL = logging.INFO

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

# Initialize Logger
logger = logging.getLogger('WorkflowLogger')
logger.setLevel(LOG_LEVEL)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Setup File Handler
try:
    # Changed mode from 'a' (append) to 'w' (write) to overwrite existing log file
    file_handler = logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    logger.info(f"===== New logging session started =====")
    
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: [FallbackLog] %(message)s')
    logger = logging.getLogger('MyWorkflowLogger_Fallback')
    logger.critical(f"!!! Failed to setup file logging to '{LOG_FILENAME}': {e}. Using fallback console logging. !!!")

class ConfigurationError(Exception):
    """Custom exception for configuration related errors"""
    pass

def read_yaml_config(config_path='config.yaml'):
    """
    Read YAML configuration while preserving format and comments.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: The loaded YAML configuration
        
    Raises:
        FileNotFoundError: If config file is not found
        ConfigurationError: If YAML structure is invalid or parsing fails
    """
    yaml_obj = YAML()
    
    try:
        logger.info(f"Reading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml_obj.load(f)
            
            if not isinstance(data, dict):
                raise ConfigurationError(
                    f"Configuration root must be a dictionary, found {type(data).__name__}"
                )
            
            logger.info("Configuration loaded successfully")
            return data
            
    except FileNotFoundError:
        logger.error(f"Configuration file not found: '{config_path}'")
        raise
        
    except YAMLError as yaml_err:
        logger.error(f"YAML parsing error in '{config_path}': {yaml_err}")
        raise ConfigurationError(f"Invalid YAML format in '{config_path}'") from yaml_err
        
    except ConfigurationError as cfg_err:
        logger.error(str(cfg_err))
        raise
        
    except Exception as exc:
        logger.error(f"Unexpected error reading '{config_path}': {exc}")
        raise ConfigurationError(f"Failed to read '{config_path}'") from exc

# Load configuration
try:
    WORKFLOW_CONFIG = read_yaml_config('config.yaml')
    logger.info("Workflow configuration loaded successfully, ready to start execution.")
    
except (FileNotFoundError, ConfigurationError) as e:
    logger.critical(f"Failed to load configuration. Check log file '{LOG_FILENAME}' for details.")
    print(f"CRITICAL ERROR: Failed to load configuration. Check logs in '{LOG_FILENAME}'. Exiting.", 
          file=sys.stderr)
    sys.exit(1)

# Configuration is now available as WORKFLOW_CONFIG for use in rules
rule all:
    input:
    output:
    log:
        "logs/all.log"
    shell:
        """
        echo "Processing data from {input} to {output}" >> {log}
        """

rule list_all_zips:
    input:
        WORKFLOW_CONFIG['storage']['cold_storage']['raw_zip']['folders'],
    output:
        WORKFLOW_CONFIG['dataset']['pre-processed']['raw_zip_list']['file']
    log:
        WORKFLOW_CONFIG['dataset']['pre-processed']['raw_zip_list']['log']
    shell:
    """
        python scripts/dataset/pre-process/list_all_zips.py \
        --input_folder_list {input} \
        --output_file_path  {output} >> {log} 2>&1
    """

rule unzip_and_to_json:
    input:
        zip_list=WORKFLOW_CONFIG['dataset']['pre-processed']['raw_zip_list']['file']
    output:
        status=WORKFLOW_CONFIG['dataset']['pre-processed']['unzip_to_json']['status'],
        zip_dir=WORKFLOW_CONFIG['dataset']['pre-processed']['unzip_to_json']['folder'],
        json_dir=WORKFLOW_CONFIG['dataset']['pre-processed']['json']['folder']
    shell:
    """
        python scripts/dataset/pre-process/unzip_and_to_json.py \
        --input_zip_list {input.zip_list} \
        --output_folder_path {output.zip_dir} \
        --status_file {output.status} >> {log} 2>&1
        --json_folder_path {output.json_dir} >> {log} 2>&1
    """
