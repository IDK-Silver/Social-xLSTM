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
        --output_file_path {output} >> {log} 2>&1
        """

rule unzip_and_to_json:
    input:
        zip_list_path=WORKFLOW_CONFIG['dataset']['pre-processed']['raw_zip_list']['file']
    output:
        status=WORKFLOW_CONFIG['dataset']['pre-processed']['unzip_to_json']['status'],
        zip_dir=WORKFLOW_CONFIG['dataset']['pre-processed']['unzip_to_json']['folder']
    log:
        WORKFLOW_CONFIG['dataset']['pre-processed']['unzip_to_json']['log']
    shell:
        """
        python scripts/dataset/pre-process/unzip_and_to_json.py \
        --input_zip_list_path {input.zip_list_path} \
        --output_folder_path {output.zip_dir} \
        --status_file {output.status} >> {log} 2>&1
        """

rule create_h5_file:
    input:
        source_dir=WORKFLOW_CONFIG['dataset']['pre-processed']['unzip_to_json']['folder']
    output:
        h5_file=WORKFLOW_CONFIG['dataset']['pre-processed']['h5']['file']
    params:
        selected_vdids=WORKFLOW_CONFIG['dataset']['pre-processed']['h5'].get('selected_vdids', None),
        time_range=WORKFLOW_CONFIG['dataset']['pre-processed']['h5'].get('time_range', None),
        overwrite=WORKFLOW_CONFIG['dataset']['pre-processed']['h5'].get('overwrite', False)
    log:
        WORKFLOW_CONFIG['dataset']['pre-processed']['h5']['log']
    shell:
        """
        cmd="python scripts/dataset/pre-process/create_h5_file.py --source_dir {input.source_dir} --output_path {output.h5_file}"
        
        if [ -n "{params.selected_vdids}" ] && [ "{params.selected_vdids}" != "None" ]; then
            cmd="$cmd --selected_vdids {params.selected_vdids}"
        fi
        
        if [ -n "{params.time_range}" ] && [ "{params.time_range}" != "None" ]; then
            cmd="$cmd --time_range {params.time_range}"
        fi
        
        if [ "{params.overwrite}" = "True" ]; then
            cmd="$cmd --overwrite"
        fi
        
        echo "Executing: $cmd" >> {log}
        $cmd >> {log} 2>&1
        """


rule train_single_vd_without_social_pooling:
    input:
        h5_file=WORKFLOW_CONFIG['dataset']['pre-processed']['h5']['file']
    output:
        experiment_dir=directory(WORKFLOW_CONFIG['training']['single_vd']['experiment_dir'])
    log:
        WORKFLOW_CONFIG['training']['single_vd']['log']
    params:
        epochs=WORKFLOW_CONFIG['training']['single_vd']['epochs'],
        batch_size=WORKFLOW_CONFIG['training']['single_vd']['batch_size'],
        sequence_length=WORKFLOW_CONFIG['training']['single_vd']['sequence_length'],
        model_type=WORKFLOW_CONFIG['training']['single_vd']['model_type'],
        experiment_name=os.path.basename(WORKFLOW_CONFIG['training']['single_vd']['experiment_dir'])
    shell:
        """
        python scripts/train/without_social_pooling/train_single_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} >> {log} 2>&1
        """

rule train_multi_vd_without_social_pooling:
    input:
        h5_file=WORKFLOW_CONFIG['dataset']['pre-processed']['h5']['file']
    output:
        experiment_dir=directory(WORKFLOW_CONFIG['training']['multi_vd']['experiment_dir'])
    log:
        WORKFLOW_CONFIG['training']['multi_vd']['log']
    params:
        epochs=WORKFLOW_CONFIG['training']['multi_vd']['epochs'],
        batch_size=WORKFLOW_CONFIG['training']['multi_vd']['batch_size'],
        sequence_length=WORKFLOW_CONFIG['training']['multi_vd']['sequence_length'],
        num_vds=WORKFLOW_CONFIG['training']['multi_vd']['num_vds'],
        model_type=WORKFLOW_CONFIG['training']['multi_vd']['model_type'],
        experiment_name=os.path.basename(WORKFLOW_CONFIG['training']['multi_vd']['experiment_dir'])
    shell:
        """
        python scripts/train/without_social_pooling/train_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --num_vds {params.num_vds} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} >> {log} 2>&1
        """

rule train_independent_multi_vd_without_social_pooling:
    input:
        h5_file=WORKFLOW_CONFIG['dataset']['pre-processed']['h5']['file']
    output:
        experiment_dir=directory(WORKFLOW_CONFIG['training']['independent_multi_vd']['experiment_dir'])
    log:
        WORKFLOW_CONFIG['training']['independent_multi_vd']['log']
    params:
        epochs=WORKFLOW_CONFIG['training']['independent_multi_vd']['epochs'],
        batch_size=WORKFLOW_CONFIG['training']['independent_multi_vd']['batch_size'],
        sequence_length=WORKFLOW_CONFIG['training']['independent_multi_vd']['sequence_length'],
        num_vds=WORKFLOW_CONFIG['training']['independent_multi_vd']['num_vds'],
        target_vd_index=WORKFLOW_CONFIG['training']['independent_multi_vd']['target_vd_index'],
        model_type=WORKFLOW_CONFIG['training']['independent_multi_vd']['model_type'],
        experiment_name=os.path.basename(WORKFLOW_CONFIG['training']['independent_multi_vd']['experiment_dir'])
    shell:
        """
        python scripts/train/without_social_pooling/train_independent_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --num_vds {params.num_vds} \
        --target_vd_index {params.target_vd_index} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} >> {log} 2>&1
        """

