import argparse
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from pathlib import Path
from typing import Dict, Any
import json

from social_xlstm.models.xlstm import TrafficXLSTMConfig
from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
from social_xlstm.models.distributed_config import DistributedSocialXLSTMConfig, SocialPoolingConfig
from social_xlstm.dataset.core.datamodule import TrafficDataModule
from social_xlstm.dataset.config.base import TrafficDatasetConfig
from social_xlstm.training.recorder import TrainingRecorder


from social_xlstm.utils.yaml import (
    load_yaml_file_to_dict,
    load_profile_config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Distributed Social-xLSTM model for traffic prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file (unified configuration system)
    parser.add_argument('--config', type=str, 
                       help='YAML configuration file containing all training parameters')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()


    logger.info("Starting Distributed Social-xLSTM training")    
    
    # Try to load configuration (support both profile and regular YAML)
    config_path = Path(args.config)
    
    if 'profiles/' in str(config_path):
        logger.info(f"Loading profile configuration from {config_path}")
        config: dict = load_profile_config(args.config)
    else:
        logger.info(f"Loading regular YAML configuration from {config_path}")
        config: dict = load_yaml_file_to_dict(args.config)

    # Ensure config is loaded successfully
    if config is None:
        logger.error("Failed to load config file")
        exit(-1)
    
    logger.info(f"Configuration loaded successfully with {len(config)} top-level sections")
    
    
    logger.info("Building dataset config and datamodule")
    dataset_config = TrafficDatasetConfig(
        hdf5_path=config["data"]["path"],
        sequence_length=config["data"]["sequence_length"],
        prediction_length=config["data"]["prediction_length"],
        selected_vdids=config["data"].get("selected_vdids"),
        selected_features=config["data"]["selected_features"],
        train_ratio=config["data"]["split"]["train"],
        val_ratio=config["data"]["split"]["val"],
        test_ratio=config["data"]["split"]["test"],
        normalize=config["data"]["normalize"],
        normalization_method=config["data"]["normalization_method"],
        fill_missing=config["data"]["fill_missing"],
        stride=config["data"]["stride"],
        batch_size=config["data"]["loader"]["batch_size"],
        num_workers=config["data"]["loader"]["num_workers"],
        pin_memory=config["data"]["loader"]["pin_memory"],
    )

    # Set distributed batch format for per-VD processing
    dataset_config.batch_format = 'distributed'
    datamodule = TrafficDataModule(dataset_config)
    datamodule.setup(stage="fit")

    logger.info("Creating model config and distributed model")
    distributed_config = DistributedSocialXLSTMConfig(
        xlstm=TrafficXLSTMConfig(**config["model"]["xlstm"]),
        num_features=config["model"]["xlstm"]["input_size"],  # Derive from xlstm input_size
        prediction_length=config["data"]["prediction_length"],
        learning_rate=config["trainer"].get("learning_rate", 0.001),
        enable_gradient_checkpointing=config["trainer"].get("enable_gradient_checkpointing", False),
        social_pooling=SocialPoolingConfig(**config["social_pooling"])
    )
    model = DistributedSocialXLSTMModel(distributed_config)

    logger.info("Initializing minimal PyTorch Lightning Trainer")
    trainer = pl.Trainer(**config["trainer"])

    logger.info("Starting training loop")
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()