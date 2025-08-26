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
from social_xlstm.models.distributed_config import DistributedSocialXLSTMConfig, SocialPoolingConfig, OptimizerConfig
from social_xlstm.dataset.core.datamodule import TrafficDataModule
from social_xlstm.dataset.config.base import TrafficDatasetConfig
from social_xlstm.metrics.writer import TrainingMetricsWriter


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
    
    # Output directory for experiments
    parser.add_argument('--output_dir', type=str, default='./lightning_logs',
                       help='Directory to save training outputs (logs, metrics, checkpoints)')
    
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
    
    # Create optimizer config if present
    optimizer_config = None
    if "optimizer" in config:
        optimizer_config = OptimizerConfig(**config["optimizer"])
    
    distributed_config = DistributedSocialXLSTMConfig(
        xlstm=TrafficXLSTMConfig(**config["model"]["xlstm"]),
        num_features=config["model"]["xlstm"]["input_size"],  # Derive from xlstm input_size
        prediction_length=config["data"]["prediction_length"],
        learning_rate=config["trainer"].get("learning_rate", 0.001),
        enable_gradient_checkpointing=config["trainer"].get("enable_gradient_checkpointing", False),
        social_pooling=SocialPoolingConfig(**config["social_pooling"]),
        optimizer=optimizer_config
    )
    model = DistributedSocialXLSTMModel(distributed_config)

    logger.info("Setting up training callbacks")
    callbacks = []
    
    # Basic Lightning callbacks (if specified in config)
    trainer_config = config["trainer"].copy()
    
    # Extract callbacks configuration
    callbacks_config = trainer_config.pop("callbacks", {})
    
    # Add ModelCheckpoint if specified
    if "model_checkpoint" in callbacks_config:
        checkpoint_config = callbacks_config["model_checkpoint"]
        callbacks.append(ModelCheckpoint(**checkpoint_config))
    
    # Add LearningRateMonitor if specified
    if "learning_rate_monitor" in callbacks_config:
        lr_config = callbacks_config.get("learning_rate_monitor", {})
        callbacks.append(LearningRateMonitor(**lr_config))
    
    # Add TrainingMetricsWriter for basic metrics recording
    metrics_config = callbacks_config.get("training_metrics", {})
    # Use metrics subdirectory under user-specified output directory
    metrics_output_dir = metrics_config.get("output_dir", f"{args.output_dir}/metrics")
    
    metrics_writer = TrainingMetricsWriter(
        output_dir=metrics_output_dir,
        metrics=("mae", "mse", "rmse", "r2"),
        splits=("train", "val"),
    )
    callbacks.append(metrics_writer)
    
    logger.info(f"Added {len(callbacks)} callbacks including TrainingMetricsWriter")

    logger.info("Initializing PyTorch Lightning Trainer")
    
    # Set output directory for Lightning logs and checkpoints
    trainer_config['default_root_dir'] = args.output_dir
    
    trainer = pl.Trainer(callbacks=callbacks, **trainer_config)

    logger.info("Starting training loop")
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()