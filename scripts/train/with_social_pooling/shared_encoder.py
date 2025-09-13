import argparse
from pathlib import Path
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from social_xlstm.models.xlstm import TrafficXLSTMConfig
from social_xlstm.models.shared_social_xlstm import SharedSocialXLSTMModel
from social_xlstm.models.distributed_config import (
    DistributedSocialXLSTMConfig,
    SocialPoolingConfig,
    OptimizerConfig,
)
from social_xlstm.dataset.core.datamodule import TrafficDataModule
from social_xlstm.dataset.config.base import TrafficDatasetConfig
from social_xlstm.metrics.writer import TrainingMetricsWriter
from social_xlstm.utils.yaml import load_yaml_file_to_dict, load_profile_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Shared-Encoder Social-xLSTM model for traffic prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True, help='YAML profile or regular config')
    parser.add_argument('--output_dir', type=str, default='./lightning_logs', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config (profile or regular)
    config_path = Path(args.config)
    if 'profiles/' in str(config_path):
        logger.info(f"Loading profile configuration from {config_path}")
        config: dict = load_profile_config(args.config)
    else:
        logger.info(f"Loading regular YAML configuration from {config_path}")
        config: dict = load_yaml_file_to_dict(args.config)

    if config is None:
        raise RuntimeError("Failed to load config file")

    # Build dataset config (centralized batch)
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
        batch_format='centralized',
    )

    datamodule = TrafficDataModule(dataset_config)
    datamodule.setup(stage="fit")

    # Build model config
    optimizer_config = None
    if "optimizer" in config:
        optimizer_config = OptimizerConfig(**config["optimizer"])

    distributed_config = DistributedSocialXLSTMConfig(
        xlstm=TrafficXLSTMConfig(**config["model"]["xlstm"]),
        num_features=config["model"]["xlstm"]["input_size"],
        prediction_length=config["data"]["prediction_length"],
        learning_rate=config["trainer"].get("learning_rate", 0.001),
        enable_gradient_checkpointing=config["trainer"].get("enable_gradient_checkpointing", False),
        social_pooling=SocialPoolingConfig(**config["social_pooling"]),
        optimizer=optimizer_config,
    )

    model = SharedSocialXLSTMModel(distributed_config)

    # Callbacks & trainer
    callbacks = []
    trainer_config = config["trainer"].copy()
    callbacks_config = trainer_config.pop("callbacks", {})

    if "model_checkpoint" in callbacks_config:
        callbacks.append(ModelCheckpoint(**callbacks_config["model_checkpoint"]))
    if "learning_rate_monitor" in callbacks_config:
        callbacks.append(LearningRateMonitor(**callbacks_config["learning_rate_monitor"]))

    metrics_config = callbacks_config.get("training_metrics", {})
    metrics_output_dir = metrics_config.get("output_dir", f"{args.output_dir}/metrics")
    metrics_writer = TrainingMetricsWriter(
        output_dir=metrics_output_dir,
        metrics=("mae", "mse", "rmse", "r2"),
        splits=("train", "val"),
    )
    callbacks.append(metrics_writer)

    trainer_config['default_root_dir'] = args.output_dir
    # Log configured trainer settings before init
    cfg_precision = trainer_config.get('precision', 'unset')
    cfg_accelerator = trainer_config.get('accelerator', 'auto')
    cfg_devices = trainer_config.get('devices', 'auto')
    logger.info(f"Trainer config -> accelerator={cfg_accelerator}, devices={cfg_devices}, precision={cfg_precision}")

    trainer = pl.Trainer(callbacks=callbacks, **trainer_config)
    try:
        logger.info(f"Trainer initialized -> accelerator={trainer.accelerator.__class__.__name__}, devices={trainer.num_devices}, precision={trainer.precision}")
    except Exception:
        pass

    logger.info("Starting training (Shared-Encoder)")
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
