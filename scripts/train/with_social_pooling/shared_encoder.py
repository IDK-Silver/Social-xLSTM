import argparse
from pathlib import Path
import logging
import torch
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
    parser.add_argument('--eval_test', action='store_true', help='Run test evaluation after training')
    parser.add_argument('--test_ckpt_mode', type=str, choices=['best', 'last', 'none'], default='best',
                        help='Checkpoint to use for test evaluation (requires ModelCheckpoint for best/last)')
    parser.add_argument('--test_ckpt_path', type=str, default=None,
                        help='Explicit checkpoint path for test evaluation (overrides --test_ckpt_mode)')
    return parser.parse_args()


def main():
    args = parse_args()
    # Enable Tensor Cores-friendly matmul precision on supported GPUs
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass

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
    
    if "model_checkpoint" in config["trainer"].get("callbacks", {}):
        callbacks.append(
            ModelCheckpoint(**config["trainer"]["callbacks"]["model_checkpoint"]) 
        )
    
    output_dir = (
        config["trainer"]
        .get("callbacks", {})
        .get("output_dir", None)
    )
    
    if output_dir is None:
        logger.warning("output_dir is not set in config; using default './output_metrics_is_not_set'")
        output_dir = "./output_dir_is_not_set"
        
    output_dir = Path(output_dir)
    
    # Allow configuring which metrics to record; optionally include normalized versions
    base_metrics = tuple(
        config["trainer"].get("callbacks", {})
        .get("training_metrics", {})
        .get("metrics", ("mae", "rmse", "mape"))
    )
    splits = tuple(
        config["trainer"].get("callbacks", {})
        .get("training_metrics", {})
        .get("splits", ("train", "val"))
    )

    metrics_writer = TrainingMetricsWriter(
        output_dir=output_dir / 'metrics',
        metrics=base_metrics,
        splits=splits,
        csv_filename=(
            config["trainer"].get("callbacks", {})
            .get("training_metrics", {})
            .get("csv_filename", "metrics.csv")
        ),
        json_filename=(
            config["trainer"].get("callbacks", {})
            .get("training_metrics", {})
            .get("json_filename", "metrics_summary.json")
        ),
        append_mode=bool(
            config["trainer"].get("callbacks", {})
            .get("training_metrics", {})
            .get("append_mode", True)
        ),
    )
    callbacks.append(metrics_writer)

    logger.info(
        "Trainer config -> accelerator=%s, devices=%s, precision=%s",
        config["trainer"].get("accelerator", "auto"),
        config["trainer"].get("devices", "auto"),
        config["trainer"].get("precision", "unset"),
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        default_root_dir=output_dir,
        **{k: v for k, v in config["trainer"].items() if k != "callbacks"},
    )
    try:
        logger.info(f"Trainer initialized -> accelerator={trainer.accelerator.__class__.__name__}, devices={trainer.num_devices}, precision={trainer.precision}")
    except Exception:
        pass

    logger.info("Starting training (Shared-Encoder)")
    trainer.fit(model, datamodule)

    # Optional: run test evaluation and record metrics
    if args.eval_test:
        ckpt_path = None
        if args.test_ckpt_path:
            ckpt_path = args.test_ckpt_path
        else:
            if args.test_ckpt_mode in ('best', 'last'):
                ckpt_path = args.test_ckpt_mode
        logger.info(f"Running test evaluation (ckpt_path={ckpt_path})")
        try:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        except Exception as e:
            logger.warning(f"Test evaluation failed: {e}")


if __name__ == "__main__":
    main()
