"""
TrainingMetricsWriter - Lightning Callback for basic metrics recording.

Captures MAE, MSE, RMSE, R² metrics during Lightning training and saves to CSV/JSON
for later visualization without re-training.
"""

import csv
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Handle different PyTorch Lightning versions
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except ImportError:
        # Fallback for very old versions or if not available
        def rank_zero_only(func):
            """Fallback decorator that always executes the function"""
            return func


class TrainingMetricsWriter(Callback):
    """
    Lightning Callback for recording basic training metrics to CSV and JSON.
    
    Records epoch-level metrics (MAE, MSE, RMSE, R²) for train/val/test splits
    with distributed training safety and atomic file operations.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        metrics: Tuple[str, ...] = ("mae", "mse", "rmse", "r2"),
        splits: Tuple[str, ...] = ("train", "val"),
        csv_filename: str = "metrics.csv",
        json_filename: str = "metrics_summary.json",
        append_mode: bool = True,
    ):
        """
        Initialize TrainingMetricsWriter.
        
        Args:
            output_dir: Directory to save metrics files
            metrics: Tuple of metric names to record
            splits: Tuple of data splits to track (train, val, test)
            csv_filename: Name of CSV output file
            json_filename: Name of JSON summary file
            append_mode: Whether to append to existing files (for resume training)
        """
        super().__init__()
        
        self.output_dir = Path(output_dir)
        self.metrics = metrics
        self.splits = splits
        self.csv_filename = csv_filename
        self.json_filename = json_filename
        self.append_mode = append_mode
        
        # Internal state
        self._csv_path = self.output_dir / csv_filename
        self._json_path = self.output_dir / json_filename
        self._csv_headers_written = False
        self._recorded_epochs: set = set()  # Track to avoid duplicates
        
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup callback - create output directory and initialize files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if resuming from existing CSV
        if self.append_mode and self._csv_path.exists():
            self._load_existing_epochs()
            self._csv_headers_written = True
        else:
            self._csv_headers_written = False
    
    @rank_zero_only
    def _load_existing_epochs(self) -> None:
        """Load existing epochs from CSV to avoid duplicates."""
        try:
            with open(self._csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(row['epoch'])
                    split = row['split']
                    self._recorded_epochs.add((epoch, split))
        except (FileNotFoundError, ValueError, KeyError):
            # File doesn't exist or is malformed, start fresh
            self._recorded_epochs.clear()
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record training metrics at end of training epoch."""
        self._record_epoch_metrics(trainer, pl_module, "train")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record validation metrics at end of validation epoch."""
        if getattr(trainer, 'sanity_checking', False):
            return
        if "val" in self.splits:
            self._record_epoch_metrics(trainer, pl_module, "val")
    
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record test metrics at end of test epoch."""
        if "test" in self.splits:
            self._record_epoch_metrics(trainer, pl_module, "test")
    
    @rank_zero_only
    def _record_epoch_metrics(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        split: str
    ) -> None:
        """
        Record metrics for current epoch and split.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
            split: Data split name (train, val, test)
        """
        current_epoch = trainer.current_epoch
        
        # Skip if already recorded (for resume scenarios)
        if (current_epoch, split) in self._recorded_epochs:
            return
        
        # Extract metrics from logged values
        metrics_data = self._extract_metrics(pl_module, split)
        
        if not metrics_data:
            warnings.warn(f"No metrics found for split '{split}' at epoch {current_epoch}")
            return
        
        # Write to CSV
        self._write_csv_row(current_epoch, split, metrics_data)
        
        # Track recorded epoch
        self._recorded_epochs.add((current_epoch, split))
    
    def _extract_metrics(
        self, 
        pl_module: pl.LightningModule, 
        split: str
    ) -> Dict[str, float]:
        """
        Extract metrics from Lightning module's logged values.
        
        Tries multiple common naming patterns: {split}_{metric}, {metric}_{split}, {metric}
        """
        logged_metrics = pl_module.trainer.callback_metrics
        metrics_data = {}
        
        for metric in self.metrics:
            value = None
            
            # Try different naming patterns
            patterns = [
                f"{split}_{metric}",  # e.g., "train_mae"
                f"{metric}_{split}",  # e.g., "mae_train"
                f"{metric}",          # e.g., "mae" (if split context is clear)
            ]
            
            for pattern in patterns:
                if pattern in logged_metrics:
                    value = float(logged_metrics[pattern])
                    break
            
            if value is not None:
                metrics_data[metric] = value
            else:
                # Try with uppercase variations
                for pattern in patterns:
                    upper_pattern = pattern.upper()
                    if upper_pattern in logged_metrics:
                        value = float(logged_metrics[upper_pattern])
                        metrics_data[metric] = value
                        break
        
        return metrics_data
    
    @rank_zero_only
    def _write_csv_row(
        self, 
        epoch: int, 
        split: str, 
        metrics_data: Dict[str, float]
    ) -> None:
        """Write single row to CSV file with atomic operation."""
        
        # Prepare row data
        row_data = {
            'epoch': epoch,
            'split': split,
            'wall_time': time.time(),
            **metrics_data
        }
        
        # Ensure all expected metrics are present (fill with NaN if missing)
        for metric in self.metrics:
            if metric not in row_data:
                row_data[metric] = float('nan')
        
        # Write headers if first time
        if not self._csv_headers_written:
            self._write_csv_headers()
            self._csv_headers_written = True
        
        # Atomic write: write to temp file then rename
        temp_path = self._csv_path.with_suffix('.tmp')
        
        try:
            # Read existing data if appending
            existing_rows = []
            if self.append_mode and self._csv_path.exists():
                with open(self._csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            
            # Write all data to temp file
            with open(temp_path, 'w', newline='') as f:
                fieldnames = ['epoch', 'split'] + list(self.metrics) + ['wall_time']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write existing rows
                for row in existing_rows:
                    writer.writerow(row)
                
                # Write new row
                writer.writerow(row_data)
            
            # Atomic rename
            temp_path.rename(self._csv_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to write metrics CSV: {e}") from e
    
    @rank_zero_only
    def _write_csv_headers(self) -> None:
        """Write CSV headers if file doesn't exist."""
        if not self._csv_path.exists() or not self.append_mode:
            with open(self._csv_path, 'w', newline='') as f:
                fieldnames = ['epoch', 'split'] + list(self.metrics) + ['wall_time']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save final JSON summary when training completes."""
        self._save_json_summary(trainer, pl_module)
    
    @rank_zero_only
    def _save_json_summary(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save training summary to JSON file."""
        try:
            # Read final metrics from CSV
            final_metrics = {}
            if self._csv_path.exists():
                with open(self._csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    # Group by split and get final epoch metrics
                    for split in self.splits:
                        split_rows = [r for r in rows if r['split'] == split]
                        if split_rows:
                            # Get metrics from last epoch for this split
                            last_row = max(split_rows, key=lambda x: int(x['epoch']))
                            final_metrics[split] = {
                                metric: float(last_row[metric]) if last_row[metric] != 'nan' else None
                                for metric in self.metrics
                            }
            
            # Create summary
            summary = {
                'experiment_info': {
                    'total_epochs': trainer.current_epoch + 1,
                    'metrics_tracked': list(self.metrics),
                    'splits_tracked': list(self.splits),
                    'completed_at': time.time(),
                },
                'final_metrics': final_metrics,
                'files': {
                    'csv_path': str(self._csv_path),
                    'json_path': str(self._json_path),
                }
            }
            
            # Atomic write for JSON
            temp_json = self._json_path.with_suffix('.tmp')
            with open(temp_json, 'w') as f:
                json.dump(summary, f, indent=2)
            temp_json.rename(self._json_path)
            
        except Exception as e:
            warnings.warn(f"Failed to save JSON summary: {e}")
