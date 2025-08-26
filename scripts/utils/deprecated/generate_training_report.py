#!/usr/bin/env python3
"""
Training Report Generator

Generate comprehensive training reports from training_history.json files.
Supports multiple experiments and comparison analysis.

Author: Social-xLSTM Project Team
License: MIT
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_training_history(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    """Load training history from experiment directory."""
    history_path = experiment_dir / "training_history.json"
    
    if not history_path.exists():
        logger.warning(f"No training history found in {experiment_dir}")
        return None
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded training history from {history_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {history_path}: {e}")
        return None


def format_metrics_table(epochs: List[Dict]) -> str:
    """Format epoch metrics as a markdown table."""
    if not epochs:
        return "No epoch data available.\n"
    
    # Header
    table = "| Epoch | Train Loss | Val Loss | MAE | MSE | RMSE | MAPE | R² | Learning Rate |\n"
    table += "|-------|------------|----------|-----|-----|------|------|----|--------------|\n"
    
    # Data rows
    for epoch_data in epochs:
        epoch = epoch_data.get('epoch', 'N/A')
        train_loss = f"{epoch_data.get('train_loss', 0):.6f}"
        val_loss = f"{epoch_data.get('val_loss', 0):.6f}" if epoch_data.get('val_loss') else "N/A"
        
        train_metrics = epoch_data.get('train_metrics', {})
        mae = train_metrics.get('mae', 'N/A')
        mse = train_metrics.get('mse', 'N/A')
        rmse = train_metrics.get('rmse', 'N/A')
        mape = train_metrics.get('mape', 'N/A')
        r2 = train_metrics.get('r2', 'N/A')
        
        lr = f"{epoch_data.get('learning_rate', 0):.6f}"
        
        table += f"| {epoch} | {train_loss} | {val_loss} | {mae} | {mse} | {rmse} | {mape} | {r2} | {lr} |\n"
    
    return table


def format_system_info(metadata: Dict) -> str:
    """Format system information as markdown."""
    if not metadata:
        return "No system information available.\n"
    
    device_info = metadata.get('device_info', {})
    system_info = metadata.get('system_info', {})
    
    info = "## System Information\n\n"
    info += f"- **Python Version**: {metadata.get('python_version', 'N/A')}\n"
    info += f"- **PyTorch Version**: {metadata.get('pytorch_version', 'N/A')}\n"
    info += f"- **CUDA Version**: {metadata.get('cuda_version', 'N/A')}\n"
    info += f"- **Git Commit**: `{metadata.get('git_commit', 'N/A')}`\n\n"
    
    if device_info:
        info += "### GPU Information\n"
        info += f"- **CUDA Available**: {device_info.get('cuda_available', 'N/A')}\n"
        info += f"- **Device Count**: {device_info.get('cuda_device_count', 'N/A')}\n"
        info += f"- **Device Name**: {device_info.get('cuda_device_name', 'N/A')}\n"
        memory_gb = device_info.get('cuda_memory_total', 0) / (1024**3) if device_info.get('cuda_memory_total') else 0
        info += f"- **Memory Total**: {memory_gb:.1f} GB\n\n"
    
    if system_info:
        info += "### System Resources\n"
        info += f"- **CPU Count**: {system_info.get('cpu_count', 'N/A')}\n"
        memory_gb = system_info.get('memory_total', 0) / (1024**3) if system_info.get('memory_total') else 0
        info += f"- **Memory Total**: {memory_gb:.1f} GB\n"
        info += f"- **Platform**: {system_info.get('platform', 'N/A')}\n\n"
    
    return info


def format_training_summary(summary: Dict) -> str:
    """Format training summary as markdown."""
    if not summary:
        return "No training summary available.\n"
    
    info = "## Training Summary\n\n"
    info += f"- **Total Epochs**: {summary.get('total_epochs', 'N/A')}\n"
    info += f"- **Total Time**: {summary.get('total_time', 'N/A'):.2f} seconds\n" if summary.get('total_time') else "- **Total Time**: N/A\n"
    info += f"- **Average Epoch Time**: {summary.get('avg_epoch_time', 'N/A'):.2f} seconds\n" if summary.get('avg_epoch_time') else "- **Average Epoch Time**: N/A\n"
    info += f"- **Best Epoch**: {summary.get('best_epoch', 'N/A')}\n"
    info += f"- **Best Train Loss**: {summary.get('best_train_loss', 'N/A'):.6f}\n" if summary.get('best_train_loss') else "- **Best Train Loss**: N/A\n"
    info += f"- **Best Val Loss**: {summary.get('best_val_loss', 'N/A'):.6f}\n" if summary.get('best_val_loss') else "- **Best Val Loss**: N/A\n"
    info += f"- **Final Learning Rate**: {summary.get('final_learning_rate', 'N/A'):.6f}\n" if summary.get('final_learning_rate') else "- **Final Learning Rate**: N/A\n"
    
    # Stability analysis
    stability = summary.get('stability_analysis', {})
    if stability:
        info += "\n### Stability Analysis\n"
        info += f"- **Training Trend**: {stability.get('train_trend', 'N/A'):.6f}\n" if stability.get('train_trend') else "- **Training Trend**: N/A\n"
        info += f"- **Validation Trend**: {stability.get('val_trend', 'N/A'):.6f}\n" if stability.get('val_trend') else "- **Validation Trend**: N/A\n"
        info += f"- **Training Volatility**: {stability.get('train_volatility', 'N/A'):.6f}\n" if stability.get('train_volatility') else "- **Training Volatility**: N/A\n"
        info += f"- **Validation Volatility**: {stability.get('val_volatility', 'N/A'):.6f}\n" if stability.get('val_volatility') else "- **Validation Volatility**: N/A\n"
        info += f"- **Overfitting Score**: {stability.get('overfitting_score', 'N/A'):.6f}\n" if stability.get('overfitting_score') else "- **Overfitting Score**: N/A\n"
        info += f"- **Convergence Status**: {stability.get('convergence_status', 'N/A')}\n"
    
    return info + "\n"


def format_configuration(training_config: Dict, model_config: str) -> str:
    """Format model and training configuration as markdown."""
    info = "## Configuration\n\n"
    
    info += "### Model Configuration\n"
    info += f"```\n{model_config}\n```\n\n"
    
    if training_config:
        info += "### Training Configuration\n"
        info += f"- **Epochs**: {training_config.get('epochs', 'N/A')}\n"
        info += f"- **Batch Size**: {training_config.get('batch_size', 'N/A')}\n"
        info += f"- **Learning Rate**: {training_config.get('learning_rate', 'N/A')}\n"
        info += f"- **Optimizer**: {training_config.get('optimizer_type', 'N/A')}\n"
        info += f"- **Weight Decay**: {training_config.get('weight_decay', 'N/A')}\n"
        info += f"- **Device**: {training_config.get('device', 'N/A')}\n"
        info += f"- **Mixed Precision**: {training_config.get('mixed_precision', 'N/A')}\n"
        info += f"- **Gradient Clip**: {training_config.get('gradient_clip_value', 'N/A')}\n"
        
        if training_config.get('use_scheduler'):
            info += f"- **Scheduler**: {training_config.get('scheduler_type', 'N/A')}\n"
            info += f"- **Scheduler Patience**: {training_config.get('scheduler_patience', 'N/A')}\n"
            info += f"- **Scheduler Factor**: {training_config.get('scheduler_factor', 'N/A')}\n"
    
    return info + "\n"


def generate_single_experiment_report(experiment_dir: Path) -> str:
    """Generate report for a single experiment."""
    history = load_training_history(experiment_dir)
    if not history:
        return f"# Training Report - {experiment_dir.name}\n\n❌ **Error**: Could not load training history.\n"
    
    # Header
    report = f"# Training Report - {history.get('experiment_name', experiment_dir.name)}\n\n"
    report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Experiment Directory**: `{experiment_dir}`\n\n"
    
    # Training metrics table
    report += "## Training Progress\n\n"
    epochs = history.get('epochs', [])
    if epochs:
        report += format_metrics_table(epochs)
        report += "\n"
        
        # Performance improvement
        if len(epochs) > 1:
            first_loss = epochs[0].get('train_loss', 0)
            last_loss = epochs[-1].get('train_loss', 0)
            improvement = ((first_loss - last_loss) / first_loss * 100) if first_loss > 0 else 0
            report += f"**Training Loss Improvement**: {improvement:.1f}% (from {first_loss:.6f} to {last_loss:.6f})\n\n"
    else:
        report += "No training data available.\n\n"
    
    # Configuration
    report += format_configuration(
        history.get('training_config', {}),
        history.get('model_config', 'N/A')
    )
    
    # Training summary
    report += format_training_summary(history.get('training_summary', {}))
    
    # System information
    report += format_system_info(history.get('experiment_metadata', {}))
    
    # Timestamps
    report += "## Timeline\n\n"
    report += f"- **Created**: {history.get('created_at', 'N/A')}\n"
    report += f"- **Saved**: {history.get('saved_at', 'N/A')}\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Generate training reports from experiment results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory containing training_history.json')
    parser.add_argument('--output_file', type=str,
                       help='Output markdown file path (default: experiment_dir/training_report.md)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate experiment directory
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory does not exist: {experiment_dir}")
        sys.exit(1)
    
    if not experiment_dir.is_dir():
        logger.error(f"Path is not a directory: {experiment_dir}")
        sys.exit(1)
    
    # Generate report
    logger.info(f"Generating report for experiment: {experiment_dir}")
    report = generate_single_experiment_report(experiment_dir)
    
    # Determine output file
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = experiment_dir / "training_report.md"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write report
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Training report generated: {output_path}")
        print(f"Training report saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write report to {output_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()