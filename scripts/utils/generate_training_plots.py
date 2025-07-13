#!/usr/bin/env python3
"""
Training Visualization Generator

Generate comprehensive training plots from training_history.json files.
Uses TrainingVisualizer to create various charts and dashboards.

Author: Social-xLSTM Project Team
License: MIT
"""

import argparse
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from social_xlstm.training.recorder import TrainingRecorder
    from social_xlstm.visualization.training_visualizer import TrainingVisualizer
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're in the Social-xLSTM environment and the package is installed")
    sys.exit(1)


def generate_plots_from_history(experiment_dir: Path, output_dir: Path = None):
    """Generate training plots from experiment directory."""
    
    # Default output directory
    if output_dir is None:
        output_dir = experiment_dir / "plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training history
    history_path = experiment_dir / "training_history.json"
    if not history_path.exists():
        logger.error(f"No training history found at {history_path}")
        return False
    
    try:
        # Load the training recorder
        recorder = TrainingRecorder.load(history_path)
        logger.info(f"Loaded training history from {history_path}")
        
        # Create visualizer
        visualizer = TrainingVisualizer()
        logger.info("Created training visualizer")
        
        # Generate different types of plots
        plots_generated = []
        
        # 1. Training Dashboard (comprehensive overview)
        try:
            dashboard_path = output_dir / "training_dashboard.png"
            fig = visualizer.plot_training_dashboard(recorder, save_path=dashboard_path)
            logger.info(f"Generated training dashboard: {dashboard_path}")
            plots_generated.append(str(dashboard_path))
        except Exception as e:
            logger.warning(f"Failed to generate training dashboard: {e}")
        
        # 2. Basic Training Curves
        try:
            curves_path = output_dir / "training_curves.png"
            fig = visualizer.plot_basic_training_curves(recorder, save_path=curves_path)
            logger.info(f"Generated training curves: {curves_path}")
            plots_generated.append(str(curves_path))
        except Exception as e:
            logger.warning(f"Failed to generate training curves: {e}")
        
        # 3. Metric Evolution (if available)
        if recorder.epochs and len(recorder.epochs) > 1:
            try:
                metrics_path = output_dir / "metric_evolution.png"
                fig = visualizer.plot_metric_evolution(recorder, save_path=metrics_path)
                logger.info(f"Generated metric evolution: {metrics_path}")
                plots_generated.append(str(metrics_path))
            except Exception as e:
                logger.warning(f"Failed to generate metric evolution: {e}")
        
        # 4. Advanced Metrics (if available)
        try:
            advanced_path = output_dir / "advanced_metrics.png"
            fig = visualizer.plot_advanced_metrics(recorder, save_path=advanced_path)
            logger.info(f"Generated advanced metrics: {advanced_path}")
            plots_generated.append(str(advanced_path))
        except Exception as e:
            logger.warning(f"Failed to generate advanced metrics: {e}")
        
        if plots_generated:
            logger.info(f"Successfully generated {len(plots_generated)} plots:")
            for plot_path in plots_generated:
                logger.info(f"  - {plot_path}")
            return True
        else:
            logger.error("No plots were generated successfully")
            return False
            
    except Exception as e:
        logger.error(f"Failed to load training history: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate training visualization plots')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory containing training_history.json')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for plots (default: experiment_dir/plots)')
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
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Generate plots
    logger.info(f"Generating training plots for experiment: {experiment_dir}")
    success = generate_plots_from_history(experiment_dir, output_dir)
    
    if success:
        logger.info("Plot generation completed successfully")
        print("Training plots generated successfully!")
    else:
        logger.error("Plot generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()