#!/usr/bin/env python3
"""
Basic Metrics Plotting Tool

Generate simple training plots from metrics CSV files created by TrainingMetricsWriter.
Focuses on MAE, MSE, RMSE, R² visualization without complex dashboards.

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
    from social_xlstm.metrics.plotter import TrainingMetricsPlotter
except ImportError as e:
    logger.error(f"Failed to import TrainingMetricsPlotter: {e}")
    logger.error("Make sure you're in the Social-xLSTM environment and the package is installed")
    sys.exit(1)


def generate_plots_from_csv(
    csv_path: Path, 
    output_dir: Path = None,
    plot_individual: bool = True,
    plot_combined: bool = True
):
    """Generate training plots from CSV file."""
    
    # Default output directory
    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create plotter
        plotter = TrainingMetricsPlotter(csv_path)
        logger.info(f"Loaded metrics from {csv_path}")
        
        # Print summary
        plotter.print_summary()
        
        plots_generated = []
        
        # Generate combined plot
        if plot_combined:
            try:
                combined_path = output_dir / "all_metrics.png"
                fig = plotter.plot_all_metrics(save_path=combined_path)
                logger.info(f"Generated combined plot: {combined_path}")
                plots_generated.append(str(combined_path))
                fig.close()  # Free memory
            except Exception as e:
                logger.warning(f"Failed to generate combined plot: {e}")
        
        # Generate individual metric plots
        if plot_individual:
            for metric in plotter.metrics:
                try:
                    metric_path = output_dir / f"{metric}_plot.png"
                    fig = plotter.plot_single_metric(metric, save_path=metric_path)
                    logger.info(f"Generated {metric} plot: {metric_path}")
                    plots_generated.append(str(metric_path))
                    fig.close()  # Free memory
                except Exception as e:
                    logger.warning(f"Failed to generate {metric} plot: {e}")
        
        if plots_generated:
            logger.info(f"Successfully generated {len(plots_generated)} plots:")
            for plot_path in plots_generated:
                logger.info(f"  - {plot_path}")
            return True
        else:
            logger.error("No plots were generated successfully")
            return False
            
    except Exception as e:
        logger.error(f"Failed to create plotter: {e}")
        return False


def generate_plots_from_experiment_dir(
    experiment_dir: Path, 
    output_dir: Path = None,
    csv_filename: str = "metrics.csv",
    plot_individual: bool = True,
    plot_combined: bool = True
):
    """Generate plots from experiment directory containing metrics.csv."""
    
    csv_path = experiment_dir / csv_filename
    if not csv_path.exists():
        logger.error(f"No metrics CSV found at {csv_path}")
        return False
    
    # Default output directory inside experiment dir
    if output_dir is None:
        output_dir = experiment_dir / "plots"
    
    return generate_plots_from_csv(
        csv_path, output_dir, plot_individual, plot_combined
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate basic training metrics plots from CSV data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--csv_path', type=str,
        help='Direct path to metrics CSV file'
    )
    input_group.add_argument(
        '--experiment_dir', type=str,
        help='Path to experiment directory containing metrics.csv'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir', type=str,
        help='Output directory for plots (default: auto-determined based on input)'
    )
    parser.add_argument(
        '--csv_filename', type=str, default='metrics.csv',
        help='Name of CSV file when using --experiment_dir'
    )
    
    # Plot options
    parser.add_argument(
        '--skip_individual', action='store_true',
        help='Skip generating individual metric plots'
    )
    parser.add_argument(
        '--skip_combined', action='store_true',
        help='Skip generating combined metrics plot'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Generate plots based on input type
    if args.csv_path:
        # Direct CSV path
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            logger.error(f"CSV file does not exist: {csv_path}")
            sys.exit(1)
        
        logger.info(f"Generating plots from CSV: {csv_path}")
        success = generate_plots_from_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            plot_individual=not args.skip_individual,
            plot_combined=not args.skip_combined
        )
        
    else:
        # Experiment directory
        experiment_dir = Path(args.experiment_dir)
        if not experiment_dir.exists():
            logger.error(f"Experiment directory does not exist: {experiment_dir}")
            sys.exit(1)
        
        if not experiment_dir.is_dir():
            logger.error(f"Path is not a directory: {experiment_dir}")
            sys.exit(1)
        
        logger.info(f"Generating plots from experiment directory: {experiment_dir}")
        success = generate_plots_from_experiment_dir(
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            csv_filename=args.csv_filename,
            plot_individual=not args.skip_individual,
            plot_combined=not args.skip_combined
        )
    
    if success:
        logger.info("Plot generation completed successfully")
        print("Training metrics plots generated successfully!")
    else:
        logger.error("Plot generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()