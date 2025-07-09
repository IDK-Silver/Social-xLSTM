"""
Test suite for TrainingVisualizer class.

This module tests the visualization functionality including:
- Basic plot generation
- Advanced visualization features
- Export functionality
- Error handling with empty data
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from social_xlstm.training.recorder import TrainingRecorder
from social_xlstm.visualization.training_visualizer import TrainingVisualizer

# Use non-interactive backend for testing
matplotlib.use('Agg')


class TestTrainingVisualizer:
    """Test suite for TrainingVisualizer functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_recorder(self):
        """Create a recorder with sample training data."""
        recorder = TrainingRecorder(
            experiment_name="test_experiment",
            model_config={'hidden_size': 128, 'num_layers': 2},
            training_config={'epochs': 10, 'learning_rate': 0.001}
        )
        
        # Add realistic training data
        for epoch in range(10):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 * np.exp(-0.1 * epoch) + 0.1 * np.random.rand(),
                val_loss=1.1 * np.exp(-0.08 * epoch) + 0.15 * np.random.rand(),
                train_metrics={
                    'mae': 0.5 * np.exp(-0.1 * epoch),
                    'mse': 0.25 * np.exp(-0.1 * epoch),
                    'rmse': 0.5 * np.exp(-0.1 * epoch),
                    'r2': 0.5 + 0.05 * epoch
                },
                val_metrics={
                    'mae': 0.55 * np.exp(-0.08 * epoch),
                    'mse': 0.3 * np.exp(-0.08 * epoch),
                    'rmse': 0.55 * np.exp(-0.08 * epoch),
                    'r2': 0.45 + 0.04 * epoch
                },
                learning_rate=0.001 * (0.9 ** epoch),
                epoch_time=30 + 5 * np.random.rand(),
                gradient_norm=1.0 * np.exp(-0.05 * epoch)
            )
        
        return recorder
    
    @pytest.fixture
    def empty_recorder(self):
        """Create an empty recorder for edge case testing."""
        return TrainingRecorder(
            experiment_name="empty_experiment",
            model_config={},
            training_config={}
        )
    
    @pytest.fixture
    def visualizer(self):
        """Create a TrainingVisualizer instance."""
        return TrainingVisualizer()
    
    def test_initialization(self, visualizer):
        """Test proper initialization of TrainingVisualizer."""
        assert visualizer.style == 'seaborn-v0_8'
        assert visualizer.palette == 'Set2'
    
    def test_plot_basic_training_curves(self, visualizer, sample_recorder, temp_dir):
        """Test basic training curves plotting."""
        plot_path = temp_dir / "basic_curves.png"
        
        # Generate plot
        fig = visualizer.plot_basic_training_curves(sample_recorder, save_path=plot_path)
        
        # Verify figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Verify file was saved
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
        
        # Close figure to free memory
        plt.close(fig)
    
    def test_plot_advanced_metrics(self, visualizer, sample_recorder, temp_dir):
        """Test advanced metrics plotting."""
        plot_path = temp_dir / "advanced_metrics.png"
        
        # Generate plot
        fig = visualizer.plot_advanced_metrics(sample_recorder, save_path=plot_path)
        
        assert fig is not None
        assert plot_path.exists()
        
        plt.close(fig)
    
    def test_plot_training_dashboard(self, visualizer, sample_recorder, temp_dir):
        """Test comprehensive dashboard generation."""
        plot_path = temp_dir / "dashboard.png"
        
        # Generate dashboard
        fig = visualizer.plot_training_dashboard(sample_recorder, save_path=plot_path)
        
        assert fig is not None
        assert plot_path.exists()
        
        # Dashboard should be larger
        assert plot_path.stat().st_size > 10000  # Should be a substantial file
        
        plt.close(fig)
    
    def test_plot_experiment_comparison(self, visualizer, sample_recorder, temp_dir):
        """Test experiment comparison plotting."""
        # Create a second recorder with different performance
        recorder2 = TrainingRecorder(
            experiment_name="improved_experiment",
            model_config={'hidden_size': 256, 'num_layers': 3},
            training_config={'epochs': 10, 'learning_rate': 0.0005}
        )
        
        for epoch in range(10):
            recorder2.log_epoch(
                epoch=epoch,
                train_loss=0.8 * np.exp(-0.15 * epoch),
                val_loss=0.9 * np.exp(-0.12 * epoch)
            )
        
        plot_path = temp_dir / "comparison.png"
        
        # Generate comparison plot
        fig = visualizer.plot_experiment_comparison(
            [sample_recorder, recorder2], 
            save_path=plot_path
        )
        
        assert fig is not None
        assert plot_path.exists()
        
        plt.close(fig)
    
    def test_plot_metric_evolution(self, visualizer, sample_recorder, temp_dir):
        """Test metric evolution plotting."""
        plot_path = temp_dir / "metric_evolution.png"
        
        # Plot specific metrics
        fig = visualizer.plot_metric_evolution(
            sample_recorder,
            metrics=['mae', 'mse', 'r2'],
            save_path=plot_path
        )
        
        assert fig is not None
        assert plot_path.exists()
        
        plt.close(fig)
    
    def test_create_training_report(self, visualizer, sample_recorder, temp_dir):
        """Test complete training report generation."""
        report_dir = temp_dir / "report"
        
        # Generate full report
        visualizer.create_training_report(sample_recorder, report_dir, include_all=True)
        
        # Check all expected files were created
        assert report_dir.exists()
        assert (report_dir / "basic_training_curves.png").exists()
        assert (report_dir / "training_dashboard.png").exists()
        assert (report_dir / "advanced_metrics.png").exists()
        assert (report_dir / "metric_evolution.png").exists()
        assert (report_dir / "training_summary.txt").exists()
        
        # Verify text summary has content
        summary_text = (report_dir / "training_summary.txt").read_text()
        assert "Training Report" in summary_text
        assert "test_experiment" in summary_text
    
    def test_empty_recorder_handling(self, visualizer, empty_recorder, temp_dir):
        """Test handling of empty recorder (no epochs)."""
        plot_path = temp_dir / "empty_plot.png"
        
        # Should handle empty data gracefully
        fig = visualizer.plot_basic_training_curves(empty_recorder, save_path=plot_path)
        
        assert fig is not None
        # File might be created but plot will be empty
        
        plt.close(fig)
    
    def test_partial_data_handling(self, visualizer, temp_dir):
        """Test handling of partial data (e.g., no validation loss)."""
        recorder = TrainingRecorder("partial_test", {}, {})
        
        # Add epochs with only training data
        for epoch in range(5):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),
                val_loss=None,  # No validation
                train_metrics={'mae': 0.1},
                val_metrics=None
            )
        
        plot_path = temp_dir / "partial_plot.png"
        fig = visualizer.plot_basic_training_curves(recorder, save_path=plot_path)
        
        assert fig is not None
        assert plot_path.exists()
        
        plt.close(fig)
    
    def test_gradient_norm_plot(self, visualizer, sample_recorder):
        """Test gradient norm plotting specifically."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        visualizer._plot_gradient_norm(sample_recorder, ax)
        
        # Should have plotted gradient norms
        assert len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_memory_usage_plot(self, visualizer, sample_recorder):
        """Test memory usage plotting."""
        # Add memory usage data
        for epoch in sample_recorder.epochs:
            epoch.memory_usage = 1000 + 100 * np.random.rand()
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        visualizer._plot_memory_usage(sample_recorder, ax)
        
        # Should have plotted memory usage
        assert len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_overfitting_analysis_plot(self, visualizer):
        """Test overfitting analysis visualization."""
        # Create recorder with overfitting pattern
        recorder = TrainingRecorder("overfit_test", {}, {})
        
        for epoch in range(20):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),  # Continuously decreasing
                val_loss=0.5 - 0.1 * epoch if epoch < 5 else 0.1 + 0.05 * epoch  # U-shape
            )
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        visualizer._plot_overfitting_analysis(recorder, ax)
        
        # Should show overfitting regions
        assert len(ax.lines) > 0
        assert len(ax.collections) > 0  # Fill regions
        
        plt.close(fig)
    
    def test_convergence_analysis_plot(self, visualizer, sample_recorder):
        """Test convergence analysis visualization."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        visualizer._plot_convergence_analysis(sample_recorder, ax)
        
        # Should show trend line and moving average
        assert len(ax.lines) >= 2
        
        plt.close(fig)
    
    def test_custom_style(self, temp_dir):
        """Test custom style and palette."""
        custom_visualizer = TrainingVisualizer(style='ggplot', palette='husl')
        
        assert custom_visualizer.style == 'ggplot'
        assert custom_visualizer.palette == 'husl'
    
    def test_best_epoch_highlighting(self, visualizer, sample_recorder):
        """Test that best epoch is highlighted in plots."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        visualizer._plot_loss_curves(sample_recorder, ax)
        
        # Should have scatter point for best epoch
        scatter_collections = [c for c in ax.collections if hasattr(c, 'get_offsets')]
        assert len(scatter_collections) > 0  # Best epoch marker
        
        plt.close(fig)
    
    def test_text_report_content(self, visualizer, sample_recorder, temp_dir):
        """Test text report generation content."""
        report_path = temp_dir / "summary.txt"
        
        visualizer._generate_text_report(sample_recorder, report_path)
        
        assert report_path.exists()
        
        content = report_path.read_text()
        
        # Check key sections
        assert "Training Report" in content
        assert "Experiment Metadata" in content
        assert "Training Summary" in content
        assert "Model Configuration" in content
        assert "Training Configuration" in content
        
        # Check specific values
        assert "hidden_size: 128" in content
        assert "epochs: 10" in content
    
    def test_figure_cleanup(self, visualizer, sample_recorder, temp_dir):
        """Test that figures are properly saved and don't accumulate in memory."""
        initial_figs = plt.get_fignums()
        
        # Generate multiple plots
        for i in range(5):
            plot_path = temp_dir / f"plot_{i}.png"
            fig = visualizer.plot_basic_training_curves(sample_recorder, save_path=plot_path)
            plt.close(fig)
        
        # Should not have accumulated figures
        final_figs = plt.get_fignums()
        assert len(final_figs) <= len(initial_figs) + 1
    
    def test_plot_with_single_metric(self, visualizer):
        """Test plotting with only one metric type."""
        recorder = TrainingRecorder("single_metric_test", {}, {})
        
        # Only MAE metric
        for epoch in range(5):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),
                train_metrics={'mae': 0.1 / (epoch + 1)}
            )
        
        # Should handle single metric gracefully
        fig = visualizer.plot_metric_evolution(recorder, metrics=['mae'])
        assert fig is not None
        
        plt.close(fig)
    
    def test_comparison_with_different_epoch_counts(self, visualizer, temp_dir):
        """Test comparing experiments with different numbers of epochs."""
        # First recorder: 10 epochs
        recorder1 = TrainingRecorder("exp1", {}, {})
        for epoch in range(10):
            recorder1.log_epoch(epoch, train_loss=1.0/(epoch+1))
        
        # Second recorder: 5 epochs
        recorder2 = TrainingRecorder("exp2", {}, {})
        for epoch in range(5):
            recorder2.log_epoch(epoch, train_loss=0.8/(epoch+1))
        
        plot_path = temp_dir / "different_epochs.png"
        fig = visualizer.plot_experiment_comparison([recorder1, recorder2], save_path=plot_path)
        
        assert fig is not None
        assert plot_path.exists()
        
        plt.close(fig)