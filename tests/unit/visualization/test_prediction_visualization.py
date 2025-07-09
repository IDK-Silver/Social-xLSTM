"""
Test prediction visualization features integrated into TrainingVisualizer.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from social_xlstm.visualization.training_visualizer import TrainingVisualizer


@pytest.mark.unit
class TestPredictionVisualization:
    """Test prediction visualization methods in TrainingVisualizer."""
    
    @pytest.fixture
    def sample_prediction_data(self):
        """Generate sample prediction data."""
        np.random.seed(42)
        
        # Create sample data
        num_samples = 10
        seq_length = 12
        pred_length = 1
        num_features = 3
        
        # Generate realistic predictions and targets
        predictions = np.random.normal(0.5, 0.1, (num_samples, pred_length, num_features))
        targets = np.random.normal(0.5, 0.1, (num_samples, pred_length, num_features))
        inputs = np.random.normal(0.5, 0.1, (num_samples, seq_length, num_features))
        
        # Add some correlation between predictions and targets
        noise = np.random.normal(0, 0.05, targets.shape)
        predictions = 0.8 * targets + 0.2 * predictions + noise
        
        return {
            'predictions': predictions,
            'targets': targets,
            'inputs': inputs,
            'feature_names': ['volume', 'speed', 'occupancy'],
            'metrics': {
                'mae': 0.05,
                'mse': 0.003,
                'rmse': 0.055,
                'r2': 0.85
            }
        }
    
    @pytest.fixture
    def visualizer(self):
        """Create TrainingVisualizer instance."""
        return TrainingVisualizer()
    
    def test_plot_time_series_comparison(self, visualizer, sample_prediction_data):
        """Test time series comparison plot."""
        fig = visualizer.plot_time_series_comparison(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            inputs=sample_prediction_data['inputs'],
            feature_names=sample_prediction_data['feature_names'],
            sample_idx=0,
            feature_idx=0
        )
        
        assert fig is not None
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_title().startswith('Time Series Prediction')
        assert ax.get_xlabel() == 'Time Step'
        assert ax.get_ylabel() == 'Normalized Value'
        
        # Check that lines are plotted
        lines = ax.get_lines()
        assert len(lines) >= 3  # Input history, ground truth, predicted
        
        plt.close(fig)
    
    def test_plot_prediction_scatter(self, visualizer, sample_prediction_data):
        """Test prediction scatter plot."""
        fig = visualizer.plot_prediction_scatter(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            metrics=sample_prediction_data['metrics']
        )
        
        assert fig is not None
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_title() == 'Predictions vs Ground Truth'
        assert ax.get_xlabel() == 'Ground Truth'
        assert ax.get_ylabel() == 'Predictions'
        
        # Check scatter plot exists
        collections = ax.collections
        assert len(collections) > 0  # Should have scatter plot
        
        plt.close(fig)
    
    def test_plot_feature_performance(self, visualizer, sample_prediction_data):
        """Test feature performance plot."""
        fig = visualizer.plot_feature_performance(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            feature_names=sample_prediction_data['feature_names']
        )
        
        assert fig is not None
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_title() == 'Mean Absolute Error by Feature'
        assert ax.get_xlabel() == 'Feature'
        assert ax.get_ylabel() == 'MAE'
        
        # Check bar plot
        bars = ax.patches
        assert len(bars) == len(sample_prediction_data['feature_names'])
        
        plt.close(fig)
    
    def test_plot_prediction_dashboard(self, visualizer, sample_prediction_data):
        """Test prediction dashboard."""
        fig = visualizer.plot_prediction_dashboard(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            inputs=sample_prediction_data['inputs'],
            feature_names=sample_prediction_data['feature_names'],
            metrics=sample_prediction_data['metrics']
        )
        
        assert fig is not None
        assert len(fig.axes) == 4  # 4 subplots in dashboard
        
        # Check main title
        assert fig._suptitle.get_text() == 'Prediction Evaluation Dashboard'
        
        plt.close(fig)
    
    def test_single_feature_plotting(self, visualizer):
        """Test plotting with single feature."""
        np.random.seed(42)
        
        # Single feature data
        predictions = np.random.normal(0.5, 0.1, (5, 1, 1))
        targets = np.random.normal(0.5, 0.1, (5, 1, 1))
        inputs = np.random.normal(0.5, 0.1, (5, 12, 1))
        
        fig = visualizer.plot_prediction_dashboard(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            feature_names=['volume'],
            metrics={'mae': 0.05, 'r2': 0.8}
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_multiple_features_plotting(self, visualizer, sample_prediction_data):
        """Test plotting with multiple features."""
        # Test with all 3 features
        fig = visualizer.plot_prediction_dashboard(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            inputs=sample_prediction_data['inputs'],
            feature_names=sample_prediction_data['feature_names'],
            metrics=sample_prediction_data['metrics']
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_save_functionality(self, visualizer, sample_prediction_data, temp_dir):
        """Test save functionality."""
        save_path = temp_dir / "test_plot.png"
        
        fig = visualizer.plot_time_series_comparison(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            inputs=sample_prediction_data['inputs'],
            feature_names=sample_prediction_data['feature_names'],
            save_path=save_path
        )
        
        # Check that file was created
        assert save_path.exists()
        plt.close(fig)
    
    def test_edge_cases(self, visualizer):
        """Test edge cases."""
        # Very small dataset
        predictions = np.random.normal(0.5, 0.1, (2, 1, 1))
        targets = np.random.normal(0.5, 0.1, (2, 1, 1))
        inputs = np.random.normal(0.5, 0.1, (2, 5, 1))
        
        fig = visualizer.plot_time_series_comparison(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            feature_names=['test'],
            sample_idx=0,
            feature_idx=0
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_helper_methods(self, visualizer, sample_prediction_data):
        """Test helper methods."""
        fig, ax = plt.subplots()
        
        # Test helper methods
        visualizer._plot_single_time_series(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            inputs=sample_prediction_data['inputs'],
            feature_names=sample_prediction_data['feature_names'],
            ax=ax
        )
        
        assert ax.get_title().startswith('Time Series')
        plt.close(fig)
        
        # Test another helper
        fig, ax = plt.subplots()
        visualizer._plot_single_prediction_scatter(
            predictions=sample_prediction_data['predictions'],
            targets=sample_prediction_data['targets'],
            metrics=sample_prediction_data['metrics'],
            ax=ax
        )
        
        assert ax.get_title() == 'Predictions vs Ground Truth'
        plt.close(fig)