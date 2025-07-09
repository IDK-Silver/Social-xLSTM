"""
Test suite for TrainingRecorder class.

This module tests the training recording functionality including:
- Basic recording operations
- Data persistence (save/load)
- Export functionality (CSV, TensorBoard)
- Analysis methods
- Edge cases and error handling
"""

import pytest
import json
import csv
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

from social_xlstm.training.recorder import TrainingRecorder, EpochRecord


class TestTrainingRecorder:
    """Test suite for TrainingRecorder functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        model_config = {
            'input_size': 3,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        training_config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        return model_config, training_config
    
    @pytest.fixture
    def recorder(self, sample_config):
        """Create a TrainingRecorder instance with sample data."""
        model_config, training_config = sample_config
        return TrainingRecorder("test_experiment", model_config, training_config)
    
    @pytest.fixture
    def recorder_with_data(self, recorder):
        """Create a recorder with some logged epochs."""
        # Add 5 epochs of data
        for epoch in range(5):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),  # Decreasing loss
                val_loss=1.2 / (epoch + 1),
                train_metrics={'mae': 0.1 / (epoch + 1), 'mse': 0.01 / (epoch + 1)},
                val_metrics={'mae': 0.12 / (epoch + 1), 'mse': 0.014 / (epoch + 1)},
                learning_rate=0.001 * (0.9 ** epoch),  # Decaying LR
                epoch_time=30.0 + np.random.rand() * 10,
                gradient_norm=0.5 / (epoch + 1)
            )
        return recorder
    
    def test_initialization(self, recorder, sample_config):
        """Test proper initialization of TrainingRecorder."""
        model_config, training_config = sample_config
        
        assert recorder.experiment_name == "test_experiment"
        assert recorder.model_config == model_config
        assert recorder.training_config == training_config
        assert len(recorder.epochs) == 0
        assert recorder.best_epoch is None
        assert recorder.best_val_loss == float('inf')
        assert isinstance(recorder.start_time, datetime)
        assert 'python_version' in recorder.experiment_metadata
        assert 'pytorch_version' in recorder.experiment_metadata
    
    def test_log_epoch_basic(self, recorder):
        """Test basic epoch logging functionality."""
        recorder.log_epoch(
            epoch=0,
            train_loss=1.0,
            val_loss=1.1,
            learning_rate=0.001,
            epoch_time=30.5
        )
        
        assert len(recorder.epochs) == 1
        epoch_record = recorder.epochs[0]
        
        assert epoch_record.epoch == 0
        assert epoch_record.train_loss == 1.0
        assert epoch_record.val_loss == 1.1
        assert epoch_record.learning_rate == 0.001
        assert epoch_record.epoch_time == 30.5
        assert epoch_record.is_best is True  # First epoch is best
        assert recorder.best_epoch == 0
    
    def test_log_epoch_with_metrics(self, recorder):
        """Test epoch logging with metrics."""
        train_metrics = {'mae': 0.1, 'mse': 0.01, 'rmse': 0.1}
        val_metrics = {'mae': 0.12, 'mse': 0.014, 'rmse': 0.118}
        
        recorder.log_epoch(
            epoch=0,
            train_loss=1.0,
            val_loss=1.1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            gradient_norm=0.5
        )
        
        epoch_record = recorder.epochs[0]
        assert epoch_record.train_metrics == train_metrics
        assert epoch_record.val_metrics == val_metrics
        assert epoch_record.gradient_norm == 0.5
    
    def test_best_epoch_tracking(self, recorder):
        """Test that best epoch is properly tracked."""
        # Log epochs with decreasing validation loss
        recorder.log_epoch(0, train_loss=1.0, val_loss=1.0)
        assert recorder.best_epoch == 0
        
        recorder.log_epoch(1, train_loss=0.8, val_loss=0.9)
        assert recorder.best_epoch == 1
        
        recorder.log_epoch(2, train_loss=0.7, val_loss=0.95)  # Worse than previous
        assert recorder.best_epoch == 1  # Should remain 1
        
        recorder.log_epoch(3, train_loss=0.6, val_loss=0.85)  # Better again
        assert recorder.best_epoch == 3
    
    def test_get_metric_history(self, recorder_with_data):
        """Test retrieving metric history."""
        # Test train metrics
        train_mae = recorder_with_data.get_metric_history('mae', 'train')
        assert len(train_mae) == 5
        assert all(train_mae[i] > train_mae[i+1] for i in range(4))  # Decreasing
        
        # Test val metrics
        val_mae = recorder_with_data.get_metric_history('mae', 'val')
        assert len(val_mae) == 5
        
        # Test non-existent metric
        train_fake = recorder_with_data.get_metric_history('fake_metric', 'train')
        assert all(x == 0 for x in train_fake)
        
        # Test invalid split
        with pytest.raises(ValueError):
            recorder_with_data.get_metric_history('mae', 'test')
    
    def test_get_loss_history(self, recorder_with_data):
        """Test retrieving loss history."""
        train_losses, val_losses = recorder_with_data.get_loss_history()
        
        assert len(train_losses) == 5
        assert len(val_losses) == 5
        assert all(train_losses[i] > train_losses[i+1] for i in range(4))
        assert all(val_losses[i] > val_losses[i+1] for i in range(4))
    
    def test_get_best_epoch(self, recorder_with_data):
        """Test retrieving best epoch."""
        best_epoch = recorder_with_data.get_best_epoch()
        
        assert best_epoch is not None
        assert best_epoch.epoch == 4  # Last epoch should be best (lowest loss)
        assert best_epoch.is_best is True
    
    def test_get_training_summary(self, recorder_with_data):
        """Test training summary generation."""
        summary = recorder_with_data.get_training_summary()
        
        assert summary['total_epochs'] == 5
        assert summary['best_epoch'] == 4
        assert 'total_time' in summary
        assert 'avg_epoch_time' in summary
        assert 'best_train_loss' in summary
        assert 'best_val_loss' in summary
        assert 'final_learning_rate' in summary
        assert 'convergence_info' in summary
        assert 'stability_analysis' in summary
    
    def test_save_and_load(self, recorder_with_data, temp_dir):
        """Test saving and loading functionality."""
        save_path = temp_dir / "test_record.json"
        
        # Save
        recorder_with_data.save(save_path)
        assert save_path.exists()
        
        # Load
        loaded_recorder = TrainingRecorder.load(save_path)
        
        # Verify loaded data matches original
        assert loaded_recorder.experiment_name == recorder_with_data.experiment_name
        assert loaded_recorder.model_config == recorder_with_data.model_config
        assert loaded_recorder.training_config == recorder_with_data.training_config
        assert len(loaded_recorder.epochs) == len(recorder_with_data.epochs)
        assert loaded_recorder.best_epoch == recorder_with_data.best_epoch
        
        # Verify epoch data
        for orig, loaded in zip(recorder_with_data.epochs, loaded_recorder.epochs):
            assert orig.epoch == loaded.epoch
            assert orig.train_loss == loaded.train_loss
            assert orig.val_loss == loaded.val_loss
            assert orig.train_metrics == loaded.train_metrics
    
    def test_export_to_csv(self, recorder_with_data, temp_dir):
        """Test CSV export functionality."""
        csv_path = temp_dir / "test_history.csv"
        
        # Export
        recorder_with_data.export_to_csv(csv_path)
        assert csv_path.exists()
        
        # Read and verify CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 5  # 5 epochs
        
        # Check first row
        first_row = rows[0]
        assert 'epoch' in first_row
        assert 'train_loss' in first_row
        assert 'val_loss' in first_row
        assert 'learning_rate' in first_row
        assert 'train_mae' in first_row
        assert 'val_mae' in first_row
        
        # Verify values
        assert int(first_row['epoch']) == 0
        assert float(first_row['train_loss']) == pytest.approx(1.0)
    
    def test_export_to_tensorboard(self, recorder_with_data, temp_dir):
        """Test TensorBoard export functionality."""
        tb_dir = temp_dir / "tensorboard"
        
        # Export (will print message if TensorBoard not available)
        recorder_with_data.export_to_tensorboard(tb_dir)
        
        # Basic check - directory should be created
        assert tb_dir.exists()
    
    def test_analyze_training_stability(self, recorder_with_data):
        """Test training stability analysis."""
        stability = recorder_with_data.analyze_training_stability()
        
        assert 'train_trend' in stability
        assert 'val_trend' in stability
        assert 'train_volatility' in stability
        assert 'val_volatility' in stability
        assert 'overfitting_score' in stability
        assert 'convergence_status' in stability
        
        # Trends should be negative (loss decreasing)
        assert stability['train_trend'] < 0
        assert stability['val_trend'] < 0
    
    def test_compare_with(self, recorder_with_data):
        """Test experiment comparison functionality."""
        # Create another recorder with different data
        recorder2 = TrainingRecorder("test_experiment_2", {}, {})
        for epoch in range(3):
            recorder2.log_epoch(
                epoch=epoch,
                train_loss=2.0 / (epoch + 1),  # Higher loss
                val_loss=2.2 / (epoch + 1)
            )
        
        comparison = recorder_with_data.compare_with(recorder2)
        
        assert 'experiment_names' in comparison
        assert comparison['experiment_names'] == ["test_experiment", "test_experiment_2"]
        assert 'total_epochs' in comparison
        assert comparison['total_epochs'] == [5, 3]
        assert 'best_val_loss' in comparison
        assert 'convergence_comparison' in comparison
    
    def test_empty_recorder_operations(self, recorder):
        """Test operations on empty recorder."""
        # Get loss history from empty recorder
        train_losses, val_losses = recorder.get_loss_history()
        assert train_losses == []
        assert val_losses == []
        
        # Get best epoch from empty recorder
        assert recorder.get_best_epoch() is None
        
        # Get training summary from empty recorder
        summary = recorder.get_training_summary()
        assert summary == {}
        
        # Analyze stability of empty recorder
        stability = recorder.analyze_training_stability()
        assert stability == {}
    
    def test_nan_and_inf_handling(self, recorder):
        """Test handling of NaN and infinity values."""
        # Log epoch with NaN
        recorder.log_epoch(
            epoch=0,
            train_loss=float('nan'),
            val_loss=float('inf')
        )
        
        # Should still record the epoch
        assert len(recorder.epochs) == 1
        assert np.isnan(recorder.epochs[0].train_loss)
        assert np.isinf(recorder.epochs[0].val_loss)
    
    def test_memory_usage_recording(self, recorder):
        """Test that memory usage is automatically recorded."""
        recorder.log_epoch(epoch=0, train_loss=1.0)
        
        # Memory usage should be recorded (or None if psutil fails)
        epoch = recorder.epochs[0]
        assert epoch.memory_usage is None or isinstance(epoch.memory_usage, float)
    
    def test_sample_predictions_storage(self, recorder):
        """Test storing sample predictions."""
        sample_preds = {
            'inputs': np.random.rand(5, 12, 3).tolist(),
            'predictions': np.random.rand(5, 1, 3).tolist(),
            'targets': np.random.rand(5, 1, 3).tolist()
        }
        
        recorder.log_epoch(
            epoch=0,
            train_loss=1.0,
            sample_predictions=sample_preds
        )
        
        assert recorder.epochs[0].sample_predictions == sample_preds
    
    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        # Create recorder with more epochs for convergence analysis
        recorder = TrainingRecorder("convergence_test", {}, {})
        
        # Add 15 epochs to have enough data
        for epoch in range(15):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),  # Clearly decreasing
                val_loss=1.1 / (epoch + 1)
            )
        
        # Private method but important to test
        convergence = recorder._analyze_convergence()
        
        assert 'status' in convergence
        assert convergence['status'] in ['converging', 'plateaued', 'insufficient_data']
        
        # With decreasing loss, should be converging
        assert convergence['status'] == 'converging'
        assert 'improvement' in convergence
        assert convergence['improvement'] > 0
    
    def test_overfitting_detection(self):
        """Test overfitting detection."""
        recorder = TrainingRecorder("overfit_test", {}, {})
        
        # Simulate overfitting: train loss decreases, val loss increases
        for epoch in range(10):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),  # Decreasing
                val_loss=0.5 + 0.1 * epoch      # Increasing
            )
        
        # Check overfitting score
        overfitting_score = recorder._detect_overfitting()
        assert overfitting_score > 0  # Should detect overfitting
    
    def test_serialization_edge_cases(self, temp_dir):
        """Test serialization of edge cases."""
        recorder = TrainingRecorder("edge_case_test", {}, {})
        
        # Add epoch with None values
        recorder.log_epoch(
            epoch=0,
            train_loss=1.0,
            val_loss=None,
            gradient_norm=None,
            sample_predictions=None
        )
        
        # Save and load
        save_path = temp_dir / "edge_case.json"
        recorder.save(save_path)
        loaded = TrainingRecorder.load(save_path)
        
        # Verify None values are preserved
        assert loaded.epochs[0].val_loss is None
        assert loaded.epochs[0].gradient_norm is None