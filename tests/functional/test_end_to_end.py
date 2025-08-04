"""
Functional tests for end-to-end workflows.

These tests verify that the complete system works together
as expected from a user's perspective.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.training.recorder import TrainingRecorder
from social_xlstm.visualization.training_visualizer import TrainingVisualizer


@pytest.mark.functional
class TestEndToEndWorkflows:
    """Functional tests for complete user workflows."""
    
    @pytest.fixture
    def functional_temp_dir(self):
        """Create temporary directory for functional tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_baseline_experiment_workflow(self, functional_temp_dir):
        """Test complete baseline experiment workflow."""
        
        # Step 1: Create and train single VD model
        print("Step 1: Creating single VD model...")
        single_vd_model = TrafficLSTM.create_single_vd_model(
            input_size=3,
            hidden_size=64,
            num_layers=2
        )
        
        # Step 2: Create recorder for single VD
        single_vd_recorder = TrainingRecorder(
            experiment_name="single_vd_baseline",
            model_config=single_vd_model.config.__dict__,
            training_config={'epochs': 10, 'batch_size': 32}
        )
        
        # Step 3: Simulate training
        print("Step 3: Simulating training...")
        for epoch in range(10):
            single_vd_recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 * np.exp(-0.1 * epoch) + 0.01 * np.random.rand(),
                val_loss=1.1 * np.exp(-0.08 * epoch) + 0.02 * np.random.rand(),
                train_metrics={
                    'mae': 0.5 * np.exp(-0.1 * epoch),
                    'mse': 0.25 * np.exp(-0.1 * epoch),
                    'rmse': 0.5 * np.exp(-0.1 * epoch)
                },
                val_metrics={
                    'mae': 0.55 * np.exp(-0.08 * epoch),
                    'mse': 0.3 * np.exp(-0.08 * epoch),
                    'rmse': 0.55 * np.exp(-0.08 * epoch)
                },
                learning_rate=0.001 * (0.9 ** epoch),
                epoch_time=30.0 + 5 * np.random.rand()
            )
        
        # Step 4: Save experiment results
        experiment_dir = functional_temp_dir / "baseline_experiments"
        experiment_dir.mkdir()
        
        single_vd_recorder.save(experiment_dir / "training_record.json")
        single_vd_recorder.export_to_csv(experiment_dir / "training_history.csv")
        
        # Step 5: Verify outputs exist
        assert (experiment_dir / "training_record.json").exists()
        assert (experiment_dir / "training_history.csv").exists()
        
        # Step 6: Test loading
        loaded_recorder = TrainingRecorder.load(experiment_dir / "training_record.json")
        assert len(loaded_recorder.epochs) == 10
        assert loaded_recorder.experiment_name == "single_vd_baseline"
        
        print("PASS: Baseline experiment workflow completed successfully!")
    
    def test_new_metric_calculation_workflow(self, functional_temp_dir):
        """Test workflow for calculating new metrics post-training."""
        
        # Step 1: Load existing experiment
        recorder = TrainingRecorder(
            experiment_name="metric_test",
            model_config={'input_size': 3},
            training_config={'epochs': 5}
        )
        
        # Add some training data
        for epoch in range(5):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),
                val_loss=1.1 / (epoch + 1),
                train_metrics={'mae': 0.1 / (epoch + 1), 'mse': 0.01 / (epoch + 1)},
                val_metrics={'mae': 0.11 / (epoch + 1), 'mse': 0.011 / (epoch + 1)},
                learning_rate=0.001,
                epoch_time=30.0
            )
        
        # Save experiment
        save_path = functional_temp_dir / "experiment.json"
        recorder.save(save_path)
        
        # Step 2: Load experiment later
        loaded_recorder = TrainingRecorder.load(save_path)
        
        # Step 3: Calculate new custom metrics
        def custom_metric_1(epoch_record):
            """Custom metric: ratio of val_loss to train_loss"""
            if epoch_record.val_loss and epoch_record.train_loss:
                return epoch_record.val_loss / epoch_record.train_loss
            return None
        
        def custom_metric_2(epoch_record):
            """Custom metric: improvement rate"""
            if epoch_record.epoch == 0:
                return 0.0
            return (1.0 - epoch_record.train_loss) * 100  # Improvement percentage
        
        # Calculate custom metrics for all epochs
        custom_metrics_1 = []
        custom_metrics_2 = []
        
        for epoch_record in loaded_recorder.epochs:
            custom_metrics_1.append(custom_metric_1(epoch_record))
            custom_metrics_2.append(custom_metric_2(epoch_record))
        
        # Step 4: Verify custom metrics
        assert len(custom_metrics_1) == 5
        assert len(custom_metrics_2) == 5
        
        # All ratios should be > 1 (val_loss > train_loss)
        assert all(ratio > 1.0 for ratio in custom_metrics_1 if ratio is not None)
        
        # Improvement should increase over time
        assert custom_metrics_2[-1] > custom_metrics_2[0]
        
        # Step 5: Verify custom metrics calculation
        assert len(custom_metrics_1) == 5
        assert len(custom_metrics_2) == 5
        
        print("PASS: New metric calculation workflow completed successfully!")
    
    def test_basic_experiment_workflow(self, functional_temp_dir):
        """Test basic experiment workflow."""
        
        # Create single experiment
        recorder = TrainingRecorder(
            experiment_name='basic_model',
            model_config={'hidden_size': 64, 'num_layers': 2},
            training_config={'epochs': 3, 'batch_size': 32}
        )
        
        # Simulate training
        for epoch in range(3):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 * np.exp(-0.1 * epoch),
                val_loss=1.1 * np.exp(-0.08 * epoch),
                learning_rate=0.001,
                epoch_time=30.0
            )
        
        # Save experiment
        experiment_dir = functional_temp_dir / "basic_experiment"
        experiment_dir.mkdir()
        recorder.save(experiment_dir / "experiment.json")
        
        # Load and verify
        loaded_recorder = TrainingRecorder.load(experiment_dir / "experiment.json")
        assert len(loaded_recorder.epochs) == 3
        assert loaded_recorder.experiment_name == 'basic_model'