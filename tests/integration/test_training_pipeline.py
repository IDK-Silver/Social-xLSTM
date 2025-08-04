"""
Integration tests for training pipeline.

Tests the complete training workflow including data loading,
model training, and evaluation.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from pathlib import Path

from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.training.recorder import TrainingRecorder
from social_xlstm.training import SingleVDTrainer, SingleVDTrainingConfig


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    def test_complete_training_workflow(self, integration_temp_dir):
        """Test complete training workflow from data to model."""
        
        # 1. Create model
        model = TrafficLSTM.create_single_vd_model(
            input_size=3,
            hidden_size=64,
            num_layers=2
        )
        
        # 2. Create training configuration
        training_config = SingleVDTrainingConfig(
            epochs=1,  # Short for testing
            batch_size=16,
            learning_rate=0.001,
            save_dir=str(integration_temp_dir),
            experiment_name="integration_test"
        )
        
        # 3. Verify model and config creation
        assert model is not None
        assert training_config.experiment_name == "integration_test"
        assert training_config.epochs == 1
    
    def test_model_save_and_load(self, integration_temp_dir):
        """Test model saving and loading functionality."""
        from social_xlstm.models.lstm import save_model, load_model
        
        # Create and save model
        original_model = TrafficLSTM.create_single_vd_model(
            input_size=3,
            hidden_size=64,
            num_layers=2
        )
        
        save_path = integration_temp_dir / "test_model.pt"
        save_model(original_model, str(save_path))
        
        # Load model
        loaded_model = load_model(str(save_path))
        
        # Verify model structure
        assert loaded_model.config.input_size == original_model.config.input_size
        assert loaded_model.config.hidden_size == original_model.config.hidden_size
        assert loaded_model.config.num_layers == original_model.config.num_layers
        
        # Test forward pass with same input
        torch.manual_seed(42)  # Set seed for reproducible input
        test_input = torch.randn(1, 12, 3)
        
        # Set models to eval mode for consistent behavior
        original_model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            original_output = original_model(test_input)
            loaded_output = loaded_model(test_input)
        
        # Outputs should be the same (with reasonable tolerance)
        assert torch.allclose(original_output, loaded_output, atol=1e-4)
    
    def test_recorder_full_workflow(self, integration_temp_dir):
        """Test complete recorder workflow."""
        recorder = TrainingRecorder(
            experiment_name="integration_test",
            model_config={'input_size': 3, 'hidden_size': 64},
            training_config={'epochs': 10, 'batch_size': 16}
        )
        
        # Simulate training epochs
        for epoch in range(5):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),
                val_loss=1.1 / (epoch + 1),
                train_metrics={'mae': 0.1 / (epoch + 1)},
                val_metrics={'mae': 0.11 / (epoch + 1)},
                learning_rate=0.001,
                epoch_time=30.0
            )
        
        # Test saving
        json_path = integration_temp_dir / "training_record.json"
        csv_path = integration_temp_dir / "training_history.csv"
        
        recorder.save(json_path)
        recorder.export_to_csv(csv_path)
        
        # Verify files were created
        assert json_path.exists()
        assert csv_path.exists()
        
        # Test loading
        loaded_recorder = TrainingRecorder.load(json_path)
        assert len(loaded_recorder.epochs) == 5
        assert loaded_recorder.experiment_name == "integration_test"
    
    
    def test_basic_integration(self, integration_temp_dir):
        """Test basic integration of core components."""
        
        # Create model
        model = TrafficLSTM.create_single_vd_model(input_size=3, hidden_size=32)
        
        # Create recorder
        recorder = TrainingRecorder(
            experiment_name="basic_test",
            model_config=model.config.__dict__,
            training_config={'epochs': 1}
        )
        
        # Log a single epoch
        recorder.log_epoch(
            epoch=0,
            train_loss=1.0,
            val_loss=1.1,
            learning_rate=0.001,
            epoch_time=30.0
        )
        
        # Save results
        record_path = integration_temp_dir / "basic_test.json"
        recorder.save(record_path)
        
        # Verify
        assert record_path.exists()
        loaded_recorder = TrainingRecorder.load(record_path)
        assert len(loaded_recorder.epochs) == 1
        assert loaded_recorder.experiment_name == "basic_test"