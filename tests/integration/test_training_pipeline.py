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
    
    def test_complete_training_workflow(self, mock_training_environment, 
                                       integration_temp_dir, mock_data_pipeline):
        """Test complete training workflow from data to model."""
        
        # 1. Create model
        model = TrafficLSTM.create_single_vd_model(
            input_size=3,
            hidden_size=64,
            num_layers=2
        )
        
        # 2. Create training configuration
        training_config = SingleVDTrainingConfig(
            epochs=3,  # Short for testing
            batch_size=16,
            learning_rate=0.001,
            save_dir=str(integration_temp_dir),
            experiment_name="integration_test"
        )
        
        # 3. Create recorder
        recorder = TrainingRecorder(
            experiment_name="integration_test",
            model_config=model.config.__dict__,
            training_config=training_config.__dict__
        )
        
        # 4. Mock trainer workflow
        with patch('social_xlstm.training.SingleVDTrainer') as MockTrainer:
            mock_trainer = MockTrainer.return_value
            mock_trainer.train.return_value = {'train_loss': [0.5], 'val_loss': [0.4]}
            mock_trainer.experiment_dir = integration_temp_dir
            
            # Simulate training
            trainer = MockTrainer(
                model=model,
                config=SingleVDTrainingConfig(
                    experiment_name="integration_test",
                    save_dir=str(integration_temp_dir),
                    epochs=1
                ),
                train_loader=mock_data_pipeline['train'],
                val_loader=mock_data_pipeline['val'],
                test_loader=mock_data_pipeline['test']
            )
            
            result = trainer.train()
            
            # Verify training was called
            mock_trainer.train.assert_called_once()
            assert result == {'train_loss': [0.5], 'val_loss': [0.4]}
    
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
    
    def test_visualization_integration(self, integration_temp_dir):
        """Test visualization integration with recorder."""
        from social_xlstm.visualization.training_visualizer import TrainingVisualizer
        
        # Create recorder with data
        recorder = TrainingRecorder(
            experiment_name="viz_test",
            model_config={'input_size': 3},
            training_config={'epochs': 10}
        )
        
        for epoch in range(10):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 * np.exp(-0.1 * epoch),
                val_loss=1.1 * np.exp(-0.08 * epoch),
                train_metrics={'mae': 0.5 * np.exp(-0.1 * epoch)},
                val_metrics={'mae': 0.55 * np.exp(-0.08 * epoch)},
                learning_rate=0.001 * (0.9 ** epoch),
                epoch_time=30.0
            )
        
        # Test visualization
        visualizer = TrainingVisualizer()
        
        # Test basic plots
        fig1 = visualizer.plot_basic_training_curves(recorder)
        assert fig1 is not None
        
        # Test dashboard
        fig2 = visualizer.plot_training_dashboard(recorder)
        assert fig2 is not None
        
        # Test report generation
        report_dir = integration_temp_dir / "report"
        visualizer.create_training_report(recorder, report_dir)
        
        # Verify report files
        assert report_dir.exists()
        assert (report_dir / "training_summary.txt").exists()
        assert (report_dir / "basic_training_curves.png").exists()
        assert (report_dir / "training_dashboard.png").exists()
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self, integration_temp_dir, mock_data_pipeline):
        """Test complete end-to-end workflow."""
        
        # 1. Create model
        model = TrafficLSTM.create_single_vd_model(input_size=3, hidden_size=32)
        
        # 2. Create training config
        training_config = SingleVDTrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=0.01,
            save_dir=str(integration_temp_dir),
            experiment_name="e2e_test"
        )
        
        # 3. Create recorder
        recorder = TrainingRecorder(
            experiment_name="e2e_test",
            model_config=model.config.__dict__,
            training_config=training_config.__dict__
        )
        
        # 4. Simulate training loop
        for epoch in range(2):
            # Mock training step
            train_loss = 1.0 - epoch * 0.1
            val_loss = 1.1 - epoch * 0.08
            
            recorder.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics={'mae': train_loss * 0.1},
                val_metrics={'mae': val_loss * 0.1},
                learning_rate=0.01,
                epoch_time=30.0
            )
        
        # 5. Save results
        results_dir = integration_temp_dir / "results"
        results_dir.mkdir()
        
        # Save model
        model_path = results_dir / "model.pt"
        from social_xlstm.models.lstm import save_model
        save_model(model, str(model_path))
        
        # Save training record
        record_path = results_dir / "training.json"
        recorder.save(record_path)
        
        # Generate visualization
        from social_xlstm.visualization.training_visualizer import TrainingVisualizer
        visualizer = TrainingVisualizer()
        viz_path = results_dir / "curves.png"
        visualizer.plot_basic_training_curves(recorder, save_path=viz_path)
        
        # 6. Verify all outputs
        assert model_path.exists()
        assert record_path.exists()
        assert viz_path.exists()
        
        # 7. Test loading and inference
        from social_xlstm.models.lstm import load_model
        loaded_model = load_model(str(model_path))
        loaded_recorder = TrainingRecorder.load(record_path)
        
        # Test inference
        test_input = torch.randn(1, 12, 3)
        output = loaded_model(test_input)
        assert output.shape == (1, 1, 3)
        
        # Verify training history
        assert len(loaded_recorder.epochs) == 2
        assert loaded_recorder.experiment_name == "e2e_test"