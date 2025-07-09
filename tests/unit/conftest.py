"""
Unit test specific configuration and fixtures.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from social_xlstm.training.recorder import TrainingRecorder
from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig


@pytest.fixture
def unit_temp_dir():
    """Create a temporary directory for unit tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_lstm_config():
    """Sample LSTM configuration for unit tests."""
    return TrafficLSTMConfig(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        batch_first=True,
        device='cpu'
    )


@pytest.fixture
def sample_lstm_model(sample_lstm_config):
    """Create a sample LSTM model for testing."""
    return TrafficLSTM(sample_lstm_config)


@pytest.fixture
def sample_training_data():
    """Generate sample training data."""
    batch_size = 8
    seq_length = 12
    num_features = 3
    
    input_data = torch.randn(batch_size, seq_length, num_features)
    target_data = torch.randn(batch_size, 1, num_features)
    
    return {
        'input': input_data,
        'target': target_data,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'num_features': num_features
    }


@pytest.fixture
def sample_recorder():
    """Create a sample training recorder."""
    return TrainingRecorder(
        experiment_name="unit_test",
        model_config={'input_size': 3, 'hidden_size': 64},
        training_config={'epochs': 10, 'batch_size': 16}
    )


@pytest.fixture
def recorder_with_epochs(sample_recorder):
    """Create a recorder with sample epoch data."""
    for epoch in range(5):
        sample_recorder.log_epoch(
            epoch=epoch,
            train_loss=1.0 / (epoch + 1),
            val_loss=1.1 / (epoch + 1),
            train_metrics={'mae': 0.1 / (epoch + 1), 'mse': 0.01 / (epoch + 1)},
            val_metrics={'mae': 0.11 / (epoch + 1), 'mse': 0.011 / (epoch + 1)},
            learning_rate=0.001 * (0.9 ** epoch),
            epoch_time=30.0 + epoch * 2.0
        )
    return sample_recorder