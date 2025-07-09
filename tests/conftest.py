"""
Global pytest configuration for Social-xLSTM project.

This file contains shared fixtures and configuration that can be used
across all test modules in the project.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
import warnings

# Use non-interactive backend for testing
matplotlib.use('Agg')

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the entire test session."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_config():
    """Sample configuration for testing."""
    return {
        'model_config': {
            'input_size': 3,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_first': True
        },
        'training_config': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'device': 'cpu'  # Use CPU for testing
        }
    }


@pytest.fixture
def sample_data():
    """Generate sample traffic data for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    batch_size = 16
    seq_length = 12
    num_features = 3
    
    # Generate synthetic traffic data
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
def device():
    """Get the appropriate device for testing."""
    return torch.device('cpu')  # Always use CPU for testing


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Ensure we're in deterministic mode
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    yield
    
    # Clean up after test
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib for tests that generate plots."""
    import matplotlib.pyplot as plt
    
    # Store original functions
    original_show = plt.show
    original_savefig = plt.savefig
    
    # Mock the functions
    plt.show = lambda: None
    plt.savefig = lambda *args, **kwargs: None
    
    yield
    
    # Restore original functions
    plt.show = original_show
    plt.savefig = original_savefig


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker for tests that might be slow
        if "test_training" in str(item.fspath) or "test_visualization" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="function")
def cleanup_files():
    """Clean up test files after each test."""
    created_files = []
    
    def add_file(filepath):
        created_files.append(Path(filepath))
    
    yield add_file
    
    # Clean up created files
    for file_path in created_files:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)