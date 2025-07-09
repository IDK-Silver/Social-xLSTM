"""
Integration test specific configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch, MagicMock


@pytest.fixture
def integration_temp_dir():
    """Create a temporary directory for integration tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_training_environment():
    """Mock training environment for integration tests."""
    with patch("matplotlib.pyplot.show") as mock_show, \
         patch("matplotlib.pyplot.savefig") as mock_savefig, \
         patch("torch.save") as mock_save, \
         patch("torch.load") as mock_load:
        
        yield {
            'show': mock_show,
            'savefig': mock_savefig,
            'save': mock_save,
            'load': mock_load
        }


@pytest.fixture
def sample_hdf5_data(integration_temp_dir):
    """Create sample HDF5 data for integration tests."""
    import h5py
    
    hdf5_path = integration_temp_dir / "sample_data.h5"
    
    with h5py.File(hdf5_path, 'w') as f:
        # Create sample VD data
        vd_group = f.create_group('VD001')
        
        # Generate sample time series data
        num_samples = 1000
        features = ['volume', 'speed', 'occupancy']
        
        for feature in features:
            data = np.random.randn(num_samples)
            vd_group.create_dataset(feature, data=data)
        
        # Add metadata
        vd_group.attrs['location'] = 'Test Location'
        vd_group.attrs['coordinates'] = [120.0, 24.0]
    
    return hdf5_path


@pytest.fixture
def mock_data_pipeline():
    """Mock data pipeline components."""
    mock_dataloaders = {
        'train': MagicMock(),
        'val': MagicMock(),
        'test': MagicMock()
    }
    
    # Configure mock dataloaders
    for loader in mock_dataloaders.values():
        loader.__iter__ = MagicMock(return_value=iter([
            {
                'input_seq': torch.randn(16, 12, 3),
                'target_seq': torch.randn(16, 1, 3)
            }
        ]))
        loader.__len__ = MagicMock(return_value=10)
    
    return mock_dataloaders