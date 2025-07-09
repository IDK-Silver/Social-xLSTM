"""Tests for dataset configuration classes."""

import pytest
import tempfile
from pathlib import Path
from social_xlstm.dataset.config import TrafficDatasetConfig, TrafficHDF5Config


class TestTrafficDatasetConfig:
    """Test TrafficDatasetConfig class."""
    
    def test_init_with_valid_path(self):
        """Test initialization with valid HDF5 path."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            config = TrafficDatasetConfig(hdf5_path=tmp_path)
            assert config.hdf5_path == tmp_path
            assert config.sequence_length == 60
            assert config.prediction_length == 15
            assert config.train_ratio == 0.7
            assert config.val_ratio == 0.15
            assert config.test_ratio == 0.15
        finally:
            tmp_path.unlink()
    
    def test_init_with_invalid_path(self):
        """Test initialization with invalid HDF5 path."""
        invalid_path = Path("/nonexistent/path/file.h5")
        
        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            TrafficDatasetConfig(hdf5_path=invalid_path)
    
    def test_invalid_ratios(self):
        """Test validation of train/val/test ratios."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            with pytest.raises(ValueError, match="Train/val/test ratios must sum to 1.0"):
                TrafficDatasetConfig(
                    hdf5_path=tmp_path,
                    train_ratio=0.5,
                    val_ratio=0.3,
                    test_ratio=0.3  # Sum = 1.1
                )
        finally:
            tmp_path.unlink()
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                sequence_length=30,
                prediction_length=10,
                selected_vdids=['VD001', 'VD002'],
                selected_features=['speed', 'volume'],
                normalize=False,
                batch_size=64
            )
            
            assert config.sequence_length == 30
            assert config.prediction_length == 10
            assert config.selected_vdids == ['VD001', 'VD002']
            assert config.selected_features == ['speed', 'volume']
            assert config.normalize is False
            assert config.batch_size == 64
        finally:
            tmp_path.unlink()


class TestTrafficHDF5Config:
    """Test TrafficHDF5Config class."""
    
    def test_init_with_valid_paths(self):
        """Test initialization with valid paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir)
            output_path = Path(tmp_dir) / "output.h5"
            
            config = TrafficHDF5Config(
                source_dir=source_dir,
                output_path=output_path
            )
            
            assert config.source_dir == source_dir
            assert config.output_path == output_path
            assert config.feature_names == [
                'avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count'
            ]
            assert config.compression == 'gzip'
            assert config.compression_opts == 6
            assert config.overwrite is False
            assert config.check_consistency is True
    
    def test_init_with_invalid_source_dir(self):
        """Test initialization with invalid source directory."""
        invalid_dir = Path("/nonexistent/directory")
        output_path = Path("/tmp/output.h5")
        
        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            TrafficHDF5Config(
                source_dir=invalid_dir,
                output_path=output_path
            )
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir)
            output_path = Path(tmp_dir) / "output.h5"
            
            config = TrafficHDF5Config(
                source_dir=source_dir,
                output_path=output_path,
                selected_vdids=['VD001', 'VD002'],
                feature_names=['speed', 'volume'],
                compression='lzf',
                compression_opts=None,
                overwrite=True,
                check_consistency=False
            )
            
            assert config.selected_vdids == ['VD001', 'VD002']
            assert config.feature_names == ['speed', 'volume']
            assert config.compression == 'lzf'
            assert config.compression_opts is None
            assert config.overwrite is True
            assert config.check_consistency is False
    
    def test_path_conversion(self):
        """Test that paths are converted to Path objects."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir_str = str(tmp_dir)
            output_path_str = str(Path(tmp_dir) / "output.h5")
            
            config = TrafficHDF5Config(
                source_dir=source_dir_str,
                output_path=output_path_str
            )
            
            assert isinstance(config.source_dir, Path)
            assert isinstance(config.output_path, Path)
            assert config.source_dir == Path(source_dir_str)
            assert config.output_path == Path(output_path_str)