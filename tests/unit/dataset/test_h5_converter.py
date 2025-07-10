"""Unit tests for HDF5 converter."""

import pytest
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from social_xlstm.dataset.config import TrafficHDF5Config
from social_xlstm.dataset.storage.h5_converter import TrafficHDF5Converter, TrafficFeatureExtractor
from social_xlstm.dataset.utils.json_utils import VDInfo, VDLiveList, VDLiveDetail, LinkFlowData, LaneData, VehicleData


class TestTrafficFeatureExtractor:
    """Test TrafficFeatureExtractor class."""
    
    def test_extract_lane_features(self):
        """Test lane feature extraction."""
        # Create mock lane data
        lane = LaneData(
            LaneID=1,
            LaneType=1,
            Speed=60.0,
            Occupancy=10.5,
            Vehicles=[
                VehicleData(VehicleType="S", Volume=5),
                VehicleData(VehicleType="L", Volume=2)
            ]
        )
        
        features = TrafficFeatureExtractor.extract_lane_features(lane)
        
        assert features['speed'] == 60.0
        assert features['occupancy'] == 10.5
        assert features['volume'] == 7.0  # 5 + 2
    
    def test_extract_lane_features_with_nan(self):
        """Test lane feature extraction with NaN values."""
        lane = LaneData(
            LaneID=1,
            LaneType=1,
            Speed=float('nan'),
            Occupancy=float('nan'),
            Vehicles=[]
        )
        
        features = TrafficFeatureExtractor.extract_lane_features(lane)
        
        assert np.isnan(features['speed'])
        assert np.isnan(features['occupancy'])
        assert features['volume'] == 0.0
    
    def test_aggregate_vd_features(self):
        """Test VD feature aggregation."""
        # Create mock VD detail with multiple lanes
        vd_detail = VDLiveDetail(
            VDID="VD-11-0020-002-002",
            DataCollectTime="2025-03-18T00:46:11+08:00",
            UpdateInterval=60,
            AuthorityCode="THB",
            LinkFlows=[
                LinkFlowData(
                    LinkID="3000200000176F",
                    Lanes=[
                        LaneData(
                            LaneID=0,
                            LaneType=1,
                            Speed=60.0,
                            Occupancy=5.0,
                            Vehicles=[VehicleData(VehicleType="S", Volume=10)]
                        ),
                        LaneData(
                            LaneID=1,
                            LaneType=1,
                            Speed=50.0,
                            Occupancy=10.0,
                            Vehicles=[VehicleData(VehicleType="S", Volume=20)]
                        )
                    ]
                )
            ]
        )
        
        feature_names = ['avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count']
        features = TrafficFeatureExtractor.aggregate_vd_features(vd_detail, feature_names)
        
        assert features[0] == 55.0  # avg_speed: (60 + 50) / 2
        assert features[1] == 30.0  # total_volume: 10 + 20
        assert features[2] == 7.5   # avg_occupancy: (5 + 10) / 2
        assert features[3] == 5.0   # speed_std: std([60, 50])
        assert features[4] == 2.0   # lane_count
    
    def test_aggregate_vd_features_empty_lanes(self):
        """Test VD feature aggregation with empty lanes."""
        vd_detail = VDLiveDetail(
            VDID="VD-11-0020-002-002",
            DataCollectTime="2025-03-18T00:46:11+08:00",
            UpdateInterval=60,
            AuthorityCode="THB",
            LinkFlows=[]
        )
        
        feature_names = ['avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count']
        features = TrafficFeatureExtractor.aggregate_vd_features(vd_detail, feature_names)
        
        # All features should be NaN when no lanes
        assert all(np.isnan(f) for f in features)


class TestTrafficHDF5Converter:
    """Test TrafficHDF5Converter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        source_dir = temp_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)  # Create before passing to config
        
        config = TrafficHDF5Config(
            source_dir=source_dir,
            output_path=temp_dir / "output.h5",
            selected_vdids=None,
            time_range=None,
            show_progress=False,  # Disable progress bar in tests
            verbose_warnings=False
        )
        return config
    
    def create_test_data_dir(self, source_dir: Path, timestamp: str, num_vds: int = 5):
        """Create a test data directory with VD data."""
        dir_path = source_dir / timestamp
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create VDList.json
        vd_list_data = {
            "UpdateInfo": {
                "UpdateTime": "2025-03-18T00:00:00+08:00",
                "UpdateInterval": 60
            },
            "VDList": [
                {
                    "VDID": f"VD-11-0020-00{i}-01",
                    "PositionLon": 120.0 + i * 0.01,
                    "PositionLat": 24.0 + i * 0.01,
                    "RoadID": f"R{i}",
                    "RoadName": f"Road {i}",
                    "LaneNum": 2,
                    "DetectionLinks": []
                }
                for i in range(num_vds)
            ]
        }
        
        with open(dir_path / "VDList.json", 'w') as f:
            json.dump(vd_list_data, f)
        
        # Create VDLiveList.json
        vd_live_data = []
        for i in range(num_vds):
            vd_live_data.append({
                "VDID": f"VD-11-0020-00{i}-001",  # Note: 001 vs 01 ending
                "DataCollectTime": f"2025-03-18T00:00:00+08:00",
                "UpdateInterval": 60,
                "AuthorityCode": "THB",
                "LinkFlows": [{
                    "LinkID": f"LINK{i}",
                    "Lanes": [{
                        "LaneID": 0,
                        "LaneType": 1,
                        "Speed": 50.0 + i * 5,
                        "Occupancy": 5.0 + i,
                        "Vehicles": [{"VehicleType": "S", "Volume": 10 + i}]
                    }]
                }]
            })
        
        with open(dir_path / "VDLiveList.json", 'w') as f:
            json.dump(vd_live_data, f)
    
    def test_init(self, mock_config):
        """Test converter initialization."""
        converter = TrafficHDF5Converter(mock_config)
        
        assert converter.config == mock_config
        assert converter.show_progress is False
        assert converter.verbose_warnings is False
        assert converter.max_missing_report == 10
    
    def test_parse_timestamp(self, mock_config):
        """Test timestamp parsing."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Valid timestamp
        ts = converter._parse_timestamp("2025-03-18_00-49-00")
        assert ts == datetime(2025, 3, 18, 0, 49, 0)
        
        # Invalid timestamp
        ts = converter._parse_timestamp("invalid")
        assert ts is None
    
    def test_get_sorted_time_directories(self, mock_config):
        """Test getting sorted time directories."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test directories
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00")
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_01-00-00")
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_02-00-00")
        
        # Create invalid directory (should be skipped)
        (mock_config.source_dir / "invalid").mkdir()
        
        time_dirs = converter._get_sorted_time_directories()
        
        assert len(time_dirs) == 3
        assert time_dirs[0][0] == datetime(2025, 3, 18, 0, 0, 0)
        assert time_dirs[1][0] == datetime(2025, 3, 18, 1, 0, 0)
        assert time_dirs[2][0] == datetime(2025, 3, 18, 2, 0, 0)
    
    def test_time_range_filter(self, mock_config):
        """Test time range filtering."""
        mock_config.time_range = ("2025-03-18_00-30-00", "2025-03-18_01-30-00")
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test directories
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00")  # Before range
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_01-00-00")  # In range
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_02-00-00")  # After range
        
        time_dirs = converter._get_sorted_time_directories()
        
        assert len(time_dirs) == 1
        assert time_dirs[0][0] == datetime(2025, 3, 18, 1, 0, 0)
    
    def test_load_vd_info(self, mock_config):
        """Test loading VD information."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test data
        test_dir = mock_config.source_dir / "2025-03-18_00-00-00"
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=3)
        
        vd_info = converter._load_vd_info(test_dir)
        
        assert len(vd_info) == 3
        assert "VD-11-0020-000-001" in vd_info  # Padded with zeros
        assert vd_info["VD-11-0020-000-001"]["position_lon"] == 120.0
        assert vd_info["VD-11-0020-000-001"]["road_name"] == "Road 0"
    
    def test_get_target_vdids(self, mock_config):
        """Test getting target VD IDs."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test data
        test_dir = mock_config.source_dir / "2025-03-18_00-00-00"
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=3)
        
        # Test with no selected VDIDs (should return all)
        vdids = converter._get_target_vdids(test_dir)
        assert len(vdids) == 3
        assert "VD-11-0020-000-001" in vdids
        
        # Test with selected VDIDs
        mock_config.selected_vdids = ["VD-11-0020-000-001", "VD-11-0020-002-001"]
        converter = TrafficHDF5Converter(mock_config)
        vdids = converter._get_target_vdids(test_dir)
        assert len(vdids) == 2
        assert vdids == ["VD-11-0020-000-001", "VD-11-0020-002-001"]
    
    def test_process_timestep(self, mock_config):
        """Test processing a single timestep."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test data
        test_dir = mock_config.source_dir / "2025-03-18_00-00-00"
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=3)
        
        # Load VD info first
        converter._load_vd_info(test_dir)
        target_vdids = ["VD-11-0020-000-001", "VD-11-0020-001-001", "VD-11-0020-002-001"]
        
        timestamp, features = converter._process_timestep(test_dir, target_vdids)
        
        assert timestamp == "2025-03-18T00:00:00+08:00"
        assert features.shape == (3, 5)  # 3 VDs, 5 features
        
        # Check first VD features
        assert features[0, 0] == 50.0  # avg_speed
        assert features[0, 1] == 10.0  # total_volume
        assert features[0, 2] == 5.0   # avg_occupancy
        assert np.isnan(features[0, 3])  # speed_std (only 1 lane)
        assert features[0, 4] == 1.0   # lane_count
    
    def test_create_hdf5_structure(self, mock_config):
        """Test HDF5 structure creation."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test data
        test_dir = mock_config.source_dir / "2025-03-18_00-00-00"
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=2)
        
        vd_info = converter._load_vd_info(test_dir)
        target_vdids = ["VD-11-0020-000-001", "VD-11-0020-001-001"]
        
        with h5py.File(mock_config.output_path, 'w') as h5file:
            ts_ds, feat_ds = converter._create_hdf5_structure(
                h5file, target_vdids, vd_info, num_timesteps=10
            )
            
            # Check structure
            assert 'metadata' in h5file
            assert 'data' in h5file
            assert 'metadata/vdids' in h5file
            assert 'metadata/feature_names' in h5file
            assert 'metadata/timestamps' in h5file
            assert 'data/features' in h5file
            
            # Check dimensions
            assert feat_ds.shape == (10, 2, 5)  # 10 timesteps, 2 VDs, 5 features
            
            # Check metadata
            assert h5file.attrs['num_timesteps'] == 10
            assert h5file.attrs['num_locations'] == 2
            assert h5file.attrs['num_features'] == 5
    
    def test_convert_basic(self, mock_config):
        """Test basic conversion process."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test data for 2 timesteps
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=2)
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_01-00-00", num_vds=2)
        
        # Run conversion
        converter.convert()
        
        # Check output file exists
        assert mock_config.output_path.exists()
        
        # Verify HDF5 content
        with h5py.File(mock_config.output_path, 'r') as h5file:
            # Check data shape
            features = h5file['data/features'][:]
            assert features.shape == (2, 2, 5)  # 2 timesteps, 2 VDs, 5 features
            
            # Check timestamps
            timestamps = [s.decode('utf-8') for s in h5file['metadata/timestamps'][:]]
            assert len(timestamps) == 2
            
            # Check VD IDs
            vdids = [s.decode('utf-8') for s in h5file['metadata/vdids'][:]]
            assert len(vdids) == 2
            assert "VD-11-0020-000-001" in vdids
    
    def test_convert_with_missing_vds(self, mock_config):
        """Test conversion with missing VD data."""
        mock_config.verbose_warnings = True  # Enable to test warning behavior
        converter = TrafficHDF5Converter(mock_config)
        
        # Create first timestep with 3 VDs
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=3)
        
        # Create second timestep with only 2 VDs (missing one)
        dir_path = mock_config.source_dir / "2025-03-18_01-00-00"
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # VDList still has 3 VDs
        vd_list_data = {
            "UpdateInfo": {"UpdateTime": "2025-03-18T01:00:00+08:00", "UpdateInterval": 60},
            "VDList": [
                {
                    "VDID": f"VD-11-0020-00{i}-01",
                    "PositionLon": 120.0 + i * 0.01,
                    "PositionLat": 24.0 + i * 0.01,
                    "RoadID": f"R{i}",
                    "RoadName": f"Road {i}",
                    "LaneNum": 2,
                    "DetectionLinks": []
                }
                for i in range(3)
            ]
        }
        
        with open(dir_path / "VDList.json", 'w') as f:
            json.dump(vd_list_data, f)
        
        # VDLiveList only has 2 VDs (missing VD-11-0020-002-001)
        vd_live_data = []
        for i in [0, 1]:  # Skip i=2
            vd_live_data.append({
                "VDID": f"VD-11-0020-00{i}-001",
                "DataCollectTime": f"2025-03-18T01:00:00+08:00",
                "UpdateInterval": 60,
                "AuthorityCode": "THB",
                "LinkFlows": [{
                    "LinkID": f"LINK{i}",
                    "Lanes": [{
                        "LaneID": 0,
                        "LaneType": 1,
                        "Speed": 50.0 + i * 5,
                        "Occupancy": 5.0 + i,
                        "Vehicles": [{"VehicleType": "S", "Volume": 10 + i}]
                    }]
                }]
            })
        
        with open(dir_path / "VDLiveList.json", 'w') as f:
            json.dump(vd_live_data, f)
        
        # Run conversion
        converter.convert()
        
        # Check output
        with h5py.File(mock_config.output_path, 'r') as h5file:
            features = h5file['data/features'][:]
            
            # Third VD in second timestep should have NaN values
            assert not np.isnan(features[1, 0, 0])  # First VD has data
            assert not np.isnan(features[1, 1, 0])  # Second VD has data
            assert np.isnan(features[1, 2, 0])      # Third VD is missing
    
    def test_overwrite_behavior(self, mock_config):
        """Test overwrite behavior."""
        converter = TrafficHDF5Converter(mock_config)
        
        # Create test data
        self.create_test_data_dir(mock_config.source_dir, "2025-03-18_00-00-00", num_vds=1)
        
        # First conversion
        converter.convert()
        assert mock_config.output_path.exists()
        
        # Try to convert again without overwrite (should skip)
        mock_config.overwrite = False
        converter = TrafficHDF5Converter(mock_config)
        
        # Mock the file modification time check to avoid update
        with patch.object(converter, '_is_hdf5_outdated', return_value=False):
            converter.convert()  # Should print message and return early
        
        # Now with overwrite
        mock_config.overwrite = True
        converter = TrafficHDF5Converter(mock_config)
        converter.convert()  # Should overwrite


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete conversion process."""
    
    def test_end_to_end_conversion(self, tmp_path):
        """Test complete end-to-end conversion with real-like data."""
        # Setup
        source_dir = tmp_path / "source"
        output_path = tmp_path / "output.h5"
        
        # Create multiple timesteps with varying data
        timestamps = [
            "2025-03-18_00-00-00",
            "2025-03-18_00-30-00",
            "2025-03-18_01-00-00",
            "2025-03-18_01-30-00",
            "2025-03-18_02-00-00"
        ]
        
        for ts in timestamps:
            dir_path = source_dir / ts
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create realistic VD data
            vd_list = {
                "UpdateInfo": {"UpdateTime": ts, "UpdateInterval": 60},
                "VDList": [
                    {
                        "VDID": f"VD-11-0020-00{i}-01",
                        "PositionLon": 120.0 + i * 0.01,
                        "PositionLat": 24.0 + i * 0.01,
                        "RoadID": f"Highway-{i}",
                        "RoadName": f"National Highway {i}",
                        "LaneNum": 3,
                        "DetectionLinks": []
                    }
                    for i in range(10)  # 10 VDs
                ]
            }
            
            with open(dir_path / "VDList.json", 'w') as f:
                json.dump(vd_list, f)
            
            # Create live data with some randomness
            vd_live = []
            for i in range(10):
                # Simulate some missing data
                if np.random.random() > 0.9:  # 10% chance of missing
                    continue
                    
                vd_live.append({
                    "VDID": f"VD-11-0020-00{i}-001",
                    "DataCollectTime": ts.replace('_', 'T').replace('-', ':') + "+08:00",
                    "UpdateInterval": 60,
                    "AuthorityCode": "THB",
                    "LinkFlows": [{
                        "LinkID": f"LINK{i}",
                        "Lanes": [
                            {
                                "LaneID": j,
                                "LaneType": 1,
                                "Speed": np.random.uniform(40, 80),
                                "Occupancy": np.random.uniform(0, 30),
                                "Vehicles": [
                                    {"VehicleType": "S", "Volume": np.random.randint(0, 50)},
                                    {"VehicleType": "L", "Volume": np.random.randint(0, 10)}
                                ]
                            }
                            for j in range(3)  # 3 lanes
                        ]
                    }]
                })
            
            with open(dir_path / "VDLiveList.json", 'w') as f:
                json.dump(vd_live, f)
        
        # Configure and run conversion
        config = TrafficHDF5Config(
            source_dir=source_dir,
            output_path=output_path,
            time_range=("2025-03-18_00-00-00", "2025-03-18_02-00-00"),
            show_progress=False,
            verbose_warnings=False
        )
        
        converter = TrafficHDF5Converter(config)
        converter.convert()
        
        # Verify results
        assert output_path.exists()
        
        with h5py.File(output_path, 'r') as h5file:
            # Check structure
            assert 'data/features' in h5file
            assert 'metadata/vdids' in h5file
            assert 'metadata/timestamps' in h5file
            
            # Check data
            features = h5file['data/features'][:]
            assert features.shape[0] == 5  # 5 timesteps
            assert features.shape[1] == 10  # 10 VDs
            assert features.shape[2] == 5   # 5 features
            
            # Check that we have some real data (not all NaN)
            assert not np.all(np.isnan(features))
            
            # Check feature ranges
            speeds = features[:, :, 0]  # avg_speed
            valid_speeds = speeds[~np.isnan(speeds)]
            if len(valid_speeds) > 0:
                assert np.all(valid_speeds >= 0)
                assert np.all(valid_speeds <= 100)  # Reasonable speed range