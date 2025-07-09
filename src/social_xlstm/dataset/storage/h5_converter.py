"""HDF5 converter for traffic data."""

import h5py
import numpy as np
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..config import TrafficHDF5Config
from .h5_reader import TrafficHDF5Reader
from ..utils.json_utils import VDInfo, VDLiveList


class TrafficFeatureExtractor:
    """Extracts features from traffic data structures."""
    
    @staticmethod
    def extract_lane_features(lane) -> Dict[str, float]:
        """Extract features from a single lane."""
        from ..utils.json_utils import LaneData
        
        total_volume = sum(vehicle.Volume for vehicle in lane.Vehicles) if lane.Vehicles else 0
        
        return {
            'speed': float(lane.Speed) if not np.isnan(lane.Speed) else np.nan,
            'occupancy': float(lane.Occupancy) if not np.isnan(lane.Occupancy) else np.nan,
            'volume': float(total_volume)
        }
    
    @staticmethod
    def aggregate_vd_features(vd_detail, feature_names: List[str]) -> List[float]:
        """Aggregate features from all lanes of a VD."""
        from ..utils.json_utils import VDLiveDetail
        
        if not vd_detail.Lanes:
            return [np.nan] * len(feature_names)
        
        # Extract features from each lane
        lane_features = []
        for lane in vd_detail.Lanes:
            lane_feat = TrafficFeatureExtractor.extract_lane_features(lane)
            lane_features.append(lane_feat)
        
        # Aggregate features
        aggregated = []
        for feature_name in feature_names:
            if feature_name == 'avg_speed':
                speeds = [lf['speed'] for lf in lane_features if not np.isnan(lf['speed'])]
                aggregated.append(np.mean(speeds) if speeds else np.nan)
            
            elif feature_name == 'total_volume':
                volumes = [lf['volume'] for lf in lane_features if not np.isnan(lf['volume'])]
                aggregated.append(np.sum(volumes) if volumes else np.nan)
            
            elif feature_name == 'avg_occupancy':
                occupancies = [lf['occupancy'] for lf in lane_features if not np.isnan(lf['occupancy'])]
                aggregated.append(np.mean(occupancies) if occupancies else np.nan)
            
            elif feature_name == 'speed_std':
                speeds = [lf['speed'] for lf in lane_features if not np.isnan(lf['speed'])]
                aggregated.append(np.std(speeds) if len(speeds) > 1 else np.nan)
            
            elif feature_name == 'lane_count':
                aggregated.append(float(len(vd_detail.Lanes)))
            
            else:
                aggregated.append(np.nan)
        
        return aggregated


class TrafficHDF5Converter:
    """Converts time-series traffic JSON data to HDF5 format."""
    
    def __init__(self, config: TrafficHDF5Config):
        self.config = config
        self.extractor = TrafficFeatureExtractor()
        
        # Cache for VD information
        self._vd_info_cache: Optional[Dict[str, Any]] = None
        self._available_vdids: Optional[List[str]] = None
    
    def _check_existing_file(self) -> bool:
        """Check if output file exists and validate configuration consistency."""
        if not self.config.output_path.exists():
            return False
        
        if not self.config.check_consistency:
            return True
        
        try:
            reader = TrafficHDF5Reader(self.config.output_path)
            metadata = reader.get_metadata()
            
            # Check if configuration matches existing file
            existing_vdids = set(metadata['vdids'])
            existing_features = set(metadata['feature_names'])
            existing_source = metadata.get('source_directory', '')
            
            config_vdids = set(self.config.selected_vdids) if self.config.selected_vdids else set()
            config_features = set(self.config.feature_names)
            config_source = str(self.config.source_dir.resolve())
            
            # Compare configurations
            vdids_match = (not config_vdids) or (config_vdids == existing_vdids)
            features_match = config_features == existing_features
            source_match = config_source == existing_source
            
            if not (vdids_match and features_match and source_match):
                print("Warning: Existing HDF5 file configuration differs from current config:")
                if not vdids_match:
                    print(f"  VDIDs differ: {config_vdids} vs {existing_vdids}")
                if not features_match:
                    print(f"  Features differ: {config_features} vs {existing_features}")
                if not source_match:
                    print(f"  Source differs: {config_source} vs {existing_source}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Warning: Cannot validate existing HDF5 file: {e}")
            return False
    
    def _get_source_modification_time(self) -> float:
        """Get the latest modification time of source JSON files."""
        latest_mtime = 0.0
        
        for dir_path in self.config.source_dir.iterdir():
            if not dir_path.is_dir():
                continue
            
            vd_list_file = dir_path / "VDList.json"
            vd_live_file = dir_path / "VDLiveList.json"
            
            if vd_list_file.exists():
                latest_mtime = max(latest_mtime, vd_list_file.stat().st_mtime)
            if vd_live_file.exists():
                latest_mtime = max(latest_mtime, vd_live_file.stat().st_mtime)
        
        return latest_mtime
    
    def _is_hdf5_outdated(self) -> bool:
        """Check if HDF5 file is older than source data."""
        if not self.config.output_path.exists():
            return True
        
        hdf5_mtime = self.config.output_path.stat().st_mtime
        source_mtime = self._get_source_modification_time()
        
        return source_mtime > hdf5_mtime
    
    def _parse_timestamp(self, dirname: str) -> Optional[datetime.datetime]:
        """Parse timestamp from directory name."""
        try:
            return datetime.datetime.strptime(dirname, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            return None
    
    def _get_sorted_time_directories(self) -> List[Tuple[datetime.datetime, Path]]:
        """Get time-sorted data directories."""
        time_dirs = []
        
        for dir_path in self.config.source_dir.iterdir():
            if not dir_path.is_dir():
                continue
            
            # Skip common non-data directories
            if dir_path.name in ['cache', '__pycache__', '.git', '.DS_Store', 'logs']:
                continue
            
            timestamp = self._parse_timestamp(dir_path.name)
            if timestamp is None:
                print(f"Skipping directory with invalid timestamp format: {dir_path.name}")
                continue
            
            # Filter by time range if specified
            if self.config.time_range:
                start_str, end_str = self.config.time_range
                start_time = datetime.datetime.strptime(start_str, "%Y-%m-%d_%H-%M-%S")
                end_time = datetime.datetime.strptime(end_str, "%Y-%m-%d_%H-%M-%S")
                
                if not (start_time <= timestamp <= end_time):
                    continue
        
            # Check if required files exist
            vd_list_file = dir_path / "VDList.json"
            vd_live_file = dir_path / "VDLiveList.json"
            
            if vd_list_file.exists() and vd_live_file.exists():
                time_dirs.append((timestamp, dir_path))
            else:
                print(f"Skipping directory missing required JSON files: {dir_path.name}")
    
        return sorted(time_dirs, key=lambda x: x[0])
    
    def _load_vd_info(self, sample_dir: Path) -> Dict[str, Any]:
        """Load VD information from a sample directory."""
        if self._vd_info_cache is not None:
            return self._vd_info_cache
            
        vd_list_file = sample_dir / "VDList.json"
        vd_info = VDInfo.load_from_json(vd_list_file)
        
        # Create mapping from VDID to VD info
        vd_mapping = {}
        for vd in vd_info.VDList:
            vd_mapping[vd.VDID] = {
                'position_lon': vd.PositionLon,
                'position_lat': vd.PositionLat,
                'road_id': vd.RoadID,
                'road_name': vd.RoadName,
                'lane_num': vd.LaneNum
            }
        
        self._vd_info_cache = vd_mapping
        self._available_vdids = list(vd_mapping.keys())
        
        return vd_mapping
    
    def _get_target_vdids(self, sample_dir: Path) -> List[str]:
        """Get the list of target VDIDs to process."""
        if self.config.selected_vdids is not None:
            return self.config.selected_vdids
        
        # Load all available VDIDs
        self._load_vd_info(sample_dir)
        return self._available_vdids
    
    def _process_timestep(self, dir_path: Path, target_vdids: List[str]) -> Tuple[str, np.ndarray]:
        """Process a single timestep directory."""
        vd_live_file = dir_path / "VDLiveList.json"
        vd_live_list = VDLiveList.load_from_json(vd_live_file)
        
        # Create mapping for quick lookup
        vd_data_map = {vd.VDID: vd for vd in vd_live_list.LiveTrafficData}
        
        # Extract timestamp
        if vd_live_list.LiveTrafficData:
            timestamp = vd_live_list.LiveTrafficData[0].DataCollectTime
        else:
            timestamp = self._parse_timestamp(dir_path.name).isoformat()
        
        # Initialize feature matrix
        features = np.full(
            (len(target_vdids), len(self.config.feature_names)), 
            np.nan, 
            dtype=np.float32
        )
        
        # Check for missing VDIDs and report count
        missing_vdids = [vdid for vdid in target_vdids if vdid not in vd_data_map]
        if missing_vdids:
            print(f"Warning: {len(missing_vdids)} missing VDIDs in {dir_path.name}: {missing_vdids}")
        
        # Extract features for each VD
        for i, vdid in enumerate(target_vdids):
            if vdid in vd_data_map:
                try:
                    vd_features = self.extractor.aggregate_vd_features(
                        vd_data_map[vdid], 
                        self.config.feature_names
                    )
                    vd_features_array = np.array(vd_features, dtype=np.float32)
                    features[i, :] = vd_features_array
                except Exception as e:
                    print(f"Error processing VDID {vdid} in {dir_path.name}: {e}")
                    continue
        
        return timestamp, features
    
    def _create_hdf5_structure(self, h5file: h5py.File, target_vdids: List[str], 
                              vd_info: Dict[str, Any], num_timesteps: int):
        """Create the HDF5 file structure."""
        # Create groups
        metadata_group = h5file.create_group('metadata')
        data_group = h5file.create_group('data')
        
        # Store VD information
        vd_group = metadata_group.create_group('vd_info')
        for vdid in target_vdids:
            if vdid in vd_info:
                vd_subgroup = vd_group.create_group(vdid)
                info = vd_info[vdid]
                vd_subgroup.attrs['position_lon'] = info['position_lon']
                vd_subgroup.attrs['position_lat'] = info['position_lat']
                vd_subgroup.attrs['road_id'] = info['road_id']
                vd_subgroup.attrs['road_name'] = info['road_name']
                vd_subgroup.attrs['lane_num'] = info['lane_num']
        
        # Store metadata arrays
        vdids_data = np.array(target_vdids, dtype=h5py.string_dtype(encoding='utf-8'))
        metadata_group.create_dataset('vdids', data=vdids_data)
        
        features_data = np.array(self.config.feature_names, dtype=h5py.string_dtype(encoding='utf-8'))
        metadata_group.create_dataset('feature_names', data=features_data)
        
        # Create datasets for time-series data
        timestamps_dataset = metadata_group.create_dataset(
            'timestamps', 
            shape=(num_timesteps,), 
            dtype=h5py.string_dtype(encoding='utf-8'),
            compression=self.config.compression,
            compression_opts=self.config.compression_opts
        )
        
        features_dataset = data_group.create_dataset(
            'features',
            shape=(num_timesteps, len(target_vdids), len(self.config.feature_names)),
            dtype=np.float32,
            compression=self.config.compression,
            compression_opts=self.config.compression_opts,
            chunks=True
        )
        
        # Add attributes
        h5file.attrs['description'] = 'Traffic time-series data from VD sensors'
        h5file.attrs['creation_date'] = datetime.datetime.now().isoformat()
        h5file.attrs['source_directory'] = str(self.config.source_dir.resolve())
        h5file.attrs['num_timesteps'] = num_timesteps
        h5file.attrs['num_locations'] = len(target_vdids)
        h5file.attrs['num_features'] = len(self.config.feature_names)
        
        return timestamps_dataset, features_dataset
    
    def convert(self) -> None:
        """Convert traffic data to HDF5 format with smart checking."""
        # Check if conversion is needed
        if self.config.output_path.exists():
            if not self.config.overwrite:
                if self._check_existing_file():
                    if not self._is_hdf5_outdated():
                        print(f"HDF5 file already exists and is up-to-date: {self.config.output_path}")
                        print("Use overwrite=True to force regeneration")
                        return
                    else:
                        print("Source data is newer than HDF5 file, updating...")
                else:
                    print("Existing HDF5 configuration differs, recreating...")
            else:
                print(f"Overwriting existing HDF5 file: {self.config.output_path}")
        
        print(f"Starting conversion from {self.config.source_dir} to {self.config.output_path}")
        
        # Get sorted time directories
        time_dirs = self._get_sorted_time_directories()
        if not time_dirs:
            raise ValueError("No valid time directories found")
        
        print(f"Found {len(time_dirs)} time directories")
        
        # Load VD information and determine target VDIDs
        sample_dir = time_dirs[0][1]
        vd_info = self._load_vd_info(sample_dir)
        target_vdids = self._get_target_vdids(sample_dir)
        
        print(f"Processing {len(target_vdids)} VDs with {len(self.config.feature_names)} features")
        
        # Create output directory
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 file
        with h5py.File(self.config.output_path, 'w') as h5file:
            timestamps_ds, features_ds = self._create_hdf5_structure(
                h5file, target_vdids, vd_info, len(time_dirs)
            )
            
            # Process each timestep
            for i, (timestamp_dt, dir_path) in enumerate(time_dirs):
                try:
                    timestamp_str, features = self._process_timestep(dir_path, target_vdids)
                    
                    timestamps_ds[i] = timestamp_str
                    features_ds[i, :, :] = features
                    
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(time_dirs)} timesteps")
                        
                except Exception as e:
                    print(f"Error processing {dir_path.name}: {e}")
                    continue
        
        print(f"Conversion completed. Output saved to {self.config.output_path}")