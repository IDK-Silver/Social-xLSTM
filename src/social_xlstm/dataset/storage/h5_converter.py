"""HDF5 converter for traffic data."""

import h5py
import numpy as np
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging

from ..config import TrafficHDF5Config
from .h5_reader import TrafficHDF5Reader
from ..utils.json_utils import VDInfo, VDLiveList

logger = logging.getLogger(__name__)


class TrafficFeatureExtractor:
    """Extracts features from traffic data structures."""
    
    @staticmethod
    def validate_dataset_quality(dataset_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive dataset quality validation.
        
        Args:
            dataset_path: Path to the H5 dataset file
            
        Returns:
            Tuple of (quality_good: bool, quality_metrics: dict)
        """
        logger.info(f"Validating dataset quality for {dataset_path}")
        
        with h5py.File(dataset_path, 'r') as h5file:
            features = h5file['data/features'][:]
            feature_names = h5file['metadata/feature_names'][:]
            
            # Convert feature names to strings
            feature_names = [name.decode() if isinstance(name, bytes) else name for name in feature_names]
        
        # Test 80/20 split on data
        total_samples = len(features)
        train_end = int(total_samples * 0.8)
        
        train_data = features[:train_end]
        val_data = features[train_end:]
        
        logger.info(f"Dataset split - Total: {total_samples}, Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Check distribution consistency (primary VD only)
        quality_metrics = {}
        
        for feat_idx, feat_name in enumerate(feature_names[:3]):
            train_feat = train_data[:, 0, feat_idx]  # Primary VD
            val_feat = val_data[:, 0, feat_idx]
            
            # Remove invalid data
            train_valid = train_feat[np.isfinite(train_feat) & (train_feat != 0)]
            val_valid = val_feat[np.isfinite(val_feat) & (val_feat != 0)]
            
            if len(train_valid) > 10 and len(val_valid) > 10:
                train_mean = np.mean(train_valid)
                val_mean = np.mean(val_valid)
                train_std = np.std(train_valid)
                val_std = np.std(val_valid)
                
                mean_diff = abs(train_mean - val_mean) / train_mean if train_mean != 0 else 0
                std_diff = abs(train_std - val_std) / train_std if train_std != 0 else 0
                
                quality_metrics[feat_name] = {
                    'mean_diff': mean_diff, 
                    'std_diff': std_diff,
                    'train_mean': train_mean,
                    'val_mean': val_mean,
                    'train_std': train_std,
                    'val_std': val_std
                }
                
                # Log validation results
                mean_ok = mean_diff <= 0.10  # 10% threshold for stable data
                std_ok = std_diff <= 0.10
                overall_ok = mean_ok and std_ok
                
                logger.info(f"Feature {feat_name}: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}, quality={'OK' if overall_ok else 'POOR'}")
        
        # Overall assessment
        all_mean_diffs = [metrics['mean_diff'] for metrics in quality_metrics.values()]
        all_std_diffs = [metrics['std_diff'] for metrics in quality_metrics.values()]
        
        max_mean_diff = max(all_mean_diffs) if all_mean_diffs else 0
        max_std_diff = max(all_std_diffs) if all_std_diffs else 0
        
        data_quality_good = max_mean_diff <= 0.10 and max_std_diff <= 0.10
        
        quality_metrics['overall'] = {
            'max_mean_diff': max_mean_diff,
            'max_std_diff': max_std_diff,
            'quality_good': data_quality_good
        }
        
        logger.info(f"Overall dataset quality: {'GOOD' if data_quality_good else 'NEEDS IMPROVEMENT'}")
        
        return data_quality_good, quality_metrics
    
    @staticmethod
    def stabilize_dataset(input_h5_path: str, output_h5_path: str, start_ratio: float = 0.3) -> str:
        """
        Create dataset using only the stable later portion of the data.
        
        Based on analysis, data quality improves significantly after ~30% of the timeline.
        This method removes the problematic early period to ensure consistent quality.
        
        Args:
            input_h5_path: Path to the input H5 dataset file
            output_h5_path: Path to save the stabilized dataset
            start_ratio: Ratio of data to skip from the beginning (default: 0.3)
            
        Returns:
            Path to the stabilized dataset file
        """
        logger.info(f"Creating stable dataset from {input_h5_path}")
        
        with h5py.File(input_h5_path, 'r') as input_file:
            # Load original data
            features = input_file['data/features'][:]
            timestamps = input_file['metadata/timestamps'][:]
            feature_names = input_file['metadata/feature_names'][:]
            vdids = input_file['metadata/vdids'][:]
            
            logger.info(f"Original data shape: {features.shape}")
            
            # Use data from start_ratio onwards (skip problematic early period)
            start_idx = int(len(features) * start_ratio)
            
            stable_features = features[start_idx:]
            stable_timestamps = timestamps[start_idx:]
            
            logger.info(f"Stable data shape: {stable_features.shape}")
            logger.info(f"Removed {start_idx} samples ({start_ratio*100:.0f}% of data)")
            
            # Check quality of stable data
            logger.info("Checking stable data quality...")
            for vd_idx in range(min(3, stable_features.shape[1])):
                vd_name = vdids[vd_idx].decode() if isinstance(vdids[vd_idx], bytes) else vdids[vd_idx]
                
                for feat_idx in range(min(3, stable_features.shape[2])):
                    feat_name = feature_names[feat_idx].decode() if isinstance(feature_names[feat_idx], bytes) else feature_names[feat_idx]
                    feat_data = stable_features[:, vd_idx, feat_idx]
                    
                    valid_count = np.sum(np.isfinite(feat_data) & (feat_data != 0))
                    quality = valid_count / len(feat_data) * 100 if len(feat_data) > 0 else 0
                    
                    logger.info(f"  {vd_name} - {feat_name}: {quality:.1f}% valid")
        
        # Save stable dataset
        Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_h5_path, 'w') as output_file:
            # Copy structure
            data_group = output_file.create_group('data')
            data_group.create_dataset('features', data=stable_features)
            
            metadata_group = output_file.create_group('metadata')
            metadata_group.create_dataset('feature_names', data=feature_names)
            metadata_group.create_dataset('timestamps', data=stable_timestamps)
            metadata_group.create_dataset('vdids', data=vdids)
            
            # Add cleaning info
            cleaning_group = metadata_group.create_group('cleaning_info')
            cleaning_group.attrs['original_samples'] = len(features)
            cleaning_group.attrs['stable_samples'] = len(stable_features)
            cleaning_group.attrs['start_ratio'] = start_ratio
            cleaning_group.attrs['removed_samples'] = start_idx
        
        logger.info(f"Stable dataset saved to: {output_h5_path}")
        return output_h5_path
    
    @staticmethod
    def extract_lane_features(lane) -> Dict[str, float]:
        """Extract features from a single lane."""
        from ..utils.json_utils import LaneData
        
        # Filter out error codes and invalid values
        def is_valid_value(value, min_val=0, max_val=None):
            """Check if value is valid (not error code or out of range)."""
            if value is None or np.isnan(value):
                return False
            # Common error codes in traffic data: -99, -1, 255, etc.
            if value in [-99, -1, 255]:
                return False
            if value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        
        # Process speed (reasonable range: 0-200 km/h)
        speed = float(lane.Speed) if is_valid_value(lane.Speed, 0, 200) else np.nan
        
        # Process occupancy (reasonable range: 0-100%)
        occupancy = float(lane.Occupancy) if is_valid_value(lane.Occupancy, 0, 100) else np.nan
        
        # Process volume (filter out negative volumes from error codes)
        if lane.Vehicles:
            valid_volumes = [v.Volume for v in lane.Vehicles if is_valid_value(v.Volume, 0)]
            total_volume = sum(valid_volumes) if valid_volumes else 0
        else:
            total_volume = 0
        
        return {
            'speed': speed,
            'occupancy': occupancy,
            'volume': float(total_volume)
        }
    
    @staticmethod
    def aggregate_vd_features(vd_detail, feature_names: List[str]) -> List[float]:
        """Aggregate features from all lanes of a VD using vectorized operations."""
        from ..utils.json_utils import VDLiveDetail
        
        # Get all lanes from all LinkFlows
        all_lanes = []
        for link_flow in vd_detail.LinkFlows:
            all_lanes.extend(link_flow.Lanes)
        
        if not all_lanes:
            return [np.nan] * len(feature_names)
        
        # Vectorized feature extraction
        return TrafficFeatureExtractor._vectorized_lane_aggregation(all_lanes, feature_names)
    
    @staticmethod
    def _vectorized_lane_aggregation(all_lanes, feature_names: List[str]) -> List[float]:
        """Vectorized aggregation of lane features for better performance."""
        if not all_lanes:
            return [np.nan] * len(feature_names)
        
        # Pre-allocate arrays for all lane data
        num_lanes = len(all_lanes)
        speeds = np.full(num_lanes, np.nan, dtype=np.float32)
        occupancies = np.full(num_lanes, np.nan, dtype=np.float32)
        volumes = np.full(num_lanes, 0.0, dtype=np.float32)
        
        # Vectorized data extraction with error filtering
        for i, lane in enumerate(all_lanes):
            # Process speed with error code filtering
            speed_val = float(lane.Speed)
            if TrafficFeatureExtractor._is_valid_speed(speed_val):
                speeds[i] = speed_val
            
            # Process occupancy with error code filtering
            occupancy_val = float(lane.Occupancy)
            if TrafficFeatureExtractor._is_valid_occupancy(occupancy_val):
                occupancies[i] = occupancy_val
            
            # Process volume (sum all vehicle volumes in lane)
            if lane.Vehicles:
                lane_volume = sum(
                    v.Volume for v in lane.Vehicles 
                    if TrafficFeatureExtractor._is_valid_volume(v.Volume)
                )
                volumes[i] = float(lane_volume)
        
        # Vectorized feature computation
        aggregated = []
        for feature_name in feature_names:
            if feature_name == 'avg_speed':
                valid_speeds = speeds[~np.isnan(speeds)]
                aggregated.append(np.mean(valid_speeds) if len(valid_speeds) > 0 else np.nan)
            
            elif feature_name == 'total_volume':
                # Sum all volumes (NaN values are already filtered out)
                aggregated.append(np.sum(volumes))
            
            elif feature_name == 'avg_occupancy':
                valid_occupancies = occupancies[~np.isnan(occupancies)]
                aggregated.append(np.mean(valid_occupancies) if len(valid_occupancies) > 0 else np.nan)
            
            elif feature_name == 'speed_std':
                valid_speeds = speeds[~np.isnan(speeds)]
                aggregated.append(np.std(valid_speeds) if len(valid_speeds) > 1 else np.nan)
            
            elif feature_name == 'lane_count':
                aggregated.append(float(num_lanes))
            
            else:
                aggregated.append(np.nan)
        
        return aggregated
    
    @staticmethod
    def _is_valid_speed(value) -> bool:
        """Fast speed validation (0-200 km/h range)."""
        return not (value in [-99, -1, 255] or value < 0 or value > 200 or np.isnan(value))
    
    @staticmethod
    def _is_valid_occupancy(value) -> bool:
        """Fast occupancy validation (0-100% range)."""
        return not (value in [-99, -1, 255] or value < 0 or value > 100 or np.isnan(value))
    
    @staticmethod
    def _is_valid_volume(value) -> bool:
        """Fast volume validation (non-negative)."""
        return not (value in [-99, -1, 255] or value < 0 or np.isnan(value))


class TrafficHDF5Converter:
    """Converts time-series traffic JSON data to HDF5 format."""
    
    def __init__(self, config: TrafficHDF5Config):
        self.config = config
        self.extractor = TrafficFeatureExtractor()
        
        # Cache for VD information
        self._vd_info_cache: Optional[Dict[str, Any]] = None
        self._available_vdids: Optional[List[str]] = None
        
        # Performance options
        self.show_progress = getattr(config, 'show_progress', True)
        self.verbose_warnings = getattr(config, 'verbose_warnings', False)
        self.max_missing_report = getattr(config, 'max_missing_report', 10)
    
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
                if self.verbose_warnings:
                    logger.warning(f"Skipping directory with invalid timestamp format: {dir_path.name}")
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
            elif self.verbose_warnings:
                logger.warning(f"Skipping directory missing required JSON files: {dir_path.name}")
    
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
    
    def _batch_read_vdlive_files(self, dir_paths: List[Path]) -> List[Tuple[str, 'VDLiveList']]:
        """Batch read VDLiveList files using existing load_from_json method."""
        import concurrent.futures
        from ..utils.json_utils import VDLiveList
        
        def read_single_vdlive(dir_path: Path) -> Tuple[str, 'VDLiveList']:
            """Read a single VDLiveList file using standard method."""
            vd_live_file = dir_path / "VDLiveList.json"
            try:
                # Use existing standard method (maintains consistency)
                vd_live_list = VDLiveList.load_from_json(vd_live_file)
                return dir_path.name, vd_live_list
            except Exception as e:
                logger.warning(f"Failed to read {vd_live_file}: {e}")
                return dir_path.name, None
        
        # Batch process with threading for I/O bound operations
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(dir_paths))) as executor:
            # Submit all read tasks
            future_to_path = {
                executor.submit(read_single_vdlive, dir_path): dir_path 
                for dir_path in dir_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                result = future.result()
                if result[1] is not None:  # Only add successful reads
                    results.append(result)
        
        return results
    
    def _process_vdlive_data(self, vd_live_list: 'VDLiveList', dir_name: str, target_vdids: List[str]) -> Tuple[str, np.ndarray]:
        """Process VDLiveList data using vectorized operations for better performance."""
        # Create mapping for quick lookup
        vd_data_map = {vd.VDID: vd for vd in vd_live_list.LiveTrafficData}
        
        # Extract timestamp
        if vd_live_list.LiveTrafficData:
            timestamp = vd_live_list.LiveTrafficData[0].DataCollectTime
        else:
            timestamp = self._parse_timestamp(dir_name).isoformat()
        
        # Use vectorized batch processing for multiple VDs
        features = self._vectorized_vd_batch_processing(vd_data_map, target_vdids, dir_name)
        
        return timestamp, features
    
    def _vectorized_vd_batch_processing(self, vd_data_map: Dict, target_vdids: List[str], dir_name: str) -> np.ndarray:
        """Vectorized processing of multiple VDs for better performance."""
        num_vds = len(target_vdids)
        num_features = len(self.config.feature_names)
        
        # Pre-allocate result matrix
        features = np.full((num_vds, num_features), np.nan, dtype=np.float32)
        
        # Group VDs that exist in data for batch processing
        valid_vd_indices = []
        valid_vd_data = []
        
        for i, vdid in enumerate(target_vdids):
            if vdid in vd_data_map:
                valid_vd_indices.append(i)
                valid_vd_data.append(vd_data_map[vdid])
        
        if not valid_vd_data:
            return features
        
        # Process valid VDs in batch
        try:
            batch_features = self._batch_extract_features(valid_vd_data, self.config.feature_names)
            
            # Assign results back to the full matrix
            for idx, vd_idx in enumerate(valid_vd_indices):
                features[vd_idx, :] = batch_features[idx, :]
                
        except Exception as e:
            if self.verbose_warnings:
                logger.error(f"Error in batch processing for {dir_name}: {e}")
            # Fallback to individual processing
            for i, vdid in enumerate(target_vdids):
                if vdid in vd_data_map:
                    try:
                        vd_features = self.extractor.aggregate_vd_features(
                            vd_data_map[vdid], 
                            self.config.feature_names
                        )
                        features[i, :] = np.array(vd_features, dtype=np.float32)
                    except Exception as inner_e:
                        if self.verbose_warnings:
                            logger.error(f"Error processing VDID {vdid} in {dir_name}: {inner_e}")
                        continue
        
        return features
    
    def _batch_extract_features(self, vd_data_list: List, feature_names: List[str]) -> np.ndarray:
        """Extract features from multiple VDs in batch for maximum performance."""
        num_vds = len(vd_data_list)
        num_features = len(feature_names)
        
        # Pre-allocate result matrix
        batch_features = np.full((num_vds, num_features), np.nan, dtype=np.float32)
        
        # Pre-compute all lane data for all VDs
        all_lanes_data = []
        vd_lane_counts = []
        
        for vd_detail in vd_data_list:
            # Get all lanes from all LinkFlows for this VD
            vd_lanes = []
            for link_flow in vd_detail.LinkFlows:
                vd_lanes.extend(link_flow.Lanes)
            all_lanes_data.append(vd_lanes)
            vd_lane_counts.append(len(vd_lanes))
        
        # Process each VD's lanes with vectorized operations
        for vd_idx, lanes in enumerate(all_lanes_data):
            if not lanes:
                continue
            
            try:
                # Use the optimized vectorized aggregation
                vd_features = TrafficFeatureExtractor._vectorized_lane_aggregation(lanes, feature_names)
                batch_features[vd_idx, :] = np.array(vd_features, dtype=np.float32)
            except Exception as e:
                if self.verbose_warnings:
                    logger.warning(f"Error processing VD {vd_idx} lanes: {e}")
                continue
        
        return batch_features
    
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
        
        # Only report detailed missing VDIDs in verbose mode
        if self.verbose_warnings:
            missing_vdids = [vdid for vdid in target_vdids if vdid not in vd_data_map]
            if missing_vdids:
                # Limit the number of VDIDs shown
                shown_vdids = missing_vdids[:self.max_missing_report]
                extra = len(missing_vdids) - len(shown_vdids)
                msg = f"Warning: {len(missing_vdids)} missing VDIDs in {dir_path.name}"
                if extra > 0:
                    msg += f": {shown_vdids} ... and {extra} more"
                else:
                    msg += f": {shown_vdids}"
                logger.warning(msg)
        
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
                    if self.verbose_warnings:
                        logger.error(f"Error processing VDID {vdid} in {dir_path.name}: {e}")
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
        
        logger.info(f"Starting conversion from {self.config.source_dir} to {self.config.output_path}")
        
        # Get sorted time directories
        time_dirs = self._get_sorted_time_directories()
        if not time_dirs:
            raise ValueError("No valid time directories found")
        
        logger.info(f"Found {len(time_dirs)} time directories")
        
        # Load VD information and determine target VDIDs
        sample_dir = time_dirs[0][1]
        vd_info = self._load_vd_info(sample_dir)
        target_vdids = self._get_target_vdids(sample_dir)
        
        logger.info(f"Processing {len(target_vdids)} VDs with {len(self.config.feature_names)} features")
        
        # Create output directory
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 file
        with h5py.File(self.config.output_path, 'w') as h5file:
            timestamps_ds, features_ds = self._create_hdf5_structure(
                h5file, target_vdids, vd_info, len(time_dirs)
            )
            
            # Process timesteps with batch optimization
            batch_size = min(20, len(time_dirs))  # Adjust batch size based on data size
            successful_count = 0
            
            # Progress bar for batches
            dir_paths_only = [dir_path for _, dir_path in time_dirs]
            
            with tqdm(total=len(time_dirs), desc="Processing timesteps", disable=not self.show_progress) as progress_bar:
                # Process in batches
                for batch_start in range(0, len(dir_paths_only), batch_size):
                    batch_end = min(batch_start + batch_size, len(dir_paths_only))
                    batch_paths = dir_paths_only[batch_start:batch_end]
                    
                    try:
                        # Batch read VDLiveList files using standard methods
                        batch_vdlive_data = self._batch_read_vdlive_files(batch_paths)
                        
                        # Process each file in the batch
                        for dir_name, vd_live_list in batch_vdlive_data:
                            try:
                                # Find the corresponding index
                                dir_index = None
                                for j, (_, path) in enumerate(time_dirs):
                                    if path.name == dir_name:
                                        dir_index = j
                                        break
                                
                                if dir_index is None:
                                    continue
                                
                                # Process using standard VDLiveList object
                                timestamp_str, features = self._process_vdlive_data(vd_live_list, dir_name, target_vdids)
                                
                                timestamps_ds[dir_index] = timestamp_str
                                features_ds[dir_index, :, :] = features
                                successful_count += 1
                                
                            except Exception as e:
                                logger.error(f"Error processing {dir_name}: {e}")
                                continue
                        
                        # Update progress
                        progress_bar.update(len(batch_paths))
                        if self.show_progress:
                            progress_bar.set_postfix({
                                'success': successful_count,
                                'failed': progress_bar.n - successful_count,
                                'batch': f"{batch_start//batch_size + 1}/{(len(dir_paths_only) + batch_size - 1)//batch_size}"
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing batch starting at {batch_start}: {e}")
                        progress_bar.update(len(batch_paths))
                        continue
        
        logger.info(f"Conversion completed. Processed {successful_count}/{len(time_dirs)} timesteps.")
        logger.info(f"Output saved to {self.config.output_path}")