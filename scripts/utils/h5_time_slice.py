#!/usr/bin/env python3
"""
Universal HDF5 time slicing tool for Social-xLSTM project.

This tool creates smaller HDF5 files by extracting specified time ranges
from larger datasets, maintaining complete data structure and metadata.

Usage examples:
    # Slice by index range (72 timesteps = 6 hours at 5min intervals)
    python scripts/utils/h5_time_slice.py \
        --input blob/dataset/processed/pems_bay.h5 \
        --output blob/dataset/processed/pems_bay_test.h5 \
        --start-index 0 --length 72 --progress --atomic

    # Slice by absolute time and duration
    python scripts/utils/h5_time_slice.py \
        --input blob/dataset/processed/pems_bay.h5 \
        --output blob/dataset/processed/pems_bay_test.h5 \
        --start-ts "2017-05-01T00:00:00Z" \
        --duration 6h --progress --atomic
"""

import argparse
import os
import sys
import shutil
import h5py
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def parse_duration_to_seconds(s: str) -> int:
    """Parse duration string to seconds (e.g., '6h', '30m', '120s')."""
    s = s.strip().lower()
    if s.startswith('+'):
        s = s[1:]
    
    units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    for unit, multiplier in units.items():
        if s.endswith(unit):
            return int(float(s[:-1]) * multiplier)
    
    # Default to seconds if no unit specified
    return int(float(s))


def parse_epoch_or_iso(s: str) -> int:
    """Parse timestamp string to epoch seconds."""
    s = s.strip()
    
    # Handle numeric epoch (seconds, milliseconds, or nanoseconds)
    if s.isdigit():
        value = int(s)
        if value > 1e12:  # nanoseconds
            return value // 1_000_000_000
        elif value > 1e10:  # milliseconds
            return value // 1000
        else:  # seconds
            return value
    
    # Handle ISO8601 format
    iso_string = s.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(iso_string)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {s}")


def detect_timestamp_unit_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps to epoch seconds, auto-detecting the unit."""
    timestamps = np.asarray(timestamps)
    
    # Handle string timestamps (convert to datetime then epoch)
    if timestamps.dtype.kind in ['S', 'U', 'O']:  # String or object types
        # Try to parse as datetime strings
        try:
            # Convert bytes to strings if needed
            if timestamps.dtype.kind == 'S':
                timestamps = np.array([ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps])
            
            # Parse datetime strings to epoch seconds
            epoch_times = []
            for ts_str in timestamps:
                try:
                    # Handle common formats
                    if isinstance(ts_str, str):
                        dt = datetime.fromisoformat(ts_str.replace(' ', 'T'))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        epoch_times.append(int(dt.timestamp()))
                    else:
                        epoch_times.append(0)  # fallback
                except:
                    epoch_times.append(0)  # fallback for unparseable dates
            
            return np.array(epoch_times, dtype=np.int64)
        except Exception:
            raise ValueError(f"Cannot parse timestamp strings: {timestamps[:3]}")
    
    # Handle numeric timestamps
    max_value = np.nanmax(timestamps)
    if max_value > 1e12:  # nanoseconds
        return (timestamps // 1_000_000_000).astype(np.int64)
    elif max_value > 1e10:  # milliseconds
        return (timestamps // 1000).astype(np.int64)
    else:  # already in seconds
        return timestamps.astype(np.int64)


def search_time_range_indices(
    timestamps_epoch: np.ndarray, 
    start_epoch: Optional[int], 
    end_epoch: Optional[int]
) -> Tuple[int, int]:
    """Find start and end indices for the given time range."""
    if start_epoch is None:
        start_idx = 0
    else:
        start_idx = int(np.searchsorted(timestamps_epoch, start_epoch, side='left'))
    
    if end_epoch is None:
        end_idx = len(timestamps_epoch)
    else:
        end_idx = int(np.searchsorted(timestamps_epoch, end_epoch, side='right'))
    
    return max(0, start_idx), min(len(timestamps_epoch), end_idx)


def is_time_like_dataset(dataset: h5py.Dataset, total_timesteps: int, time_axis: int) -> bool:
    """Check if a dataset should be sliced along the time axis."""
    if dataset.shape is None or len(dataset.shape) == 0:
        return False
    
    # Adjust negative time_axis
    if time_axis < 0:
        time_axis += len(dataset.shape)
    
    # Check if the dataset has the expected number of timesteps along time_axis
    if time_axis >= len(dataset.shape):
        return False
    
    return dataset.shape[time_axis] == total_timesteps


def get_dataset_creation_kwargs(src_dataset: h5py.Dataset, new_shape: Optional[tuple] = None) -> dict:
    """Extract creation parameters from source dataset."""
    kwargs = {'dtype': src_dataset.dtype}
    
    # Handle chunks - adjust if new shape is smaller
    if src_dataset.chunks:
        chunks = list(src_dataset.chunks)
        if new_shape:
            # Adjust chunk size to be no larger than the new shape
            for i, (chunk_size, new_size) in enumerate(zip(chunks, new_shape)):
                chunks[i] = min(chunk_size, new_size)
        kwargs['chunks'] = tuple(chunks)
    
    # Preserve compression and layout settings
    if src_dataset.compression:
        kwargs['compression'] = src_dataset.compression
        if src_dataset.compression_opts:
            kwargs['compression_opts'] = src_dataset.compression_opts
    if src_dataset.shuffle:
        kwargs['shuffle'] = True
    if src_dataset.fletcher32:
        kwargs['fletcher32'] = True
    if src_dataset.scaleoffset is not None:
        kwargs['scaleoffset'] = src_dataset.scaleoffset
    if src_dataset.fillvalue is not None:
        kwargs['fillvalue'] = src_dataset.fillvalue
        
    return kwargs


def copy_attributes(src, dst):
    """Copy all attributes from source to destination."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def calculate_auto_block_size(dataset: h5py.Dataset, time_axis: int) -> int:
    """Calculate appropriate block size for chunked copying."""
    if dataset.chunks:
        return max(1, dataset.chunks[time_axis])
    
    # Estimate block size to target ~8MB per block
    element_size = dataset.dtype.itemsize
    elements_per_timestep = int(np.prod([
        dataset.shape[i] for i in range(dataset.ndim) 
        if i != time_axis
    ]))
    
    if elements_per_timestep == 0:
        return 1
    
    target_bytes = 8 * 1024 * 1024  # 8MB
    timesteps_per_block = max(1, target_bytes // (element_size * elements_per_timestep))
    return int(timesteps_per_block)


def slice_and_copy_dataset(
    src_dataset: h5py.Dataset,
    dst_dataset: h5py.Dataset,
    time_slice: slice,
    time_axis: int,
    block_size: int,
    show_progress: bool
):
    """Copy dataset data with time slicing using block-wise operations."""
    start_idx = time_slice.start
    end_idx = time_slice.stop
    total_timesteps = end_idx - start_idx
    
    # Create progress bar if requested and available
    progress_iter = range(start_idx, end_idx, block_size)
    if show_progress and HAS_TQDM and total_timesteps > block_size:
        progress_iter = tqdm(
            progress_iter, 
            desc=f"Copying {src_dataset.name}",
            unit="blocks",
            leave=False
        )
    
    # Copy data in blocks
    for block_start in progress_iter:
        block_end = min(end_idx, block_start + block_size)
        
        # Create slicing tuples
        src_slice_tuple = [slice(None)] * src_dataset.ndim
        src_slice_tuple[time_axis] = slice(block_start, block_end)
        src_slice_tuple = tuple(src_slice_tuple)
        
        dst_slice_tuple = [slice(None)] * src_dataset.ndim
        dst_slice_tuple[time_axis] = slice(block_start - start_idx, block_end - start_idx)
        dst_slice_tuple = tuple(dst_slice_tuple)
        
        # Copy data block
        data_block = src_dataset[src_slice_tuple]
        dst_dataset[dst_slice_tuple] = data_block


def find_timestamps_dataset(h5_file: h5py.File, user_path: Optional[str]) -> Tuple[str, np.ndarray]:
    """Find and return the timestamps dataset path and data."""
    # Try user-specified path first
    candidates = []
    if user_path:
        candidates.append(user_path)
    
    # Prefer numeric epoch timestamps if available
    epoch_candidates = [
        '/metadata/timestamps_epoch',
        '/meta/timestamps_epoch',
        '/timestamps_epoch',
        '/time/timestamps_epoch'
    ]
    
    # Add common timestamp paths (prefer numeric first)
    candidates.extend(epoch_candidates)
    candidates.extend([
        '/metadata/timestamps',
        '/meta/timestamps', 
        '/timestamps',
        '/time/timestamps',
        '/data/timestamps',
        '/time_index'
    ])
    
    # Check each candidate path
    for path in candidates:
        if path in h5_file:
            dataset = h5_file[path]
            if isinstance(dataset, h5py.Dataset) and dataset.ndim == 1:
                return path, np.array(dataset)
    
    # Fallback: search for 1D datasets with 'time' in the name
    def find_time_dataset(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
            basename = os.path.basename(name).lower()
            if 'time' in basename or 'ts' in basename:
                raise StopIteration((name, np.array(obj)))
    
    try:
        h5_file.visititems(find_time_dataset)
    except StopIteration as e:
        return e.value
    
    raise RuntimeError(
        "Cannot find timestamps dataset. Please specify --timestamp-dset path."
    )


def update_time_related_attributes(
    root_group: h5py.Group, 
    start_epoch: int, 
    end_epoch: int, 
    num_timesteps: int
):
    """Update time-related attributes in the root group."""
    # Common time attribute names to update
    time_attrs = {
        'start_time': start_epoch,
        'end_time': end_epoch,
        't0': start_epoch,
        't1': end_epoch,
        'num_timesteps': num_timesteps,
        'T': num_timesteps,
        'time_length': num_timesteps
    }
    
    # Update existing time attributes
    for attr_name, attr_value in time_attrs.items():
        if attr_name in root_group.attrs:
            root_group.attrs[attr_name] = int(attr_value)


def should_slice_dataset(
    dataset: h5py.Dataset,
    total_timesteps: int,
    time_axis: int,
    include_patterns: List[str],
    exclude_patterns: List[str]
) -> bool:
    """Determine if a dataset should be time-sliced."""
    from fnmatch import fnmatch
    
    dataset_name = dataset.name
    
    # Check exclude patterns first
    if any(fnmatch(dataset_name, pattern) for pattern in exclude_patterns):
        return False
    
    # Check include patterns
    if any(fnmatch(dataset_name, pattern) for pattern in include_patterns):
        return True
    
    # Default: slice if it looks like a time series dataset
    return is_time_like_dataset(dataset, total_timesteps, time_axis)


def h5_time_slice(input_path: str, output_path: str, args):
    """Main function to slice HDF5 file by time range."""
    # Setup atomic writing
    temp_output = output_path + ".tmp" if args.atomic else output_path
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(temp_output)), exist_ok=True)
    
    try:
        with h5py.File(input_path, 'r') as src_file, h5py.File(temp_output, 'w') as dst_file:
            # Copy root attributes
            copy_attributes(src_file, dst_file)
            
            # Find timestamps dataset and convert to epoch seconds
            timestamps_path, timestamps_array = find_timestamps_dataset(
                src_file, args.timestamp_dset
            )
            timestamps_epoch = detect_timestamp_unit_to_seconds(timestamps_array)
            total_timesteps = len(timestamps_epoch)
            
            print(f"Found timestamps dataset: {timestamps_path}")
            print(f"Total timesteps: {total_timesteps}")
            
            # Resolve time range to indices
            start_idx = args.start_index if args.start_index is not None else 0
            end_idx = None
            
            # Handle index-based slicing
            if args.end_index is not None:
                end_idx = args.end_index
            if args.length is not None:
                end_idx = start_idx + args.length if end_idx is None else end_idx
            
            # Handle time-based slicing
            if any([args.start_ts, args.end_ts, args.duration, args.start_offset]):
                start_epoch = None
                end_epoch = None
                
                if args.start_ts:
                    start_epoch = parse_epoch_or_iso(args.start_ts)
                elif args.start_offset:
                    start_epoch = int(timestamps_epoch[0]) + parse_duration_to_seconds(args.start_offset)
                
                if args.end_ts:
                    end_epoch = parse_epoch_or_iso(args.end_ts)
                elif args.duration and start_epoch:
                    duration_seconds = parse_duration_to_seconds(args.duration)
                    end_epoch = start_epoch + duration_seconds
                
                # Map time to indices
                time_start_idx, time_end_idx = search_time_range_indices(
                    timestamps_epoch, start_epoch, end_epoch
                )
                
                # Use time-based indices if not overridden by index args
                if args.start_index is None:
                    start_idx = time_start_idx
                if args.end_index is None and args.length is None:
                    end_idx = time_end_idx
            
            # Finalize indices
            if end_idx is None:
                end_idx = total_timesteps
            
            start_idx = max(0, min(total_timesteps, int(start_idx)))
            end_idx = max(start_idx, min(total_timesteps, int(end_idx)))
            time_slice = slice(start_idx, end_idx)
            new_timesteps = end_idx - start_idx
            
            print(f"Time range: indices [{start_idx}, {end_idx}), new timesteps: {new_timesteps}")
            
            if args.dry_run:
                print(f"\n[DRY RUN] Would slice {new_timesteps} timesteps")
                
                # List datasets and their slice status
                def list_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        will_slice = should_slice_dataset(
                            obj, total_timesteps, args.time_axis,
                            args.include_dsets, args.exclude_dsets
                        )
                        status = "SLICE" if will_slice else "COPY"
                        print(f"  {status:5} {name} shape={obj.shape} dtype={obj.dtype}")
                
                src_file.visititems(list_datasets)
                return
            
            # Recursive copy function
            def copy_group_recursive(src_group: h5py.Group, dst_group: h5py.Group):
                # Copy group attributes
                copy_attributes(src_group, dst_group)
                
                # Process each item in the group
                for name, item in src_group.items():
                    if isinstance(item, h5py.Group):
                        # Create subgroup and recurse
                        new_group = dst_group.create_group(name)
                        copy_group_recursive(item, new_group)
                        
                    elif isinstance(item, h5py.Dataset):
                        # Decide whether to slice or copy completely
                        should_slice = should_slice_dataset(
                            item, total_timesteps, args.time_axis,
                            args.include_dsets, args.exclude_dsets
                        )
                        
                        if should_slice:
                            # Create sliced dataset with adjusted shape
                            new_shape = list(item.shape)
                            time_axis = args.time_axis if args.time_axis >= 0 else (item.ndim + args.time_axis)
                            new_shape[time_axis] = new_timesteps
                            new_shape_tuple = tuple(new_shape)
                            
                            kwargs = get_dataset_creation_kwargs(item, new_shape_tuple)
                            new_dataset = dst_group.create_dataset(
                                name, shape=new_shape_tuple, **kwargs
                            )
                            copy_attributes(item, new_dataset)
                            
                            # Copy data with slicing
                            block_size = calculate_auto_block_size(item, time_axis)
                            slice_and_copy_dataset(
                                item, new_dataset, time_slice, time_axis, 
                                block_size, args.progress
                            )
                        else:
                            # Copy dataset completely
                            kwargs = get_dataset_creation_kwargs(item)
                            new_dataset = dst_group.create_dataset(
                                name, data=item[()], **kwargs
                            )
                            copy_attributes(item, new_dataset)
            
            # Perform the recursive copy
            copy_group_recursive(src_file, dst_file)
            
            # Update timestamps dataset if it exists
            if timestamps_path in dst_file:
                # Check if timestamps were properly sliced
                ts_dataset = dst_file[timestamps_path]
                if ts_dataset.shape[0] != new_timesteps:
                    # Timestamps weren't sliced, force slice it
                    del dst_file[timestamps_path]
                    sliced_timestamps = timestamps_array[time_slice]
                    
                    kwargs = get_dataset_creation_kwargs(src_file[timestamps_path], sliced_timestamps.shape)
                    new_ts_dataset = dst_file.create_dataset(
                        timestamps_path, data=sliced_timestamps, **kwargs
                    )
                    copy_attributes(src_file[timestamps_path], new_ts_dataset)
            
            # Update time-related attributes
            if new_timesteps > 0:
                start_epoch_final = int(timestamps_epoch[start_idx])
                end_epoch_final = int(timestamps_epoch[end_idx - 1])
            else:
                start_epoch_final = end_epoch_final = int(timestamps_epoch[0])
            
            update_time_related_attributes(
                dst_file, start_epoch_final, end_epoch_final, new_timesteps
            )
            
            print(f"Successfully created sliced HDF5 file: {temp_output}")
            
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_output) and args.atomic:
            os.remove(temp_output)
        raise e
    
    # Atomic rename if requested
    if args.atomic:
        os.replace(temp_output, output_path)
        print(f"Atomically moved to final location: {output_path}")


def build_argument_parser():
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Universal HDF5 time slicing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Slice by index range (first 72 timesteps)
    %(prog)s -i data/full.h5 -o data/test.h5 --start-index 0 --length 72
    
    # Slice by absolute time and duration
    %(prog)s -i data/full.h5 -o data/test.h5 --start-ts "2017-05-01T00:00:00Z" --duration 6h
    
    # Preview what would be sliced
    %(prog)s -i data/full.h5 -o data/test.h5 --start-index 0 --length 100 --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help='Input HDF5 file path')
    parser.add_argument('-o', '--output', required=True, help='Output HDF5 file path')
    
    # Time range specification
    time_group = parser.add_argument_group('time range specification')
    time_group.add_argument('--start-index', type=int, help='Start timestep index')
    time_group.add_argument('--end-index', type=int, help='End timestep index (exclusive)')
    time_group.add_argument('--length', type=int, help='Number of timesteps to extract')
    time_group.add_argument('--start-ts', help='Start timestamp (ISO8601 or epoch)')
    time_group.add_argument('--end-ts', help='End timestamp (ISO8601 or epoch)')
    time_group.add_argument('--duration', help='Duration (e.g., 6h, 30m, 120s)')
    time_group.add_argument('--start-offset', help='Start offset from file beginning (e.g., +30m)')
    
    # Dataset specification
    data_group = parser.add_argument_group('dataset specification')
    data_group.add_argument('--timestamp-dset', help='Path to timestamps dataset')
    data_group.add_argument('--time-axis', type=int, default=0, help='Time axis index (default: 0)')
    data_group.add_argument('--include-dsets', nargs='*', default=[], help='Dataset patterns to slice')
    data_group.add_argument('--exclude-dsets', nargs='*', default=[], help='Dataset patterns to not slice')
    
    # Options
    options_group = parser.add_argument_group('options')
    options_group.add_argument('--dry-run', action='store_true', help='Preview without creating output')
    options_group.add_argument('--progress', action='store_true', help='Show progress bars')
    options_group.add_argument('--atomic', action='store_true', help='Use atomic writes (tmp file)')
    options_group.add_argument('--strict', action='store_true', help='Strict mode for validation')
    
    return parser


def main():
    """Main entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Validate time range arguments
    index_args = [args.start_index, args.end_index, args.length]
    time_args = [args.start_ts, args.end_ts, args.duration, args.start_offset]
    
    if not any(index_args) and not any(time_args):
        print("Error: Must specify time range using either index or time arguments", file=sys.stderr)
        sys.exit(1)
    
    try:
        h5_time_slice(args.input, args.output, args)
        print("Time slicing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()