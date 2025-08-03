#!/usr/bin/env python3
"""
Memory benchmark for distributed xLSTM architecture.

This script tests memory usage of the distributed Social-xLSTM system
with varying numbers of VDs to ensure scalability and identify memory limits.

Usage:
    python scripts/benchmark_distributed_memory.py --max_vds 20 --batch_size 4
"""

import torch
import torch.nn as nn
import argparse
import time
import psutil
import os
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.append('/home/GP/repo/Social-xLSTM/src')

from social_xlstm.models.xlstm import TrafficXLSTMConfig
from social_xlstm.models.vd_xlstm_manager import VDXLSTMManager
from social_xlstm.models.distributed_social_xlstm import DistributedSocialXLSTMModel
from social_xlstm.interfaces.tensor_spec import TensorSpec
from collections import OrderedDict


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def get_gpu_memory() -> Dict[str, float]:
    """Get GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return {'allocated_mb': 0, 'cached_mb': 0}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
    }


def create_synthetic_batch(
    vd_ids: List[str],
    batch_size: int = 4,
    seq_length: int = 12,
    num_features: int = 3
) -> Dict[str, torch.Tensor]:
    """Create synthetic batch for testing."""
    batch = OrderedDict()
    
    for vd_id in vd_ids:
        # Create random tensor [B, T, F]
        tensor = torch.randn(batch_size, seq_length, num_features)
        batch[vd_id] = tensor
    
    return batch


def benchmark_vd_manager(num_vds: int, batch_size: int = 4) -> Dict[str, float]:
    """Benchmark VDXLSTMManager with specific number of VDs."""
    logger.info(f"Benchmarking VDXLSTMManager with {num_vds} VDs, batch_size={batch_size}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Record initial memory
    initial_mem = get_memory_usage()
    initial_gpu = get_gpu_memory()
    
    # Create xLSTM config
    xlstm_config = TrafficXLSTMConfig(
        input_size=3,
        hidden_size=64,  # Smaller for testing
        num_layers=2,    # Fewer layers for testing
        sequence_length=12,
        prediction_length=3
    )
    
    # Create VD manager
    vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]
    manager = VDXLSTMManager(
        xlstm_config=xlstm_config,
        vd_ids=vd_ids,
        lazy_init=False,  # Initialize all upfront for accurate measurement
        enable_gradient_checkpointing=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manager = manager.to(device)
    
    # Record memory after initialization
    init_mem = get_memory_usage()
    init_gpu = get_gpu_memory()
    
    # Create synthetic batch
    batch = create_synthetic_batch(vd_ids, batch_size)
    batch = {vd_id: tensor.to(device) for vd_id, tensor in batch.items()}
    
    # Forward pass timing
    start_time = time.time()
    
    try:
        with torch.no_grad():  # No gradients for memory testing
            outputs = manager(batch)
        
        forward_time = time.time() - start_time
        success = True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM with {num_vds} VDs: {e}")
            success = False
            forward_time = float('inf')
        else:
            raise
    
    # Record final memory
    final_mem = get_memory_usage()
    final_gpu = get_gpu_memory()
    
    # Calculate memory usage
    memory_stats = {
        'num_vds': num_vds,
        'batch_size': batch_size,
        'success': success,
        'forward_time_ms': forward_time * 1000,
        'initial_cpu_mb': initial_mem['rss_mb'],
        'final_cpu_mb': final_mem['rss_mb'],
        'cpu_increase_mb': final_mem['rss_mb'] - initial_mem['rss_mb'],
        'init_cpu_mb': init_mem['rss_mb'] - initial_mem['rss_mb'],
        'initial_gpu_mb': initial_gpu['allocated_mb'],
        'final_gpu_mb': final_gpu['allocated_mb'],
        'gpu_increase_mb': final_gpu['allocated_mb'] - initial_gpu['allocated_mb'],
        'max_gpu_mb': final_gpu.get('max_allocated_mb', 0),
        'total_parameters': manager.get_memory_usage()['total_parameters']
    }
    
    # Cleanup
    del manager, batch
    if success and 'outputs' in locals():
        del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return memory_stats


def benchmark_full_model(num_vds: int, batch_size: int = 4) -> Dict[str, float]:
    """Benchmark full DistributedSocialXLSTMModel."""
    logger.info(f"Benchmarking full model with {num_vds} VDs, batch_size={batch_size}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    initial_mem = get_memory_usage()
    initial_gpu = get_gpu_memory()
    
    # Create model config
    xlstm_config = TrafficXLSTMConfig(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        sequence_length=12,
        prediction_length=3
    )
    
    # Create full model
    model = DistributedSocialXLSTMModel(
        xlstm_config=xlstm_config,
        num_features=3,
        hidden_dim=64,
        prediction_length=3,
        social_pool_type="mean",
        enable_gradient_checkpointing=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create synthetic batch
    vd_ids = [f"VD_{i:03d}" for i in range(num_vds)]
    batch = create_synthetic_batch(vd_ids, batch_size)
    batch = {vd_id: tensor.to(device) for vd_id, tensor in batch.items()}
    
    init_mem = get_memory_usage()
    init_gpu = get_gpu_memory()
    
    # Forward pass
    start_time = time.time()
    
    try:
        with torch.no_grad():
            outputs = model(batch)
        
        forward_time = time.time() - start_time
        success = True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"Full model OOM with {num_vds} VDs: {e}")
            success = False
            forward_time = float('inf')
        else:
            raise
    
    final_mem = get_memory_usage()
    final_gpu = get_gpu_memory()
    
    memory_stats = {
        'num_vds': num_vds,
        'batch_size': batch_size,
        'success': success,
        'forward_time_ms': forward_time * 1000,
        'cpu_increase_mb': final_mem['rss_mb'] - initial_mem['rss_mb'],
        'gpu_increase_mb': final_gpu['allocated_mb'] - initial_gpu['allocated_mb'],
        'max_gpu_mb': final_gpu.get('max_allocated_mb', 0),
        'model_info': model.get_model_info() if success else None
    }
    
    # Cleanup
    del model, batch
    if success and 'outputs' in locals():
        del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return memory_stats


def run_scaling_benchmark(max_vds: int = 20, batch_size: int = 4) -> None:
    """Run complete scaling benchmark."""
    logger.info(f"Starting scaling benchmark: max_vds={max_vds}, batch_size={batch_size}")
    
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info(f"Using device: {device_name}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    
    # Test VD counts
    vd_counts = [1, 2, 5, 10, 15, 20] if max_vds >= 20 else [1, 2, 5, max_vds]
    vd_counts = [n for n in vd_counts if n <= max_vds]
    
    print(f"\\n{'='*80}")
    print(f"MEMORY BENCHMARK RESULTS - {device_name}")
    print(f"{'='*80}")
    print(f"{'VDs':<4} {'Success':<8} {'Time(ms)':<10} {'CPU(MB)':<10} {'GPU(MB)':<10} {'Params':<12}")
    print(f"{'-'*80}")
    
    results = []
    
    for num_vds in vd_counts:
        # Test VD Manager
        stats = benchmark_vd_manager(num_vds, batch_size)
        results.append(stats)
        
        success_str = "✓" if stats['success'] else "✗ OOM"
        time_str = f"{stats['forward_time_ms']:.1f}" if stats['success'] else "N/A"
        cpu_str = f"{stats['cpu_increase_mb']:.1f}"
        gpu_str = f"{stats['gpu_increase_mb']:.1f}"
        params_str = f"{stats['total_parameters']:,}"
        
        print(f"{num_vds:<4} {success_str:<8} {time_str:<10} {cpu_str:<10} {gpu_str:<10} {params_str:<12}")
        
        # Break if OOM
        if not stats['success']:
            logger.warning(f"OOM reached at {num_vds} VDs")
            break
    
    print(f"{'-'*80}")
    
    # Find memory scaling pattern
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        # Estimate memory per VD
        mem_per_vd = []
        for i in range(1, len(successful_results)):
            curr = successful_results[i]
            prev = successful_results[i-1]
            vd_diff = curr['num_vds'] - prev['num_vds']
            gpu_diff = curr['gpu_increase_mb'] - prev['gpu_increase_mb']
            if vd_diff > 0:
                mem_per_vd.append(gpu_diff / vd_diff)
        
        if mem_per_vd:
            avg_mem_per_vd = sum(mem_per_vd) / len(mem_per_vd)
            print(f"\\nEstimated memory per VD: {avg_mem_per_vd:.2f} MB")
            
            # Estimate maximum VDs for current GPU
            if torch.cuda.is_available():
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                estimated_max_vds = int(gpu_total_mb * 0.8 / avg_mem_per_vd)  # 80% utilization
                print(f"Estimated max VDs for this GPU: {estimated_max_vds}")
    
    print(f"\\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed xLSTM memory usage")
    parser.add_argument('--max_vds', type=int, default=20, help='Maximum number of VDs to test')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--test_full_model', action='store_true', help='Also test full model')
    
    args = parser.parse_args()
    
    # Run VD Manager benchmark
    run_scaling_benchmark(args.max_vds, args.batch_size)
    
    # Optionally test full model
    if args.test_full_model:
        print(f"\\n{'='*40}")
        print("FULL MODEL BENCHMARK")
        print(f"{'='*40}")
        
        for num_vds in [1, 2, 5]:
            if num_vds <= args.max_vds:
                stats = benchmark_full_model(num_vds, args.batch_size)
                success_str = "✓" if stats['success'] else "✗ OOM"
                time_str = f"{stats['forward_time_ms']:.1f}" if stats['success'] else "N/A"
                print(f"VDs: {num_vds}, Success: {success_str}, Time: {time_str} ms")


if __name__ == "__main__":
    main()