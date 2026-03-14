#!/usr/bin/env python3
"""
Test script to find optimal num_workers for data loading with 8-GPU distributed setup.
Usage: 
    torchrun --nproc_per_node=8 \
        examples/tools/test_dataloader_speed.py \
        --config examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
        --num_workers 8 16 24 32 \
        --num_batches 50
"""

import argparse
import time
import yaml
import os
import torch
import torch.distributed as dist
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
import psutil
import gc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_memory_info():
    """Get current CPU and GPU memory usage."""
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / 1024 / 1024

    # GPU memory
    gpu_mem_mb = 0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return cpu_mem_mb, gpu_mem_mb


def test_dataloader_speed(config_path: str, num_workers: int, num_batches: int = 50, rank: int = 0, world_size: int = 1):
    """Test dataloader speed with given num_workers in distributed setting."""

    # Load config
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    # Override num_workers
    cfg.datasets.vla_data.num_workers = num_workers

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Testing with num_workers={num_workers} on {world_size} GPUs")
        print(f"{'='*60}")

    # Create dataset
    if rank == 0:
        print("Creating dataset...")
    vla_dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)

    # Create distributed sampler
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            vla_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )

    # Create dataloader
    prefetch_factor = 2 if num_workers > 0 else None
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if not using sampler
    )

    if rank == 0:
        print(f"Batch size per GPU: {cfg.datasets.vla_data.per_device_batch_size}")
        print(f"Total batch size: {cfg.datasets.vla_data.per_device_batch_size * world_size}")
        print(f"Prefetch factor: {prefetch_factor}")

    # Record initial memory
    gc.collect()
    torch.cuda.empty_cache()
    initial_cpu_mem, initial_gpu_mem = get_memory_info()

    if rank == 0:
        print(f"\nInitial memory: CPU={initial_cpu_mem:.1f}MB, GPU={initial_gpu_mem:.1f}MB")

    # Warmup
    if rank == 0:
        print("\nWarming up...")
    data_iter = iter(dataloader)
    for _ in range(5):
        _ = next(data_iter)

    # Record memory after warmup
    gc.collect()
    warmup_cpu_mem, warmup_gpu_mem = get_memory_info()

    # Measure speed and memory
    if rank == 0:
        print(f"\nMeasuring speed over {num_batches} batches...")
    times = []
    memory_samples = []

    for i in range(num_batches):
        start = time.perf_counter()
        batch = next(data_iter)
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)

        # Sample memory every 10 batches
        if (i + 1) % 10 == 0:
            cpu_mem, gpu_mem = get_memory_info()
            memory_samples.append({
                'iteration': i + 1,
                'cpu_mb': cpu_mem,
                'gpu_mb': gpu_mem
            })

            if rank == 0:
                avg_time = sum(times[-10:]) / 10
                print(f"  Batch {i+1}/{num_batches}: {avg_time:.3f}s/batch | "
                      f"CPU={cpu_mem:.1f}MB (+{cpu_mem-initial_cpu_mem:.1f}MB) | "
                      f"GPU={gpu_mem:.1f}MB (+{gpu_mem-initial_gpu_mem:.1f}MB)")

    # Final memory check
    gc.collect()
    final_cpu_mem, final_gpu_mem = get_memory_info()

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Check for memory leak
    cpu_leak = final_cpu_mem - warmup_cpu_mem
    gpu_leak = final_gpu_mem - warmup_gpu_mem

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Results for num_workers={num_workers} on {world_size} GPUs:")
        print(f"  Average time: {avg_time:.3f}s/batch")
        print(f"  Min time: {min_time:.3f}s/batch")
        print(f"  Max time: {max_time:.3f}s/batch")
        print(f"  Throughput per GPU: {cfg.datasets.vla_data.per_device_batch_size / avg_time:.1f} samples/sec")
        print(f"  Total throughput: {cfg.datasets.vla_data.per_device_batch_size * world_size / avg_time:.1f} samples/sec")
        print(f"\n  Memory usage:")
        print(f"    Initial: CPU={initial_cpu_mem:.1f}MB, GPU={initial_gpu_mem:.1f}MB")
        print(f"    After warmup: CPU={warmup_cpu_mem:.1f}MB, GPU={warmup_gpu_mem:.1f}MB")
        print(f"    Final: CPU={final_cpu_mem:.1f}MB, GPU={final_gpu_mem:.1f}MB")
        print(f"    Memory leak: CPU={cpu_leak:+.1f}MB, GPU={gpu_leak:+.1f}MB")

        if cpu_leak > 100 or gpu_leak > 100:
            print(f"\n  ⚠️  WARNING: Potential memory leak detected!")
        else:
            print(f"\n  ✅ No significant memory leak detected")
        print(f"{'='*60}\n")

    return {
        'num_workers': num_workers,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'throughput_per_gpu': cfg.datasets.vla_data.per_device_batch_size / avg_time,
        'throughput_total': cfg.datasets.vla_data.per_device_batch_size * world_size / avg_time,
        'initial_cpu_mem': initial_cpu_mem,
        'final_cpu_mem': final_cpu_mem,
        'cpu_leak': cpu_leak,
        'initial_gpu_mem': initial_gpu_mem,
        'final_gpu_mem': final_gpu_mem,
        'gpu_leak': gpu_leak,
        'memory_samples': memory_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Test dataloader speed with different num_workers on 8 GPUs')
    parser.add_argument('--config', type=str,
                       default='./examples/calvin/train_files/starvla_train_calvin.yaml',
                       help='Path to config file')
    parser.add_argument('--num_workers', nargs='+', type=int, default=[0, 2, 4, 8, 16, 24, 32],
                       help='List of num_workers to test')
    parser.add_argument('--num_batches', type=int, default=50,
                       help='Number of batches to test')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("\n" + "="*60)
        print("DataLoader Speed Test - Distributed Mode")
        print("="*60)
        print(f"Number of GPUs: {world_size}")
        print(f"Config: {args.config}")
        print(f"Testing num_workers: {args.num_workers}")
        print(f"Batches per test: {args.num_batches}")

    results = []
    for num_workers in args.num_workers:
        try:
            result = test_dataloader_speed(args.config, num_workers, args.num_batches, rank, world_size)
            if rank == 0:
                results.append(result)

            # Synchronize between GPUs
            if world_size > 1:
                dist.barrier()

            # Cleanup between tests
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)  # Give system time to cleanup

        except Exception as e:
            if rank == 0:
                print(f"\n❌ Error with num_workers={num_workers}: {e}")
                import traceback
                traceback.print_exc()
            if world_size > 1:
                dist.barrier()
            continue

    # Summary
    if rank == 0 and results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'num_workers':<12} {'Avg Time':<15} {'Throughput':<20} {'CPU Leak':<15} {'GPU Leak':<15}")
        print("-" * 90)

        for r in results:
            leak_warning = ""
            if r['cpu_leak'] > 100 or r['gpu_leak'] > 100:
                leak_warning = " ⚠️"
            print(f"{r['num_workers']:<12} {r['avg_time']:.3f}s/batch{'':<4} "
                  f"{r['throughput_total']:.1f} samples/sec{'':<3} "
                  f"{r['cpu_leak']:+.1f}MB{'':<7} "
                  f"{r['gpu_leak']:+.1f}MB{leak_warning}")

        # Best result
        best = min(results, key=lambda x: x['avg_time'])
        print("\n" + "="*60)
        print(f"✅ Best configuration: num_workers={best['num_workers']}")
        print(f"   Average time: {best['avg_time']:.3f}s/batch")
        print(f"   Total throughput: {best['throughput_total']:.1f} samples/sec")
        print(f"   Memory leak: CPU={best['cpu_leak']:+.1f}MB, GPU={best['gpu_leak']:+.1f}MB")

        # Check for memory leaks
        leak_results = [r for r in results if r['cpu_leak'] > 100 or r['gpu_leak'] > 100]
        if leak_results:
            print("\n⚠️  Configurations with potential memory leaks:")
            for r in leak_results:
                print(f"   num_workers={r['num_workers']}: CPU={r['cpu_leak']:+.1f}MB, GPU={r['gpu_leak']:+.1f}MB")
        else:
            print("\n✅ No significant memory leaks detected in any configuration")
        print("="*60)

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
