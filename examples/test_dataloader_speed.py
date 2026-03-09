#!/usr/bin/env python3
"""
Test script to find optimal num_workers for data loading.
Usage: python test_dataloader_speed.py --num_workers 0 2 4 8
"""

import argparse
import time
import yaml
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


def test_dataloader_speed(config_path: str, num_workers: int, num_batches: int = 50):
    """Test dataloader speed with given num_workers."""

    # Load config
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    # Override num_workers
    cfg.datasets.vla_data.num_workers = num_workers

    print(f"\n{'='*60}")
    print(f"Testing with num_workers={num_workers}")
    print(f"{'='*60}")

    # Create dataset
    print("Creating dataset...")
    vla_dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)

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
    )

    print(f"Batch size: {cfg.datasets.vla_data.per_device_batch_size}")
    print(f"Prefetch factor: {prefetch_factor}")

    # Warmup
    print("\nWarming up...")
    data_iter = iter(dataloader)
    for _ in range(5):
        _ = next(data_iter)

    # Measure speed
    print(f"\nMeasuring speed over {num_batches} batches...")
    times = []

    for i in range(num_batches):
        start = time.perf_counter()
        batch = next(data_iter)
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_time = sum(times[-10:]) / 10
            print(f"  Batch {i+1}/{num_batches}: {avg_time:.3f}s/batch")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n{'='*60}")
    print(f"Results for num_workers={num_workers}:")
    print(f"  Average time: {avg_time:.3f}s/batch")
    print(f"  Min time: {min_time:.3f}s/batch")
    print(f"  Max time: {max_time:.3f}s/batch")
    print(f"  Throughput: {cfg.datasets.vla_data.per_device_batch_size / avg_time:.1f} samples/sec")
    print(f"{'='*60}\n")

    return {
        'num_workers': num_workers,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'throughput': cfg.datasets.vla_data.per_device_batch_size / avg_time
    }


def main():
    parser = argparse.ArgumentParser(description='Test dataloader speed with different num_workers')
    parser.add_argument('--config', type=str,
                       default='./examples/calvin/train_files/starvla_train_calvin.yaml',
                       help='Path to config file')
    parser.add_argument('--num_workers', nargs='+', type=int, default=[0, 2, 4, 8],
                       help='List of num_workers to test')
    parser.add_argument('--num_batches', type=int, default=16,
                       help='Number of batches to test')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DataLoader Speed Test")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Testing num_workers: {args.num_workers}")
    print(f"Batches per test: {args.num_batches}")

    results = []
    for num_workers in args.num_workers:
        try:
            result = test_dataloader_speed(args.config, num_workers, args.num_batches)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error with num_workers={num_workers}: {e}")
            continue

    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'num_workers':<15} {'Avg Time':<15} {'Throughput':<20}")
        print("-" * 60)

        for r in results:
            print(f"{r['num_workers']:<15} {r['avg_time']:.3f}s/batch{'':<4} {r['throughput']:.1f} samples/sec")

        # Best result
        best = min(results, key=lambda x: x['avg_time'])
        print("\n" + "="*60)
        print(f"✅ Best configuration: num_workers={best['num_workers']}")
        print(f"   Average time: {best['avg_time']:.3f}s/batch")
        print(f"   Throughput: {best['throughput']:.1f} samples/sec")
        print("="*60)


if __name__ == '__main__':
    main()
