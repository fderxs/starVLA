#!/usr/bin/env python3
"""
LeRobot Video Preprocessing Script - Aligned with LeRobot Dataset Structure

This script preprocesses ALL videos in a LeRobot dataset directory and stores them
in a structure that mirrors the original videos/data layout:

Original:
  dataset_name/
    ├── videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4
    ├── data/chunk-XXX/episode_XXXXXX.parquet
    └── meta/

Preprocessed:
  dataset_name/
    └── images/chunk-XXX/{video_key}/episode_XXXXXX.npy (or .npz)

Usage:
    # Preprocess all datasets in a directory
    python examples/tools/preprocess_lerobot_videos.py \
        --data_root /path/to/lerobot/datasets \
        --video_backend torchvision_av \
        --num_workers 8 \
        --compress

    # Preprocess specific datasets only
    python examples/tools/preprocess_lerobot_videos.py \
        --data_root /path/to/lerobot/datasets \
        --datasets libero_spatial_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot \
        --num_workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


def find_lerobot_datasets(data_root: Path) -> List[Path]:
    """Find all LeRobot datasets in the given directory."""
    datasets = []
    for item in data_root.iterdir():
        if item.is_dir():
            # Check if it's a valid LeRobot dataset
            if (item / "meta" / "info.json").exists():
                datasets.append(item)
    return sorted(datasets)


def load_dataset_info(dataset_path: Path) -> dict:
    """Load dataset info.json."""
    info_file = dataset_path / "meta" / "info.json"
    with open(info_file, 'r') as f:
        return json.load(f)


def get_video_keys(dataset_path: Path) -> List[str]:
    """Get all video keys from the dataset."""
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        return []

    video_keys = []
    for chunk_dir in sorted(videos_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith('chunk-'):
            for video_key_dir in chunk_dir.iterdir():
                if video_key_dir.is_dir():
                    video_key = video_key_dir.name
                    if video_key not in video_keys:
                        video_keys.append(video_key)

    return video_keys


def preprocess_single_video(args) -> Tuple[str, str]:
    """Preprocess a single video file."""
    episode_index, chunk_index, video_key, dataset_path, video_backend, compress = args

    try:
        # Construct paths
        video_path = dataset_path / "videos" / f"chunk-{chunk_index:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
        data_path = dataset_path / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"

        if not video_path.exists():
            return f"video_{episode_index:06d}_{video_key}", f"Video not found: {video_path}"

        if not data_path.exists():
            return f"video_{episode_index:06d}_{video_key}", f"Data not found: {data_path}"

        # Load episode data to get timestamps
        episode_data = pd.read_parquet(data_path)
        timestamps = episode_data['timestamp'].to_numpy()

        # Decode all frames
        frames = get_frames_by_timestamps(
            str(video_path),
            timestamps,
            video_backend=video_backend,
            video_backend_kwargs={}
        )

        # Save to images directory (mirrors videos structure)
        output_path = dataset_path / "images" / f"chunk-{chunk_index:03d}" / video_key / f"episode_{episode_index:06d}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            np.savez_compressed(str(output_path).replace('.npy', '.npz'), frames=frames)
        else:
            np.save(output_path, frames)

        return f"video_{episode_index:06d}_{video_key}", None

    except Exception as e:
        return f"video_{episode_index:06d}_{video_key}", str(e)


def preprocess_dataset(dataset_path: Path, video_backend: str, num_workers: int, compress: bool):
    """Preprocess all videos in a single dataset."""
    dataset_name = dataset_path.name

    print(f"\n{'='*80}")
    print(f"Preprocessing dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*80}\n")

    # Load dataset info
    try:
        info = load_dataset_info(dataset_path)
    except Exception as e:
        print(f"❌ Failed to load dataset info: {e}")
        return

    total_episodes = info.get('total_episodes', 0)
    chunks_size = info.get('chunks_size', 1000)
    total_chunks = info.get('total_chunks', 1)

    print(f"Total episodes: {total_episodes}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunk size: {chunks_size}")

    # Get video keys
    video_keys = get_video_keys(dataset_path)
    if not video_keys:
        print(f"❌ No video keys found in {dataset_name}")
        return

    print(f"Video keys: {video_keys}")
    print(f"Using {num_workers} workers")
    print(f"Compression: {'enabled' if compress else 'disabled'}")

    # Prepare tasks (episode_index, chunk_index, video_key, ...)
    tasks = []
    for episode_index in range(total_episodes):
        chunk_index = episode_index // chunks_size
        for video_key in video_keys:
            tasks.append((episode_index, chunk_index, video_key, dataset_path, video_backend, compress))

    print(f"\nTotal tasks: {len(tasks)} ({total_episodes} episodes × {len(video_keys)} video keys)")

    # Process with multiprocessing
    start_time = time.time()
    failed_tasks = []

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(preprocess_single_video, tasks),
                total=len(tasks),
                desc=f"Preprocessing {dataset_name}"
            ))
    else:
        results = []
        for task in tqdm(tasks, desc=f"Preprocessing {dataset_name}"):
            results.append(preprocess_single_video(task))

    # Check for failures
    for task_id, error in results:
        if error:
            failed_tasks.append((task_id, error))

    elapsed_time = time.time() - start_time

    # Save preprocessing metadata
    metadata = {
        'dataset_name': dataset_name,
        'total_episodes': total_episodes,
        'video_keys': video_keys,
        'total_chunks': total_chunks,
        'chunks_size': chunks_size,
        'video_backend': video_backend,
        'compressed': compress,
        'preprocessing_time_seconds': elapsed_time,
        'total_tasks': len(tasks),
        'failed_tasks': len(failed_tasks),
        'success_rate': (len(tasks) - len(failed_tasks)) / len(tasks) if tasks else 0
    }

    metadata_path = dataset_path / "images" / "preprocessing_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Preprocessing completed for {dataset_name}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Successful: {len(tasks) - len(failed_tasks)}")
    print(f"  Failed: {len(failed_tasks)}")
    print(f"  Time elapsed: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
    if tasks:
        print(f"  Average: {elapsed_time/len(tasks):.3f}s per task")
    if failed_tasks:
        print(f"\n  ⚠️  Failed tasks: {len(failed_tasks)}")
        failed_log = dataset_path / "images" / "failed_tasks.txt"
        with open(failed_log, 'w') as f:
            for task_id, error in failed_tasks:
                f.write(f"{task_id}: {error}\n")
        print(f"  Failed tasks logged to: {failed_log}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess LeRobot videos to numpy arrays')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing LeRobot datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to process (default: all found datasets)')
    parser.add_argument('--video_backend', type=str, default='torchvision_av',
                       choices=['torchvision_av', 'decord', 'pyav', 'opencv'],
                       help='Video backend to use for decoding')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers for preprocessing')
    parser.add_argument('--compress', action='store_true',
                       help='Use npz compression to save storage (slower loading)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Just print what would be done without actually processing')

    args = parser.parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"Scanning for LeRobot datasets in: {data_root}")
    all_datasets = find_lerobot_datasets(data_root)

    if not all_datasets:
        print(f"❌ No LeRobot datasets found in {data_root}")
        return

    print(f"Found {len(all_datasets)} datasets:")
    for ds in all_datasets:
        print(f"  - {ds.name}")

    # Filter datasets if specified
    if args.datasets:
        datasets_to_process = [ds for ds in all_datasets if ds.name in args.datasets]
        if not datasets_to_process:
            print(f"❌ None of the specified datasets found: {args.datasets}")
            return
        print(f"\nProcessing {len(datasets_to_process)} specified datasets:")
        for ds in datasets_to_process:
            print(f"  - {ds.name}")
    else:
        datasets_to_process = all_datasets

    print(f"\nVideo backend: {args.video_backend}")
    print(f"Workers: {args.num_workers}")
    print(f"Compression: {args.compress}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No actual processing will be done")
        for dataset_path in datasets_to_process:
            info = load_dataset_info(dataset_path)
            video_keys = get_video_keys(dataset_path)
            print(f"\n{dataset_path.name}:")
            print(f"  Episodes: {info.get('total_episodes', 0)}")
            print(f"  Video keys: {video_keys}")
            print(f"  Total tasks: {info.get('total_episodes', 0) * len(video_keys)}")
        return

    # Confirm before proceeding
    response = input(f"\nProceed with preprocessing {len(datasets_to_process)} datasets? [y/N]: ")
    if response.lower() != 'y':
        print("Preprocessing cancelled.")
        return

    # Process each dataset
    total_start = time.time()
    for dataset_path in datasets_to_process:
        preprocess_dataset(
            dataset_path,
            args.video_backend,
            args.num_workers,
            args.compress
        )

    total_elapsed = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"ALL PREPROCESSING COMPLETED")
    print(f"Total datasets processed: {len(datasets_to_process)}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)")
    print(f"{'='*80}\n")

    print("✅ Preprocessed frames are stored in: dataset_name/images/")
    print("   Structure: images/chunk-XXX/{video_key}/episode_XXXXXX.npy")
    print("\nNext steps:")
    print("1. Update your training config:")
    print("   datasets.vla_data.use_preprocessed_frames: true")
    print("2. Run training with significantly faster data loading!")


if __name__ == '__main__':
    main()
