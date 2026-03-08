"""
Convert CALVIN dataset to LeRobot format.

This script supports reading from either a zip file or an extracted folder.
The input type is automatically detected.

Usage:
# From zip file:
python3 examples/calvin/train_files/convert_to_lerobot.py --path /path/to/task_D_D.zip

# From extracted folder:
python3 examples/calvin/train_files/convert_to_lerobot.py --path /path/to/task_D_D

# Convert without merging (keep temporary datasets for debugging):
python3 examples/calvin/train_files/convert_to_lerobot.py --path /path/to/task_D_D --skip-merge

# Merge existing temporary datasets only:
python3 examples/calvin/train_files/convert_to_lerobot.py --merge-only --repo-id calvin_abc_d_1.0.0_lerobot

# Push to Hugging Face Hub:
python3 examples/calvin/train_files/convert_to_lerobot.py --path /path/to/task_D_D --push-to-hub

The resulting dataset will get saved to `$LEROBOT_HOME/<repo_id>`.
"""

from __future__ import annotations
import debugpy
import os
import shutil
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from multiprocessing import Pool, cpu_count, Manager
import json

import tyro
import numpy as np
from tqdm import tqdm
# from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_HOME = Path("/mnt/volumes/base-3da-ali-sh-mix/xswang/object/datasets/lerobot")

@dataclass(frozen=True)
class Args:
    path: str | None = None  # Path to zip file or extracted folder (auto-detected), not needed for merge_only
    repo_id: str = "calvin_abc_d_1.0.0_lerobot"
    fps: int = 10
    splits: Literal["training", "validation", "both"] = "training"
    action_key: Literal["rel_actions", "actions"] = "rel_actions"
    max_episodes: int | None = None  # for debugging
    push_to_hub: bool = False
    num_workers: int = 16  # Number of parallel workers for processing episodes
    merge_only: bool = False  # If True, only perform merge operation on existing temp datasets
    skip_merge: bool = False  # If True, skip merge step (keep temp datasets for debugging)

@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr, including C library output."""
    # Save the original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save copies of the original file descriptors
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Save original Python stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect Python's sys.stdout and sys.stderr
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect file descriptors to devnull (this catches C library output)
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)

        # Create new file objects for Python
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        yield

    finally:
        # Restore Python stdout/stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Restore original file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)

        # Close the duplicates and devnull
        os.close(stdout_dup)
        os.close(stderr_dup)
        os.close(devnull_fd)

# ===== Functions for reading from ZIP file =====
def _load_npy_from_zip(z: zipfile.ZipFile, name: str) -> np.ndarray:
    # `np.load` supports file-like objects; ZipExtFile works, and avoids extra copies.
    with z.open(name, "r") as f:
        return np.load(f, allow_pickle=True)


def _iter_episode_lang_from_zip(
    z: zipfile.ZipFile, split: Literal["training", "validation"]
) -> list[tuple[int, int]]:
    base = f"task_D_D/{split}"
    lang_data = _load_npy_from_zip(z, f"{base}/lang_annotations/auto_lang_ann.npy").item()
    # shape: (num_eps, 2), inclusive bounds
    return lang_data


def _load_step_npz_from_zip(
    z: zipfile.ZipFile, split: Literal["training", "validation"], step_id: int
) -> dict[str, np.ndarray]:
    name = f"task_D_D/{split}/episode_{step_id:07d}.npz"
    with z.open(name, "r") as f:
        npz = np.load(f, allow_pickle=True)
        try:
            return {k: npz[k] for k in npz.files}
        finally:
            npz.close()


# ===== Functions for reading from extracted folder =====
def _load_npy_from_folder(folder_path: Path, name: str) -> np.ndarray:
    """Load a .npy file from the folder."""
    file_path = folder_path / name
    return np.load(file_path, allow_pickle=True)


def _iter_episode_lang(
    data_path: Path, split: Literal["training", "validation"]
) -> dict:
    """Load language annotations from the folder."""
    lang_file = data_path / split / "lang_annotations" / "auto_lang_ann.npy"
    lang_data = np.load(lang_file, allow_pickle=True).item()
    return lang_data


def _load_step_npz_from_folder(
    data_path: Path, split: Literal["training", "validation"], step_id: int
) -> dict[str, np.ndarray]:
    """Load a step NPZ file from the folder."""
    npz_path = data_path / split / f"episode_{step_id:07d}.npz"
    npz = np.load(npz_path, allow_pickle=True)
    try:
        return {k: npz[k] for k in npz.files}
    finally:
        npz.close()

def process_episode_batch(args_tuple):
    """Process a batch of episodes in a separate process."""
    (worker_id, data_source, split, episode_indices, ep_start_end_ids,
     lang_ann, fps, action_key, temp_dir, progress_dict, is_zip) = args_tuple

    # Create a temporary dataset for this worker
    worker_repo_id = f"temp_worker_{worker_id}"
    worker_output_path = Path(temp_dir) / worker_repo_id

    # Suppress all output in worker processes
    with suppress_output():
        dataset = LeRobotDataset.create(
            root=worker_output_path,
            repo_id=worker_repo_id,
            robot_type="franka",
            fps=fps,
            use_videos=True,
            features={
                "observation.images": {
                    "dtype": "video",
                    "shape": (200, 200, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": (84, 84, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (15,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                }
            },
            image_writer_threads=4,
            image_writer_processes=0,
        )

        # Open zip file if needed
        zipfile_obj = zipfile.ZipFile(data_source, "r") if is_zip else None

        try:
            # Process assigned episodes
            for ep_idx in episode_indices:
                start_idx, end_idx = ep_start_end_ids[ep_idx]
                task = lang_ann[ep_idx]

                for idx in range(start_idx, end_idx + 1):
                    # Load step based on data source type
                    if is_zip:
                        step = _load_step_npz_from_zip(zipfile_obj, split, idx)
                    else:
                        step = _load_step_npz_from_folder(Path(data_source), split, idx)

                    dataset.add_frame(
                        {
                            "observation.images": step["rgb_static"],
                            "observation.images.wrist": step["rgb_gripper"],
                            "observation.state": step["robot_obs"].astype(np.float32),
                            "action": step[action_key].astype(np.float32),
                        },
                        task=task,
                    )

                dataset.save_episode()

                # Update shared progress counter
                with progress_dict['lock']:
                    progress_dict['completed'] += 1
        finally:
            if zipfile_obj:
                zipfile_obj.close()

    return worker_id, str(worker_output_path), len(episode_indices)


def merge_datasets(temp_dirs: list[str], output_path: Path, repo_id: str):
    """Merge multiple temporary datasets into one final dataset.

    Args:
        temp_dirs: List of temporary dataset directories
        output_path: Final output path
        repo_id: Repository ID
    """
    print(f"\nMerging {len(temp_dirs)} temporary datasets...")

    # Create final dataset directory structure
    final_data_dir = output_path / "data"
    final_videos_dir = output_path / "videos"
    final_meta_dir = output_path / "meta"

    final_data_dir.mkdir(parents=True, exist_ok=True)
    final_videos_dir.mkdir(parents=True, exist_ok=True)
    final_meta_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []
    all_episodes_stats = []
    all_tasks = []
    info_json = None
    episode_offset = 0

    # Merge each temporary dataset
    for temp_dir in tqdm(temp_dirs, desc="Merging datasets"):
        temp_path = Path(temp_dir)

        # Load metadata
        with open(temp_path / "meta" / "episodes.jsonl", "r") as f:
            episodes = [json.loads(line) for line in f]

        # Load episodes_stats.jsonl
        with open(temp_path / "meta" / "episodes_stats.jsonl", "r") as f:
            episodes_stats = [json.loads(line) for line in f]

        # Load tasks.jsonl
        with open(temp_path / "meta" / "tasks.jsonl", "r") as f:
            tasks = [json.loads(line) for line in f]

        # Save info.json from first dataset
        if info_json is None and (temp_path / "meta" / "info.json").exists():
            with open(temp_path / "meta" / "info.json", "r") as f:
                info_json = json.load(f)

        # Copy data/videos/episodes
        # Note: tasks may have fewer entries than episodes due to deduplication
        assert len(episodes) == len(episodes_stats), f"len(episodes) = {len(episodes)}, len(episodes_stats) = {len(episodes_stats)}"

        for old_idx, (ep, ep_stats) in enumerate(zip(episodes, episodes_stats)):
            new_idx = episode_offset + old_idx

            # Calculate chunk folder based on index (0-999 -> chunk-000, 1000-1999 -> chunk-001, etc.)
            chunk_new_id = new_idx // 1000
            chunk_new_name = f"chunk-{chunk_new_id:03d}"

            chunk_old_id = old_idx // 1000
            chunk_old_name = f"chunk-{chunk_old_id:03d}"

            # Copy data file with renamed index to appropriate chunk
            old_data_file = temp_path / "data" / f"{chunk_old_name}/episode_{old_idx:06d}.parquet"
            new_data_file = final_data_dir / f"{chunk_new_name}/episode_{new_idx:06d}.parquet"
            new_data_file.parent.mkdir(parents=True, exist_ok=True)
            if old_data_file.exists():
                shutil.copy2(old_data_file, new_data_file)

            # Copy video files with renamed index to appropriate chunk
            for video_key in ["observation.images", "observation.images.wrist"]:
                old_video = temp_path / "videos" / f"{chunk_old_name}/{video_key}/episode_{old_idx:06d}.mp4"
                new_video = final_videos_dir / f"{chunk_new_name}/{video_key}/episode_{new_idx:06d}.mp4"
                new_video.parent.mkdir(parents=True, exist_ok=True)
                if old_video.exists():
                    shutil.copy2(old_video, new_video)

            ep["episode_index"] = new_idx
            ep_stats["episode_index"] = new_idx
            all_episodes.append(ep)
            all_episodes_stats.append(ep_stats)

        # Collect unique tasks and add them to the global task list
        for task in tasks:
            # Check if this task already exists in all_tasks
            task_text = task["task"]
            existing_task = next((t for t in all_tasks if t["task"] == task_text), None)
            if existing_task is None:
                # New unique task, add it with new task_index
                new_task = {"task_index": len(all_tasks), "task": task_text}
                all_tasks.append(new_task)

        episode_offset += len(episodes)
        # Clean up temp directory
        shutil.rmtree(temp_path)

    # Sort episodes by index to maintain order
    all_episodes.sort(key=lambda x: x["episode_index"])
    all_episodes_stats.sort(key=lambda x: x["episode_index"])

    # Write merged metadata
    with open(final_meta_dir / "episodes.jsonl", "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    # Write episodes_stats.jsonl
    with open(final_meta_dir / "episodes_stats.jsonl", "w") as f:
        for ep_stats in all_episodes_stats:
            f.write(json.dumps(ep_stats) + "\n")

    # Write tasks
    with open(final_meta_dir / "tasks.jsonl", "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")

    # Write info.json
    if info_json:
        # Update total episodes count and chunks information
        total_episodes = len(all_episodes)
        total_chunks = (total_episodes + 999) // 1000  # Round up to get number of chunks

        info_json["total_episodes"] = total_episodes
        info_json["total_chunks"] = total_chunks
        info_json["chunks_size"] = 1000
        info_json["splits"]["train"] = f"0:{total_episodes}"

        with open(final_meta_dir / "info.json", "w") as f:
            json.dump(info_json, f, indent=2)

    print(f"Merged {len(all_episodes)} episodes into {output_path}")


def process_split_parallel(data_source: str, split: str, lang_data: dict, args: Args,
                           output_path: Path, temp_base_dir: Path, is_zip: bool):
    """Process a split with parallel workers.

    Args:
        data_source: Path to zip file or folder
        split: "training" or "validation"
        lang_data: Language annotations data
        args: Command line arguments
        output_path: Final output path
        temp_base_dir: Temporary directory for workers
        is_zip: True if data_source is a zip file
    """
    ep_start_end_ids = lang_data["info"]["indx"]
    lang_ann = lang_data["language"]["ann"]

    total_episodes = len(ep_start_end_ids)
    if args.max_episodes:
        total_episodes = min(total_episodes, args.max_episodes)
        ep_start_end_ids = ep_start_end_ids[:total_episodes]
        lang_ann = lang_ann[:total_episodes]

    # Create shared progress tracking
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['completed'] = 0
    progress_dict['lock'] = manager.Lock()

    # Split episodes into batches for workers
    episodes_per_worker = (total_episodes + args.num_workers - 1) // args.num_workers
    worker_args = []

    for worker_id in range(args.num_workers):
        start_ep = worker_id * episodes_per_worker
        end_ep = min(start_ep + episodes_per_worker, total_episodes)

        if start_ep >= total_episodes:
            break

        episode_indices = list(range(start_ep, end_ep))
        worker_args.append((
            worker_id,
            data_source,
            split,
            episode_indices,
            ep_start_end_ids,
            lang_ann,
            args.fps,
            args.action_key,
            str(temp_base_dir),
            progress_dict,
            is_zip
        ))

    print(f"Processing {total_episodes} episodes with {len(worker_args)} workers...")

    # Process in parallel with episode-level progress bar
    with Pool(processes=args.num_workers) as pool:
        with tqdm(total=total_episodes, desc=f"Processing {split} episodes", unit="episode") as pbar:
            last_completed = 0
            async_result = pool.map_async(process_episode_batch, worker_args)

            # Monitor progress
            while not async_result.ready():
                current_completed = progress_dict['completed']
                if current_completed > last_completed:
                    pbar.update(current_completed - last_completed)
                    last_completed = current_completed
                async_result.wait(timeout=0.1)

            # Final update
            current_completed = progress_dict['completed']
            if current_completed > last_completed:
                pbar.update(current_completed - last_completed)

            results = async_result.get()
            temp_dirs = [result[1] for result in results]

    # Merge all temporary datasets (unless skip_merge is set)
    if not args.skip_merge:
        merge_datasets(temp_dirs, output_path, args.repo_id)
    else:
        print(f"\nSkipping merge. Temporary datasets saved in: {temp_base_dir}")
        return temp_dirs


def main(args: Args):
    output_path = LEROBOT_HOME / args.repo_id
    temp_base_dir = output_path.parent / f"temp_{args.repo_id}"

    # Merge-only mode: merge existing temporary datasets
    if args.merge_only:
        print(f"Merge-only mode: merging temporary datasets from {temp_base_dir}")
        if not temp_base_dir.exists():
            raise ValueError(f"Temporary dataset directory not found: {temp_base_dir}")

        # Find all temp_worker_* directories
        temp_dirs = sorted([str(d) for d in temp_base_dir.iterdir() if d.is_dir() and d.name.startswith("temp_worker_")])
        if not temp_dirs:
            raise ValueError(f"No temporary worker directories found in {temp_base_dir}")

        print(f"Found {len(temp_dirs)} temporary datasets to merge")

        # Clean up any existing final dataset
        if output_path.exists():
            shutil.rmtree(output_path)

        # Merge datasets
        merge_datasets(temp_dirs, output_path, args.repo_id)

        # Clean up temp base directory after merge
        shutil.rmtree(temp_base_dir)
        print(f"\nMerge completed. Dataset saved to: {output_path}")
        return

    # Convert mode: process data and optionally merge
    if args.path is None:
        raise ValueError("--path is required when not using --merge-only")

    # Auto-detect if path is a zip file or folder
    path_obj = Path(args.path)
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {args.path}")

    is_zip = path_obj.is_file() and (path_obj.suffix == '.zip' or zipfile.is_zipfile(args.path))
    data_source = args.path

    # Clean up any existing dataset in the output directory
    if output_path.exists():
        shutil.rmtree(output_path)

    splits: list[Literal["training", "validation"]]
    if args.splits == "both":
        splits = ["training", "validation"]
    else:
        splits = [args.splits]

    # Setup
    print(f"Reading from {'zip file' if is_zip else 'folder'}: {data_source}")
    print(f"Using {args.num_workers} parallel workers")

    temp_base_dir = output_path.parent / f"temp_{args.repo_id}"
    temp_base_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    if is_zip:
        with zipfile.ZipFile(data_source, "r") as z:
            for split in splits:
                print(f"\nProcessing {split} split...")
                lang_data = _iter_episode_lang_from_zip(z, split)
                process_split_parallel(data_source, split, lang_data, args,
                                     output_path, temp_base_dir, is_zip)
    else:
        data_path = Path(data_source)
        for split in splits:
            print(f"\nProcessing {split} split...")
            lang_data = _iter_episode_lang(data_path, split)
            process_split_parallel(data_source, split, lang_data, args,
                                 output_path, temp_base_dir, is_zip)

    # Clean up temp base directory (only if merge was performed)
    if not args.skip_merge and temp_base_dir.exists():
        shutil.rmtree(temp_base_dir)

    # Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        print("\nLoading merged dataset for pushing to hub...")
        final_dataset = LeRobotDataset(repo_id=args.repo_id, root=output_path)
        final_dataset.push_to_hub(
            tags=["calvin", "task_D_D"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    main(tyro.cli(Args))