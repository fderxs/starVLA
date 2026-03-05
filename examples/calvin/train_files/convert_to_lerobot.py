"""
Convert CALVIN dataset to LeRobot format.

This script supports reading from either:
1. A zip file (e.g. `task_D_D.zip`)
2. An extracted folder (e.g. `task_D_D/`)

Usage:
# From zip file:
python3 examples/calvin/convert_to_lerobot.py --zip-path /path/to/task_D_D.zip

# From extracted folder:
python3 examples/calvin/convert_to_lerobot.py --data-path /path/to/task_D_D

If you want to push your dataset to the Hugging Face Hub:
python3 examples/calvin/convert_to_lerobot.py --data-path /path/to/task_D_D --push-to-hub

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

import tyro
import numpy as np
from tqdm import tqdm
# from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_HOME = Path("/mnt/volumes/base-3da-ali-sh-mix/xswang/object/datasets/lerobot")


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


@dataclass(frozen=True)
class Args:
    zip_path: str | None = None  # Path to the zip file (e.g., task_D_D.zip)
    data_path: str | None = None  # Path to the extracted folder (e.g., task_D_D/)
    repo_id: str = "calvin_abc_d_1.0.0_lerobot"
    fps: int = 10
    splits: Literal["training", "validation", "both"] = "training"
    action_key: Literal["rel_actions", "actions"] = "rel_actions"
    max_episodes: int | None = None  # for debugging
    push_to_hub: bool = False


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


def main(args: Args):
    # Validate that exactly one input method is provided
    if args.zip_path is None and args.data_path is None:
        raise ValueError("Please provide either --zip-path or --data-path")
    if args.zip_path is not None and args.data_path is not None:
        raise ValueError("Please provide only one of --zip-path or --data-path, not both")

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        root=output_path,
        repo_id=args.repo_id,
        robot_type="franka",
        fps=args.fps,
        use_videos=True,  # Save images as videos instead of individual PNG files
        features={
            "observation.images": {
                "dtype": "video",  # Changed from "image" to "video"
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "video",  # Changed from "image" to "video"
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
        image_writer_threads=10,
        image_writer_processes=5,
    )

    splits: list[Literal["training", "validation"]]
    if args.splits == "both":
        splits = ["training", "validation"]
    else:
        splits = [args.splits]

    # Process data from zip or folder
    if args.zip_path is not None:
        print(f"Reading from zip file: {args.zip_path}")
        with zipfile.ZipFile(args.zip_path, "r") as z:
            for split in splits:
                lang_data = _iter_episode_lang_from_zip(z, split)
                ep_start_end_ids = lang_data["info"]["indx"]
                lang_ann = lang_data["language"]["ann"]
                lang_task = lang_data["language"]["task"]

                # Add progress bar for episode processing
                total_episodes = len(ep_start_end_ids)
                pbar = tqdm(enumerate(ep_start_end_ids),
                           total=total_episodes,
                           desc=f"Processing {split} episodes",
                           unit="episode")

                for i, (start_idx, end_idx) in pbar:
                    task = lang_ann[i]
                    num_frames = end_idx - start_idx + 1
                    pbar.set_postfix({"frames": num_frames, "episode": f"{i+1}/{total_episodes}"})

                    for idx in range(start_idx, end_idx + 1):
                        step = _load_step_npz_from_zip(z, split, idx)

                        dataset.add_frame(
                            {
                                "observation.images": step["rgb_static"],
                                "observation.images.wrist": step["rgb_gripper"],
                                "observation.state": step["robot_obs"].astype(np.float32),
                                "action": step[args.action_key].astype(np.float32),
                            },
                            task=task,
                        )

                    # Suppress output from save_episode to avoid cluttering progress bars
                    with suppress_output():
                        dataset.save_episode()
    else:
        print(f"Reading from folder: {args.data_path}")
        data_path = Path(args.data_path)
        for split in splits:
            lang_data = _iter_episode_lang(data_path, split)
            ep_start_end_ids = lang_data["info"]["indx"]
            lang_ann = lang_data["language"]["ann"]
            lang_task = lang_data["language"]["task"]

            # Add progress bar for episode processing
            total_episodes = len(ep_start_end_ids)
            pbar = tqdm(enumerate(ep_start_end_ids),
                       total=total_episodes,
                       desc=f"Processing {split} episodes",
                       unit="episode")

            for i, (start_idx, end_idx) in pbar:
                task = lang_ann[i]
                num_frames = end_idx - start_idx + 1
                pbar.set_postfix({"frames": num_frames, "episode": f"{i+1}/{total_episodes}"})

                for idx in range(start_idx, end_idx + 1):
                    step = _load_step_npz_from_folder(data_path, split, idx)

                    dataset.add_frame(
                        {
                            "observation.images": step["rgb_static"],
                            "observation.images.wrist": step["rgb_gripper"],
                            "observation.state": step["robot_obs"].astype(np.float32),  # 15个
                            "action": step[args.action_key].astype(np.float32),
                        },
                        task=task,
                    )

                # Suppress output from save_episode to avoid cluttering progress bars
                with suppress_output():
                    dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["calvin", "task_D_D"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    main(tyro.cli(Args))