#!/usr/bin/env python3
"""
Parse evaluation results from log files for LIBERO, WidowX, and CALVIN benchmarks
"""
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict


def extract_libero_success_rate(log_file):
    """Extract the total success rate from a LIBERO eval.log file"""
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Total success rate:" in line:
                    match = re.search(r'Total success rate:\s*([\d.]+)', line)
                    if match:
                        return float(match.group(1))
        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def extract_widowx_success_rate(log_file):
    """Extract the average success rate from a WidowX run log file"""
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Average success" in line:
                    match = re.search(r'Average success\s+([\d.]+)', line)
                    if match:
                        return float(match.group(1))
        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def extract_widowx_all_runs(task_dir):
    """Extract success rates from all run_X.log files in a task directory"""
    if not task_dir.exists():
        return None

    # Find all run_X.log files and sort by run number
    run_files = sorted(task_dir.glob("run_*.log"),
                      key=lambda x: int(re.search(r'run_(\d+)', x.name).group(1)))
    if not run_files:
        return None

    success_rates = []
    success_idxs = []
    for run_file in run_files:
        rate = extract_widowx_success_rate(run_file)
        if rate is not None:
            success_rates.append(rate)
            success_idxs.append(int(re.search(r'run_(\d+)', run_file.name).group(1)))

    if not success_rates:
        return None

    return {
        'rates': success_rates,
        'mean': sum(success_rates) / len(success_rates),
        'max': max(success_rates),
        'min': min(success_rates),
        'count': len(success_rates),
        'idxs': success_idxs
    }


def extract_calvin_results(log_file):
    """Extract success rates and avg.len from a CALVIN eval.log file"""
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            avg_len = None
            success_rates = {}

            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if "Average successful sequence length:" in line:
                    match = re.search(r'Average successful sequence length:\s*([\d.]+)', line)
                    if match:
                        avg_len = float(match.group(1))
                        for j in range(i + 1, min(i + 10, len(lines))):
                            rate_line = lines[j].strip()
                            rate_match = re.match(r'^(\d):\s*([\d.]+)%', rate_line)
                            if rate_match:
                                task_num = int(rate_match.group(1))
                                rate_val = float(rate_match.group(2)) / 100.0
                                success_rates[task_num] = rate_val
                        break

            if avg_len is not None and len(success_rates) == 5:
                return {
                    'avg_len': avg_len,
                    '1': success_rates.get(1),
                    '2': success_rates.get(2),
                    '3': success_rates.get(3),
                    '4': success_rates.get(4),
                    '5': success_rates.get(5),
                }
            return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def shorten_widowx_task_name(task_name):
    """Shorten WidowX task name for display"""
    abbreviations = {
        'PutCarrotOnPlateInScene-v0': 'Carrot->Plate',
        'PutEggplantInBasketScene-v0': 'Eggplant->Bskt',
        'PutSpoonOnTableClothInScene-v0': 'Spoon->Table',
        'StackGreenCubeOnYellowCubeBakedTexInScene-v0': 'Green->Yellow',
    }
    if task_name in abbreviations:
        return abbreviations[task_name]
    name = task_name.replace('BakedTexInScene-v0', '').replace('InScene-v0', '').replace('Scene-v0', '')
    if len(name) > 15:
        return name[:15]
    return name


def extract_google_robot_success_rate(log_file):
    """Extract the average success rate from a Google Robot eval.log file

    Google Robot logs contain multiple test episodes, each with an 'Average success' value.
    We need to collect all these values and compute their mean.
    """
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            success_rates = []
            for line in lines:
                if "Average success" in line:
                    match = re.search(r'Average success\s+([\d.]+)', line)
                    if match:
                        success_rates.append(float(match.group(1)))

            if success_rates:
                return sum(success_rates) / len(success_rates)
            return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def extract_google_robot_all_runs(task_dir):
    """Extract success rates from all eval*.log files in a Google Robot task directory

    Returns a dict with:
    - 'rates': list of success rates from all runs
    - 'mean': mean success rate
    - 'max': maximum success rate
    - 'min': minimum success rate
    - 'count': number of runs
    - 'file_rates': dict mapping file index to success rate
    """
    if not task_dir.exists():
        return None

    success_rates = []
    file_rates = {}  # Map file index to rate

    # Look for eval.log, eval_1.log, eval_2.log, etc.
    for log_file in sorted(task_dir.glob("eval*.log")):
        rate = extract_google_robot_success_rate(log_file)
        if rate is not None:
            success_rates.append(rate)
            # Extract file index (eval.log -> 0, eval_1.log -> 1, etc.)
            name = log_file.stem  # e.g., "eval" or "eval_1"
            if name == "eval":
                file_idx = 0
            else:
                # Extract number from eval_N
                match = re.search(r'eval_(\d+)', name)
                if match:
                    file_idx = int(match.group(1))
                else:
                    file_idx = len(file_rates)
            file_rates[file_idx] = rate

    if not success_rates:
        return None

    return {
        'rates': success_rates,
        'mean': sum(success_rates) / len(success_rates),
        'max': max(success_rates),
        'min': min(success_rates),
        'count': len(success_rates),
        'file_rates': file_rates
    }


def shorten_google_robot_task_name(task_name):
    """Shorten Google Robot task name for display"""
    abbreviations = {
        'drawer_variant_agg': 'Drawer-VA',
        'drawer_visual_matching': 'Drawer-VM',
        'move_near_variant_agg': 'MoveNear-VA',
        'move_near_visual_matching': 'MoveNear-VM',
        'pick_coke_can_variant_agg': 'PickCoke-VA',
        'pick_coke_can_visual_matching': 'PickCoke-VM',
        'put_in_drawer_variant_agg': 'PutDrawer-VA',
        'put_in_drawer_visual_matching': 'PutDrawer-VM',
    }
    if task_name in abbreviations:
        return abbreviations[task_name]
    # Fallback: capitalize and truncate
    return task_name.replace('_', ' ').title()[:15]


def extract_robotwin_success_rates(log_file):
    """Extract success rates from a Robotwin eval.log file

    Returns a dict mapping task names to their final success rates.
    Each task has 10 test runs, and we extract the final X/10 success rate.
    """
    if not os.path.exists(log_file):
        return {}

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Pattern to match task name and success rate
        # Task name uses ANSI color codes: \x1b[93m<task_name>\x1b[0m
        # Success rate uses ANSI codes: Success rate: \x1b[96mX/10\x1b[0m => \x1b[95mYY.Y%\x1b[0m
        task_results = {}

        import re
        # Match ANSI escape sequences: \x1b[XXm or \033[XXm
        task_pattern = r'\x1b\[93m([^\x1b]+)\x1b\[0m'
        success_pattern = r'Success rate:.*?\x1b\[96m(\d+)/(\d+)\x1b\[0m.*?\x1b\[95m([\d.]+)%\x1b\[0m'

        # Iterate through lines and track task-success pairs
        for i, line in enumerate(lines):
            # Check if this line contains a task name
            task_match = re.search(task_pattern, line)
            if task_match:
                task_name = task_match.group(1)

                # Look for the next line with success rate
                for j in range(i, min(i + 5, len(lines))):  # Check next few lines
                    success_match = re.search(success_pattern, lines[j])
                    if success_match:
                        succeeded = int(success_match.group(1))
                        total = int(success_match.group(2))
                        percentage = float(success_match.group(3))

                        # Only keep the final result (when total == 10)
                        if total == 10:
                            task_results[task_name] = percentage / 100.0
                        break

        return task_results

    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return {}



def parse_libero_results(base_dir, model_name):
    """Parse LIBERO results"""
    tasks = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
    results = defaultdict(dict)
    checkpoint_steps = set()
    if model_name is None: model_name = "StarVLA__Qwen2_5-VL-GR00T-LIBERO-4in1"

    for task in tasks:
        task_dir = base_dir / task / model_name
        if not task_dir.exists():
            continue
        for checkpoint_dir in sorted(task_dir.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("steps_"):
                checkpoint_steps.add(checkpoint_dir.name)
                log_file = checkpoint_dir / "eval.log"
                results[checkpoint_dir.name][task] = extract_libero_success_rate(log_file)

    sorted_checkpoints = sorted(checkpoint_steps, key=lambda x: int(re.search(r'steps_(\d+)', x).group(1)))

    col_width = 17
    total_width = 15 + col_width * (len(tasks) + 1)

    print("=" * total_width)
    print(f"Model: {model_name}")
    print("=" * total_width)
    print()

    header = f"{'Checkpoint':<15s}"
    for task in tasks:
        task_short = task.replace('libero_', 'LIBERO-').upper()
        header += f"{task_short:^{col_width}s}"
    header += f"{'Average':^{col_width}s}"
    print(header)
    print("-" * total_width)

    for checkpoint in sorted_checkpoints:
        step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
        row = f"steps_{step_num:<9s}"
        valid_results = []

        for task in tasks:
            success_rate = results[checkpoint].get(task)
            if success_rate is not None:
                cell = f"{success_rate*100:5.1f}%"
                row += f"{cell:^{col_width}s}"
                valid_results.append(success_rate)
            else:
                row += f"{'N/A':^{col_width}s}"

        if valid_results:
            avg_rate = sum(valid_results) / len(valid_results)
            cell = f"{avg_rate*100:5.1f}%"
            row += f"{cell:^{col_width}s}"
        else:
            row += f"{'N/A':^{col_width}s}"

        print(row)

    print("=" * total_width)


def parse_widowx_results(base_dir, model_name, show_all_runs=False):
    """Parse WidowX results"""
    model_dir = base_dir / model_name
    if not model_dir.exists():
        print(f"Warning: Model directory not found: {model_dir}")
        return

    # Get all tasks
    all_tasks = set()
    for checkpoint_dir in model_dir.iterdir():
        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("steps_"):
            for task_dir in checkpoint_dir.iterdir():
                if task_dir.is_dir():
                    all_tasks.add(task_dir.name)

    # Define desired task order
    task_order = [
        'PutSpoonOnTableClothInScene-v0',                   # Spoon->Table
        'PutCarrotOnPlateInScene-v0',                       # Carrot->Plate
        'StackGreenCubeOnYellowCubeBakedTexInScene-v0',     # Green->Yellow
        'PutEggplantInBasketScene-v0',                      # Eggplant->Bskt
    ]

    # Reorder all_tasks according to task_order, keeping any extra tasks at the end
    ordered_tasks = [task for task in task_order if task in all_tasks]
    extra_tasks = sorted([task for task in all_tasks if task not in task_order])
    all_tasks = ordered_tasks + extra_tasks

    if not all_tasks:
        print(f"Warning: No tasks found for model: {model_name}")
        return

    results = defaultdict(dict)
    checkpoint_steps = set()

    for checkpoint_dir in sorted(model_dir.iterdir()):
        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("steps_"):
            checkpoint_steps.add(checkpoint_dir.name)
            for task in all_tasks:
                task_dir = checkpoint_dir / task
                # Extract all runs for this task
                results[checkpoint_dir.name][task] = extract_widowx_all_runs(task_dir)

    sorted_checkpoints = sorted(checkpoint_steps, key=lambda x: int(re.search(r'steps_(\d+)', x).group(1)))

    task_col_width = 17
    total_width = 15 + task_col_width * len(all_tasks) + 17

    print("=" * total_width)
    print(f"Model: {model_name}")
    print("=" * total_width)
    print()

    header = f"{'Checkpoint':<15s}"
    for task in all_tasks:
        task_display = shorten_widowx_task_name(task)
        header += f"{task_display:^{task_col_width}s}"
    header += f"{'Average':^17s}"
    print(header)
    print("-" * total_width)

    # Print mean row for each checkpoint
    for checkpoint in sorted_checkpoints:
        step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
        row = f"steps_{step_num:<9s}"
        valid_means = []

        for task in all_tasks:
            task_result = results[checkpoint].get(task)
            if task_result is not None:
                mean_rate = task_result['mean']
                count = task_result['count']
                # Show count in brackets if more than 1 run
                cell = f"{mean_rate*100:4.1f}% [{count:2d}]"
                row += f"{cell:^{task_col_width}s}"
                valid_means.append(mean_rate)
            else:
                row += f"{'N/A':^{task_col_width}s}"

        if valid_means:
            avg_mean = sum(valid_means) / len(valid_means)
            cell = f"{avg_mean*100:5.1f}%"
            row += f"{cell:^17s}"
        else:
            row += f"{'N/A':^17s}"

        print(row)

    print("=" * total_width)

    # Show detailed results for all runs if requested
    if show_all_runs:
        print()
        print("=" * total_width)
        print("Detailed Results for All Runs (Multiple Runs Only)")
        print("=" * total_width)

        has_detailed_output = False
        for checkpoint in sorted_checkpoints:
            step_num = re.search(r'steps_(\d+)', checkpoint).group(1)

            # Check if this checkpoint has any task with multiple runs
            has_multiple_runs = any(
                results[checkpoint].get(task) is not None and results[checkpoint].get(task)['count'] > 1
                for task in all_tasks
            )

            if not has_multiple_runs:
                continue

            has_detailed_output = True
            print()
            print(f"Checkpoint: steps_{step_num}")
            print("-" * total_width)

            # Find max number of runs across all tasks for this checkpoint
            max_runs = 0
            for task in all_tasks:
                task_result = results[checkpoint].get(task)
                if task_result is not None:
                    max_runs = max(max_runs, len(task_result['rates']))

            if max_runs == 0:
                continue

            # Print header
            header = f"{'Run':<15s}"
            for task in all_tasks:
                task_display = shorten_widowx_task_name(task)
                header += f"{task_display:^{task_col_width}s}"
            header += f"{'Average':^17s}"
            print(header)
            print("-" * total_width)

            # Print each run
            for run_idx in range(max_runs):
                row = f"Run {run_idx+1:<11d}"
                valid_rates = []

                for task in all_tasks:
                    task_result = results[checkpoint].get(task)
                    if task_result is not None and (run_idx+1) in task_result['idxs']:
                        real_idx = task_result['idxs'].index(run_idx+1)
                        rate = task_result['rates'][real_idx]
                        cell = f"{rate*100:4.1f}%"
                        row += f"{cell:^{task_col_width}s}"
                        valid_rates.append(rate)
                    else:
                        row += f"{'-':^{task_col_width}s}"

                if valid_rates:
                    avg_rate = sum(valid_rates) / len(valid_rates)
                    cell = f"{avg_rate*100:4.1f}%"
                    row += f"{cell:^17s}"
                else:
                    row += f"{'-':^17s}"

                print(row)

            for metric in ['Mean', 'Max', 'Min']:
                row = f"{metric:<15s}"
                valid_metrics = []
                for task in all_tasks:
                    task_result = results[checkpoint].get(task)
                    if task_result is not None:
                        mean_rate = task_result[metric.lower()]
                        cell = f"{mean_rate*100:4.1f}%"
                        row += f"{cell:^{task_col_width}s}"
                        valid_metrics.append(mean_rate)
                    else:
                        row += f"{'N/A':^{task_col_width}s}"

                if valid_metrics:
                    avg_metric = sum(valid_metrics) / len(valid_metrics)
                    cell = f"{avg_metric*100:4.1f}%"
                    row += f"{cell:^17s}"
                else:
                    row += f"{'N/A':^17s}"
                print(row)

        if not has_detailed_output:
            print("No checkpoints with multiple runs found.")

        print("=" * total_width)

def parse_calvin_results(base_dir, model_name):
    """Parse CALVIN results"""
    if model_name is None: 
        model_name = "Simplicissimus-S__StarVLA-QwenGR00T_Qwen2_5-VL-3B-Instruct-Action_calvin_D_D"
    model_dir = base_dir / model_name
    if not model_dir.exists():
        print(f"Warning: Model directory not found: {model_dir}")
        return

    results = defaultdict(dict)
    checkpoint_steps = set()

    for checkpoint_dir in sorted(model_dir.iterdir()):
        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("steps_"):
            checkpoint_steps.add(checkpoint_dir.name)
            log_file = checkpoint_dir / "eval.log"
            result = extract_calvin_results(log_file)
            results[checkpoint_dir.name] = result

    sorted_checkpoints = sorted(checkpoint_steps, key=lambda x: int(re.search(r'steps_(\d+)', x).group(1)))

    metrics = ['1', '2', '3', '4', '5', 'avg_len']
    metric_names = ['SR-1', 'SR-2', 'SR-3', 'SR-4', 'SR-5', 'Avg.Len']

    col_width = 14
    total_width = 15 + col_width * len(metrics)

    print("=" * total_width)
    print(f"Model: {model_name}")
    print("=" * total_width)
    print()

    header = f"{'Checkpoint':<15s}"
    for metric_name in metric_names:
        header += f"{metric_name:^{col_width}s}"
    print(header)
    print("-" * total_width)

    for checkpoint in sorted_checkpoints:
        step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
        row = f"steps_{step_num:<9s}"
        checkpoint_results = results[checkpoint]

        if checkpoint_results is None:
            for _ in metrics:
                row += f"{'N/A':^{col_width}s}"
        else:
            for metric in metrics:
                value = checkpoint_results.get(metric)
                if value is not None:
                    if metric == 'avg_len':
                        cell = f"{value:.3f}"
                    else:
                        cell = f"{value*100:4.1f}%"
                    row += f"{cell:^{col_width}s}"
                else:
                    row += f"{'N/A':^{col_width}s}"

        print(row)

    print("=" * total_width)


def parse_google_robot_results(base_dir, model_name, show_all_runs=False):
    """Parse Google Robot results with support for multiple runs"""
    # Google Robot tasks grouped by type
    va_tasks = [
        'drawer_variant_agg',
        'move_near_variant_agg',
        'pick_coke_can_variant_agg',
        'put_in_drawer_variant_agg',
    ]
    vm_tasks = [
        'drawer_visual_matching',
        'move_near_visual_matching',
        'pick_coke_can_visual_matching',
        'put_in_drawer_visual_matching',
    ]
    all_tasks = va_tasks + vm_tasks

    # Ordered display: each task type's VM and VA together
    ordered_tasks = [
        'pick_coke_can_visual_matching',    # PickCoke-VM
        'pick_coke_can_variant_agg',        # PickCoke-VA
        'move_near_visual_matching',        # MoveNear-VM
        'move_near_variant_agg',            # MoveNear-VA
        'drawer_visual_matching',           # Drawer-VM
        'drawer_variant_agg',               # Drawer-VA
        'put_in_drawer_visual_matching',    # PutDrawer-VM
        'put_in_drawer_variant_agg',        # PutDrawer-VA
    ]

    model_dir = base_dir / model_name
    if not model_dir.exists():
        print(f"Warning: Model directory not found: {model_dir}")
        return

    results = defaultdict(dict)
    checkpoint_steps = set()

    # Parse all log files
    for checkpoint_dir in sorted(model_dir.iterdir()):
        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("steps_"):
            checkpoint_steps.add(checkpoint_dir.name)
            for task in all_tasks:
                task_dir = checkpoint_dir / task
                # Try to get all runs
                all_runs = extract_google_robot_all_runs(task_dir)
                results[checkpoint_dir.name][task] = all_runs

    if not checkpoint_steps:
        print(f"Warning: No checkpoints found for model: {model_name}")
        return

    sorted_checkpoints = sorted(checkpoint_steps, key=lambda x: int(re.search(r'steps_(\d+)', x).group(1)))

    # Column widths
    col_width = 13
    avg_width = 11
    total_width = 15 + col_width * 8 + avg_width * 3

    print("=" * total_width)
    print(f"Model: {model_name}")
    print("=" * total_width)
    print()

    # Print header
    header = f"{'Checkpoint':<15s}"
    for task in ordered_tasks:
        task_display = shorten_google_robot_task_name(task)
        header += f"{task_display:^{col_width}s}"
    header += f"{'VM-Avg':^{avg_width}s}"
    header += f"{'VA-Avg':^{avg_width}s}"
    header += f"{'Overall':^{avg_width}s}"
    print(header)
    print("-" * total_width)

    # Print results for each checkpoint (Mean values only)
    for checkpoint in sorted_checkpoints:
        step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
        row = f"steps_{step_num:<9s}"

        va_means = []
        vm_means = []

        # All tasks in display order
        for task in ordered_tasks:
            task_data = results[checkpoint].get(task)
            if task_data:
                mean_rate = task_data['mean']
                count = task_data['count']
                if count > 1:
                    cell = f"{mean_rate*100:4.1f}%[{count}]"
                else:
                    cell = f"{mean_rate*100:5.1f}%"
                row += f"{cell:^{col_width}s}"

                # Track for averages
                if task in vm_tasks:
                    vm_means.append(mean_rate)
                elif task in va_tasks:
                    va_means.append(mean_rate)
            else:
                row += f"{'N/A':^{col_width}s}"

        # VM average
        if vm_means:
            vm_avg = sum(vm_means) / len(vm_means)
            cell = f"{vm_avg*100:5.1f}%"
            row += f"{cell:^{avg_width}s}"
        else:
            row += f"{'N/A':^{avg_width}s}"

        # VA average
        if va_means:
            va_avg = sum(va_means) / len(va_means)
            cell = f"{va_avg*100:5.1f}%"
            row += f"{cell:^{avg_width}s}"
        else:
            row += f"{'N/A':^{avg_width}s}"

        # Overall average
        all_means = vm_means + va_means
        if all_means:
            overall_avg = sum(all_means) / len(all_means)
            cell = f"{overall_avg*100:5.1f}%"
            row += f"{cell:^{avg_width}s}"
        else:
            row += f"{'N/A':^{avg_width}s}"

        print(row)

    print("=" * total_width)

    # Show detailed results for all runs if requested
    if show_all_runs:
        print("\n")
        print("=" * total_width)
        print("Detailed Results for All Runs (Multiple Runs Only)")
        print("=" * total_width)

        has_detailed_output = False
        for checkpoint in sorted_checkpoints:
            step_num = re.search(r'steps_(\d+)', checkpoint).group(1)

            # Check if this checkpoint has multiple runs
            has_multiple_runs = any(
                results[checkpoint].get(task) is not None and
                results[checkpoint].get(task)['count'] > 1
                for task in all_tasks
            )

            if not has_multiple_runs:
                continue

            has_detailed_output = True
            print(f"\nCheckpoint: steps_{step_num}")
            print("-" * total_width)

            # Find all file indices
            all_file_indices = set()
            for task in all_tasks:
                task_data = results[checkpoint].get(task)
                if task_data and task_data['count'] > 1:
                    all_file_indices.update(task_data['file_rates'].keys())

            sorted_file_indices = sorted(all_file_indices)

            if not sorted_file_indices:
                continue

            # Print header
            header = f"{'Eval':<15s}"
            for task in ordered_tasks:
                task_display = shorten_google_robot_task_name(task)
                header += f"{task_display:^{col_width}s}"
            header += f"{'VM-Avg':^{avg_width}s}"
            header += f"{'VA-Avg':^{avg_width}s}"
            header += f"{'Overall':^{avg_width}s}"
            print(header)
            print("-" * total_width)

            # Print each eval file
            for file_idx in sorted_file_indices:
                if file_idx == 0:
                    eval_name = "eval.log"
                else:
                    eval_name = f"eval_{file_idx}.log"
                row = f"{eval_name:<15s}"

                vm_rates = []
                va_rates = []

                # All tasks in display order
                for task in ordered_tasks:
                    task_data = results[checkpoint].get(task)
                    if task_data and file_idx in task_data['file_rates']:
                        rate = task_data['file_rates'][file_idx]
                        cell = f"{rate*100:5.1f}%"
                        row += f"{cell:^{col_width}s}"

                        # Track for averages
                        if task in vm_tasks:
                            vm_rates.append(rate)
                        elif task in va_tasks:
                            va_rates.append(rate)
                    else:
                        row += f"{'-':^{col_width}s}"

                # VM average for this eval
                if vm_rates:
                    vm_avg = sum(vm_rates) / len(vm_rates)
                    cell = f"{vm_avg*100:5.1f}%"
                    row += f"{cell:^{avg_width}s}"
                else:
                    row += f"{'-':^{avg_width}s}"

                # VA average for this eval
                if va_rates:
                    va_avg = sum(va_rates) / len(va_rates)
                    cell = f"{va_avg*100:5.1f}%"
                    row += f"{cell:^{avg_width}s}"
                else:
                    row += f"{'-':^{avg_width}s}"

                # Overall average for this eval
                all_rates = vm_rates + va_rates
                if all_rates:
                    overall_avg = sum(all_rates) / len(all_rates)
                    cell = f"{overall_avg*100:5.1f}%"
                    row += f"{cell:^{avg_width}s}"
                else:
                    row += f"{'-':^{avg_width}s}"

                print(row)

            # Print Mean, Max, Min rows
            for metric_name in ['Mean', 'Max', 'Min']:
                row = f"{metric_name:<15s}"

                vm_metrics = []
                va_metrics = []

                # All tasks in display order
                for task in ordered_tasks:
                    task_data = results[checkpoint].get(task)
                    if task_data and task_data['count'] > 1:
                        metric_val = task_data[metric_name.lower()]
                        cell = f"{metric_val*100:5.1f}%"
                        row += f"{cell:^{col_width}s}"

                        # Track for averages
                        if task in vm_tasks:
                            vm_metrics.append(metric_val)
                        elif task in va_tasks:
                            va_metrics.append(metric_val)
                    else:
                        row += f"{'-':^{col_width}s}"

                # VM average
                if vm_metrics:
                    vm_avg = sum(vm_metrics) / len(vm_metrics)
                    cell = f"{vm_avg*100:5.1f}%"
                    row += f"{cell:^{avg_width}s}"
                else:
                    row += f"{'-':^{avg_width}s}"

                # VA average
                if va_metrics:
                    va_avg = sum(va_metrics) / len(va_metrics)
                    cell = f"{va_avg*100:5.1f}%"
                    row += f"{cell:^{avg_width}s}"
                else:
                    row += f"{'-':^{avg_width}s}"

                # Overall average
                all_metrics = vm_metrics + va_metrics
                if all_metrics:
                    overall_avg = sum(all_metrics) / len(all_metrics)
                    cell = f"{overall_avg*100:5.1f}%"
                    row += f"{cell:^{avg_width}s}"
                else:
                    row += f"{'-':^{avg_width}s}"

                print(row)

        if not has_detailed_output:
            print("No checkpoints with multiple runs found.")

        print("=" * total_width)


def parse_robotwin_results(base_dir, model_name, show_all_runs=False):
    """Parse Robotwin results (clean and randomized)"""
    model_dir = base_dir / model_name
    if not model_dir.exists():
        print(f"Warning: Model directory not found: {model_dir}")
        return

    results = defaultdict(lambda: {'clean': {}, 'randomized': {}})
    checkpoint_steps = set()

    def load_robotwin_results(eval_dir):
        """Load robotwin results from an eval directory (clean/ or randomized/).

        Tries in priority order:
          1. eval_all.log  — all 50 tasks recorded in a single file
          2. eval_<suite>.log — one file per sub-task (merged into one dict)
        """
        if not eval_dir.exists():
            return {}

        # 1. Single combined log
        all_log = eval_dir / "eval_all.log"
        if all_log.exists():
            return extract_robotwin_success_rates(all_log)

        # 2. Legacy filename
        legacy_log = eval_dir / "eval.log"
        if legacy_log.exists():
            return extract_robotwin_success_rates(legacy_log)

        # 3. Per-task logs: eval_<suite>.log (exclude eval_all.log already handled above)
        per_task_logs = [
            f for f in sorted(eval_dir.glob("eval_*.log"))
            if f.name != "eval_all.log"
        ]
        if per_task_logs:
            merged = {}
            for log_file in per_task_logs:
                merged.update(extract_robotwin_success_rates(log_file))
            return merged
    
    # Parse all log files
    for checkpoint_dir in sorted(model_dir.iterdir()):
        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("steps_"):
            checkpoint_steps.add(checkpoint_dir.name)

            results[checkpoint_dir.name]['clean'] = load_robotwin_results(
                checkpoint_dir / "clean"
            )
            results[checkpoint_dir.name]['randomized'] = load_robotwin_results(
                checkpoint_dir / "randomized"
            )

    if not checkpoint_steps:
        print(f"Warning: No checkpoints found for model: {model_name}")
        return

    sorted_checkpoints = sorted(checkpoint_steps, key=lambda x: int(re.search(r'steps_(\d+)', x).group(1)))

    # Display results summary
    col_width = 12
    total_width = 15 + col_width * 3

    print("=" * total_width)
    print(f"Model: {model_name}")
    print("=" * total_width)
    print()

    # Print header
    header = f"{'Checkpoint':<15s}"
    header += f"{'Clean':^{col_width}s}"
    header += f"{'Randomized':^{col_width}s}"
    header += f"{'Average':^{col_width}s}"
    print(header)
    print("-" * total_width)

    # Print results for each checkpoint
    for checkpoint in sorted_checkpoints:
        step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
        row = f"steps_{step_num:<9s}"

        # Calculate clean average
        clean_tasks = results[checkpoint]['clean']
        if clean_tasks:
            clean_avg = sum(clean_tasks.values()) / len(clean_tasks) * 100
            cell = f"{clean_avg:4.1f}%[{len(clean_tasks):2d}]"
            row += f"{cell:^{col_width}s}"
        else:
            row += f"{'N/A':^{col_width}s}"

        # Calculate randomized average
        randomized_tasks = results[checkpoint]['randomized']
        if randomized_tasks:
            randomized_avg = sum(randomized_tasks.values()) / len(randomized_tasks) * 100
            cell = f"{randomized_avg:4.1f}%[{len(randomized_tasks):2d}]"
            row += f"{cell:^{col_width}s}"
        else:
            row += f"{'N/A':^{col_width}s}"

        # Calculate overall average
        all_rates = []
        if clean_tasks:
            all_rates.extend(clean_tasks.values())
        if randomized_tasks:
            all_rates.extend(randomized_tasks.values())

        if all_rates:
            overall_avg = sum(all_rates) / len(all_rates) * 100
            cell = f"{overall_avg:4.1f}%[{len(all_rates):2d}]"
            row += f"{cell:^{col_width}s}"
        else:
            row += f"{'N/A':^{col_width}s}"

        print(row)

    print("=" * total_width)

    # Show detailed task results if requested
    if show_all_runs:
        # Collect all unique task names for clean and randomized
        all_clean_tasks = set()
        all_randomized_tasks = set()

        for checkpoint in sorted_checkpoints:
            clean_tasks = results[checkpoint]['clean']
            randomized_tasks = results[checkpoint]['randomized']
            all_clean_tasks.update(clean_tasks.keys())
            all_randomized_tasks.update(randomized_tasks.keys())

        all_clean_tasks = sorted(all_clean_tasks)
        all_randomized_tasks = sorted(all_randomized_tasks)

        if not all_clean_tasks and not all_randomized_tasks:
            print("\nNo tasks found for detailed display.")
            return

        print("\n")

        # Display Clean tasks table (vertical layout: tasks as rows, checkpoints as columns)
        if all_clean_tasks:
            checkpoint_col_width = 12
            total_clean_width = 40 + checkpoint_col_width * len(sorted_checkpoints)

            print("=" * total_clean_width)
            print("Clean Tasks - All Checkpoints")
            print("=" * total_clean_width)
            print()

            # Header (checkpoints as columns)
            header = f"{'Task ID':<8s} {'Task Name':<30s}"
            for checkpoint in sorted_checkpoints:
                step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
                header += f"{'s'+step_num:^{checkpoint_col_width}s}"
            print(header)
            print("-" * total_clean_width)

            # Each task as a row
            for idx, task in enumerate(all_clean_tasks):
                task_short = task[:28] if len(task) > 28 else task
                row = f"{idx:<8d} {task_short:<30s}"

                for checkpoint in sorted_checkpoints:
                    clean_tasks = results[checkpoint]['clean']
                    if task in clean_tasks:
                        rate = clean_tasks[task]
                        cell = f"{rate*100:5.1f}%"
                        row += f"{cell:^{checkpoint_col_width}s}"
                    else:
                        row += f"{'-':^{checkpoint_col_width}s}"

                print(row)

            print("=" * total_clean_width)

        # Display Randomized tasks table (vertical layout: tasks as rows, checkpoints as columns)
        if all_randomized_tasks:
            checkpoint_col_width = 12
            total_rand_width = 40 + checkpoint_col_width * len(sorted_checkpoints)

            print("\n")
            print("=" * total_rand_width)
            print("Randomized Tasks - All Checkpoints")
            print("=" * total_rand_width)
            print()

            # Header (checkpoints as columns)
            header = f"{'Task ID':<8s} {'Task Name':<30s}"
            for checkpoint in sorted_checkpoints:
                step_num = re.search(r'steps_(\d+)', checkpoint).group(1)
                header += f"{'s'+step_num:^{checkpoint_col_width}s}"
            print(header)
            print("-" * total_rand_width)

            # Each task as a row
            for idx, task in enumerate(all_randomized_tasks):
                task_short = task[:28] if len(task) > 28 else task
                row = f"{idx:<8d} {task_short:<30s}"

                for checkpoint in sorted_checkpoints:
                    randomized_tasks = results[checkpoint]['randomized']
                    if task in randomized_tasks:
                        rate = randomized_tasks[task]
                        cell = f"{rate*100:5.1f}%"
                        row += f"{cell:^{checkpoint_col_width}s}"
                    else:
                        row += f"{'-':^{checkpoint_col_width}s}"

                print(row)

            print("=" * total_rand_width)


def main():
    parser = argparse.ArgumentParser(
        description='Parse evaluation results from LIBERO, WidowX, CALVIN, Google Robot, and Robotwin benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse LIBERO results
  python parse_results.py -L -m StarVLA__Qwen2_5-VL-GR00T-LIBERO-4in1

  # Parse WidowX results
  python parse_results.py -W -m StarVLA__Qwen3VL-GR00T-Bridge-RT-1

  # Parse WidowX results with all runs displayed
  python parse_results.py -W -m StarVLA__Qwen3VL-GR00T-Bridge-RT-1 -a

  # Parse CALVIN results
  python parse_results.py -C -m calvin_task_D_D_qwen3gr00t

  # Parse Google Robot results
  python parse_results.py -G -m bridge_rt_1_qwen3gr00t

  # Parse Google Robot results with all runs displayed
  python parse_results.py -G -m bridge_rt_1_qwen3gr00t -a

  # Parse Robotwin results
  python parse_results.py -R -m robotwin_all_50_qwen3OFT_all

  # Parse Robotwin results with all runs displayed
  python parse_results.py -R -m robotwin_all_50_qwen3OFT_all -a

  # Parse multiple models
  python parse_results.py -L -m model1 model2 model3
        """
    )

    # Benchmark selection (mutually exclusive)
    benchmark_group = parser.add_mutually_exclusive_group(required=True)
    benchmark_group.add_argument('-L', '--libero', action='store_true',
                                 help='Parse LIBERO benchmark results')
    benchmark_group.add_argument('-W', '--widowx', action='store_true',
                                 help='Parse WidowX benchmark results')
    benchmark_group.add_argument('-C', '--calvin', action='store_true',
                                 help='Parse CALVIN benchmark results')
    benchmark_group.add_argument('-G', '--google_robot', action='store_true',
                                 help='Parse Google Robot benchmark results')
    benchmark_group.add_argument('-R', '--robotwin', action='store_true',
                                 help='Parse Robotwin benchmark results')

    parser.add_argument('-m', '--model_name', type=str, nargs='+', required=True,
                        help='Model name(s) to parse results for. Can specify multiple models.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory path (default: logs)')
    parser.add_argument('-a', '--show_all_runs', action='store_true',
                        help='Show detailed results for all runs (WidowX, Google Robot, and Robotwin)')

    args = parser.parse_args()

    # Determine base directory based on benchmark type
    if args.libero:
        base_dir = Path(__file__).parent.parent.parent / args.log_dir
        parse_func = parse_libero_results
    elif args.widowx:
        base_dir = Path(__file__).parent.parent.parent / args.log_dir / 'widowx'
        parse_func = lambda bd, mn: parse_widowx_results(bd, mn, args.show_all_runs)
    elif args.calvin:
        base_dir = Path(__file__).parent.parent.parent / args.log_dir / 'calvin'
        parse_func = parse_calvin_results
    elif args.google_robot:
        base_dir = Path(__file__).parent.parent.parent / args.log_dir / 'google_robot'
        parse_func = lambda bd, mn: parse_google_robot_results(bd, mn, args.show_all_runs)
    elif args.robotwin:
        base_dir = Path(__file__).parent.parent.parent / args.log_dir / 'Robotwin'
        parse_func = lambda bd, mn: parse_robotwin_results(bd, mn, args.show_all_runs)
    else:
        print(f"Error: No benchmark specified")
        return

    # Parse and display results for each model
    for i, model_name in enumerate(args.model_name):
        if i > 0:
            print("\n\n")
        parse_func(base_dir, model_name)


if __name__ == "__main__":
    main()
