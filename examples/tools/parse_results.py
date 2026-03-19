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


def parse_libero_results(base_dir, model_name):
    """Parse LIBERO results"""
    tasks = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
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
                cell = f"{success_rate*100:04.1f}%"
                row += f"{cell:^{col_width}s}"
                valid_results.append(success_rate)
            else:
                row += f"{'N/A':^{col_width}s}"

        if valid_results:
            avg_rate = sum(valid_results) / len(valid_results)
            cell = f"{avg_rate*100:04.1f}%"
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
    all_tasks = sorted(all_tasks)

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
                cell = f"{mean_rate*100:04.1f}% [{count:>02d}]"
                row += f"{cell:^{task_col_width}s}"
                valid_means.append(mean_rate)
            else:
                row += f"{'N/A':^{task_col_width}s}"

        if valid_means:
            avg_mean = sum(valid_means) / len(valid_means)
            cell = f"{avg_mean*100:04.1f}%"
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
                        cell = f"{rate*100:04.1f}%"
                        row += f"{cell:^{task_col_width}s}"
                        valid_rates.append(rate)
                    else:
                        row += f"{'-':^{task_col_width}s}"

                if valid_rates:
                    avg_rate = sum(valid_rates) / len(valid_rates)
                    cell = f"{avg_rate*100:04.1f}%"
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
                        cell = f"{mean_rate*100:04.1f}%"
                        row += f"{cell:^{task_col_width}s}"
                        valid_metrics.append(mean_rate)
                    else:
                        row += f"{'N/A':^{task_col_width}s}"

                if valid_metrics:
                    avg_metric = sum(valid_metrics) / len(valid_metrics)
                    cell = f"{avg_metric*100:04.1f}%"
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
                        cell = f"{value*100:04.1f}%"
                    row += f"{cell:^{col_width}s}"
                else:
                    row += f"{'N/A':^{col_width}s}"

        print(row)

    print("=" * total_width)


def main():
    parser = argparse.ArgumentParser(
        description='Parse evaluation results from LIBERO, WidowX, and CALVIN benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse LIBERO results
  python parse_results.py -b libero -m StarVLA__Qwen2_5-VL-GR00T-LIBERO-4in1

  # Parse WidowX results
  python parse_results.py -b widowx -m StarVLA__Qwen3VL-GR00T-Bridge-RT-1

  # Parse WidowX results with all runs displayed
  python parse_results.py -b widowx -m StarVLA__Qwen3VL-GR00T-Bridge-RT-1 --show_all_runs

  # Parse CALVIN results
  python parse_results.py -b calvin -m calvin_task_D_D_qwen3gr00t

  # Parse multiple models
  python parse_results.py -b libero -m model1 model2 model3
        """
    )
    parser.add_argument('-b', '--benchmark', type=str, required=True,
                        choices=['libero', 'widowx', 'calvin'],
                        help='Benchmark type: libero, widowx, or calvin')
    parser.add_argument('-m', '--model_name', type=str, nargs='+', required=True,
                        help='Model name(s) to parse results for. Can specify multiple models.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory path (default: logs)')
    parser.add_argument('--show_all_runs', action='store_true',
                        help='Show detailed results for all runs (WidowX only)')

    args = parser.parse_args()

    # Determine base directory based on benchmark type
    if args.benchmark == 'libero':
        base_dir = Path(__file__).parent.parent.parent / args.log_dir
        parse_func = parse_libero_results
    elif args.benchmark == 'widowx':
        base_dir = Path(__file__).parent.parent.parent / args.log_dir / 'widowx'
        parse_func = lambda bd, mn: parse_widowx_results(bd, mn, args.show_all_runs)
    elif args.benchmark == 'calvin':
        base_dir = Path(__file__).parent.parent.parent / args.log_dir / 'calvin'
        parse_func = parse_calvin_results
    else:
        print(f"Unknown benchmark type: {args.benchmark}")
        return

    # Parse and display results for each model
    for i, model_name in enumerate(args.model_name):
        if i > 0:
            print("\n\n")
        parse_func(base_dir, model_name)


if __name__ == "__main__":
    main()
