#!/bin/bash
# Start N labtasker evaluation workers, one per GPU, in the background.
#
# Usage:
#   bash examples/tools/labtasker/start_workers.sh [gpu_start_id=0] [n=8]
#
# Each worker runs worker_eval.sh assigned to its own GPU and
# continuously pulls tasks from the labtasker queue until stopped.
#
# Logs are written to logs/labtasker_worker/<hostname>/gpu_<id>.log
#
# To stop all workers:
#   pkill -f worker_eval.sh
#   # or: kill $(cat ./.tmp/labtasker_worker_pids.txt)

gpu_start_id=${1:-0}
n=${2:-8}

hostname=$(hostname -s)
log_dir="logs/labtasker_worker/${hostname}"
pid_dir="./.tmp/${hostname}"

mkdir -p "$log_dir"
mkdir -p "$pid_dir"

PID_FILE="${pid_dir}/labtasker_worker_pids.txt"
> "$PID_FILE"  # truncate

echo -e "\033[36mStarting $n labtasker workers on GPUs $gpu_start_id .. $((gpu_start_id + n - 1)) (host: $hostname)\033[0m"

for i in $(seq 1 $n); do
    gpu_id=$((gpu_start_id + i - 1))
    log="${log_dir}/gpu_${gpu_id}.log"

    # Check if a worker is already running on this GPU
    per_gpu_pid_file="${pid_dir}/labtasker_worker_gpu_${gpu_id}.pid"
    if [ -f "$per_gpu_pid_file" ]; then
        existing_pid=$(cat "$per_gpu_pid_file")
        if kill -0 "$existing_pid" 2>/dev/null; then
            echo -e "\033[33m  GPU $gpu_id  already has a running worker (PID $existing_pid), skipping.\033[0m"
            echo "$existing_pid" >> "$PID_FILE"
            continue
        fi
    fi

    echo "  GPU $gpu_id  →  $log"
    bash examples/tools/labtasker/worker_eval.sh "$gpu_id" > "$log" 2>&1 &
    worker_pid=$!

    # Save PID both to the summary file and to a per-GPU file for targeted release
    echo "$worker_pid" >> "$PID_FILE"
    echo "$worker_pid" > "${pid_dir}/labtasker_worker_gpu_${gpu_id}.pid"
done

echo -e "\033[32mAll workers started. PIDs saved to $PID_FILE\033[0m"
echo "To stop all workers:      kill \$(cat $PID_FILE)"
echo "To release a single GPU:  bash examples/tools/labtasker/release_gpu.sh <gpu_id>"
