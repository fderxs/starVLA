#!/bin/bash
# Release one or all GPUs by stopping labtasker workers and all child processes
# (policy server + eval client) spawned under them.
#
# Usage:
#   bash examples/tools/labtasker/release_gpu.sh <gpu_id>  # release one GPU
#   bash examples/tools/labtasker/release_gpu.sh            # release all 8 GPUs (0-7)
#
# The worker will stop accepting new tasks immediately.
# Any currently running server or eval process on that GPU is also killed.

gpu_id=$1

if [ -z "$gpu_id" ]; then
    echo -e "\033[36m[Release] No GPU specified, releasing all GPUs (0-7) ...\033[0m"
    for i in $(seq 0 7); do
        bash "$0" "$i"
    done
    exit 0
fi

PID_FILE="./.tmp/$(hostname -s)/labtasker_worker_gpu_${gpu_id}.pid"
# Fall back to legacy flat layout (workers started before hostname-based dirs were added)
if [ ! -f "$PID_FILE" ]; then
    PID_FILE="./.tmp/labtasker_worker_gpu_${gpu_id}.pid"
fi

if [ ! -f "$PID_FILE" ]; then
    echo -e "\033[31mNo PID file found for GPU $gpu_id ($PID_FILE).\033[0m"
    echo "Was the worker started with start_workers.sh?"
    exit 1
fi

root_pid=$(cat "$PID_FILE")

# Recursively collect all descendant PIDs (depth-first)
collect_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null) || true
    for child in $children; do
        collect_tree "$child"
    done
    echo "$pid"
}

echo -e "\033[36m[Release] Stopping worker tree for GPU $gpu_id (root PID $root_pid) ...\033[0m"

# Collect PIDs first so we can report them, then kill leaves → root
all_pids=$(collect_tree "$root_pid")

killed=0
for pid in $all_pids; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && echo "  killed PID $pid" && ((killed++)) || true
    fi
done

# Give processes a moment to exit, then SIGKILL any survivors
sleep 2
for pid in $all_pids; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null && echo "  force-killed PID $pid" || true
    fi
done

rm -f "$PID_FILE"

echo -e "\033[32m[Release] GPU $gpu_id released ($killed process(es) stopped).\033[0m"
