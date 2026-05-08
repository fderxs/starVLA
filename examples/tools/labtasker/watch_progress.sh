#!/bin/bash
# Monitor labtasker task progress with a tqdm-style live display.
#
# Usage:
#   bash examples/tools/labtasker/watch_progress.sh [interval=15]
#
# interval: refresh interval in seconds (default: 15)
# Press Ctrl+C to exit.

interval=${1:-15}
bar_width=38

# ---- helpers ----
count_tasks() {
    labtasker task ls --status "$1" -q --no-pager 2>/dev/null | wc -l
}

# Format seconds as HH:MM:SS (or MM:SS when < 1 hour)
fmt_time() {
    local s=$1
    [ -z "$s" ] || [ "$s" -lt 0 ] 2>/dev/null && s=0
    local h=$((s / 3600))
    local m=$(( (s % 3600) / 60 ))
    local sec=$((s % 60))
    if [ $h -gt 0 ]; then
        printf "%d:%02d:%02d" $h $m $sec
    else
        printf "%d:%02d" $m $sec
    fi
}

# ---- init ----
printf "Querying initial task counts...\n"
initial_pending=$(count_tasks pending)
initial_running=$(count_tasks running)

if [ $((initial_pending + initial_running)) -eq 0 ]; then
    echo "No pending or running tasks found."
    exit 0
fi

printf "Refresh: %ds   Ctrl+C to exit.\n\n" "$interval"

start_time=$(date +%s)
done_count=0          # monotonically increasing completed task count
prev_remaining=0      # to detect newly completed tasks each refresh

# ---- main loop ----
while true; do
    pending=$(count_tasks pending)
    running=$(count_tasks running)
    remaining=$((pending + running))

    # Accumulate completed tasks (works even when new tasks are added mid-run)
    if [ $prev_remaining -gt 0 ]; then
        newly_done=$((prev_remaining - remaining))
        [ $newly_done -gt 0 ] && done_count=$((done_count + newly_done))
    fi
    prev_remaining=$remaining

    # Total is always recalculated to absorb newly submitted tasks
    total=$((done_count + remaining))

    now=$(date +%s)
    elapsed=$((now - start_time))
    timestamp=$(date '+%H:%M:%S')

    # --- progress bar ---
    pct=$((done_count * 100 / total))
    filled=$((done_count * bar_width / total))
    empty=$((bar_width - filled))
    bar=$(printf '%0.s█' $(seq 1 $filled) 2>/dev/null)
    bar+=$(printf '%0.s░' $(seq 1 $empty) 2>/dev/null)

    # --- rate and ETA (use overall rate for stability, like tqdm) ---
    if [ $done_count -gt 0 ] && [ $elapsed -gt 0 ]; then
        # rate in tasks/min, ETA in seconds
        eta_secs=$(awk "BEGIN {printf \"%d\", $remaining * $elapsed / $done_count}")
        rate_str=$(awk "BEGIN {
            r = $done_count / $elapsed * 60
            if (r >= 1) printf \"%.1f task/min\", r
            else        printf \"%.1f task/hr\",  r * 60
        }")
    else
        eta_secs=""
        rate_str="--"
    fi

    elapsed_str=$(fmt_time $elapsed)
    eta_str=$([ -n "$eta_secs" ] && fmt_time $eta_secs || echo "--:--")

    # --- render (overwrite current line) ---
    printf "\r\033[K\033[1m%3d%%\033[0m|%s| %d/%d [\033[33m%s\033[0m<\033[36m%s\033[0m, %s]  \033[32mRunning: %d\033[0m  Pending: %d  %s" \
        "$pct" "$bar" "$done_count" "$total" \
        "$elapsed_str" "$eta_str" "$rate_str" \
        "$running" "$pending" "$timestamp"

    if [ $remaining -eq 0 ]; then
        printf "\n\033[32mAll %d tasks completed in %s.\033[0m\n" "$total" "$elapsed_str"
        break
    fi

    sleep "$interval"
done
