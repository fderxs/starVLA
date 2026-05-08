#!/bin/bash
# Start a single labtasker evaluation worker tied to one GPU.
#
# The worker continuously fetches tasks from the labtasker queue,
# spins up the policy server on the assigned GPU, runs the evaluation
# client, and shuts the server down before accepting the next task.
#
# Usage (set CUDA_VISIBLE_DEVICES before calling, or pass gpu_id as $1):
#   CUDA_VISIBLE_DEVICES=0 bash examples/tools/labtasker/worker_eval.sh
#   bash examples/tools/labtasker/worker_eval.sh 0          # equivalent
#
# To start workers on multiple GPUs at once, use:
#   bash examples/tools/labtasker/start_workers.sh [gpu_start_id=0] [n=8]

# Resolve GPU id
if [ -n "$1" ]; then
    gpu_id=$1
    export CUDA_VISIBLE_DEVICES=$gpu_id
else
    gpu_id=${CUDA_VISIBLE_DEVICES:-0}
fi

echo -e "\033[36m[Worker] Starting labtasker worker on GPU $gpu_id\033[0m"

# ---- Build the per-task script that labtasker will run for each task ----
mkdir -p ./.tmp
LABTASKER_TASK_SCRIPT=$(mktemp ./.tmp/labtasker_eval_XXXXXX.sh)

cat <<'LABTASKER_LOOP_EOF' > "$LABTASKER_TASK_SCRIPT"
#!/bin/bash
# --- Variables injected by labtasker ---
ckpt='%(ckpt)'
benchmark='%(benchmark)'
test_num='%(test_num)'
start_run_id='%(start_run_id)'
suites='%(suites)'
task_suite_name='%(task_suite_name)'
robotwin_type='%(robotwin_type)'

# --- Derive GPU id from environment (set by the worker launcher) ---
gpu_id=${CUDA_VISIBLE_DEVICES:-0}

echo "[Task] ckpt=$ckpt  benchmark=$benchmark  gpu=$gpu_id"

# ---- Map benchmark name to its directory ----
case "$benchmark" in
    widowx|google_robot) bmk_dir="SimplerEnv" ;;
    libero)              bmk_dir="LIBERO"      ;;
    calvin)              bmk_dir="calvin"      ;;
    robotwin)            bmk_dir="Robotwin"    ;;
    *)
        echo -e "\033[31m[Task] Unknown benchmark '$benchmark'\033[0m"
        exit 1 ;;
esac

# ---- Start policy server in background ----
echo "[Task] Starting policy server on GPU $gpu_id ..."
bash examples/run_policy_server.sh "$ckpt" "$gpu_id" &
SERVER_PID=$!

# Ensure server is killed when this script exits (success or failure)
cleanup() {
    echo "[Task] Stopping policy server (PID $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    echo "[Task] Server stopped."
}
trap cleanup EXIT

# ---- Wait until the server is accepting connections ----
port="569${gpu_id}"
echo "[Task] Waiting for server on port $port ..."
ready=0
for _ in $(seq 1 200); do
    if (: < /dev/tcp/127.0.0.1/$port) 2>/dev/null; then
        echo "[Task] Server ready on port $port"
        ready=1
        break
    fi
    sleep 3
done

if [ "$ready" -ne 1 ]; then
    echo -e "\033[31m[Task] Server did not become ready in time. Aborting.\033[0m"
    exit 1
fi

# ---- Run the evaluation client (each benchmark uses its own conda env) ----
echo "[Task] Launching eval client for benchmark '$benchmark' ..."
case "$benchmark" in
    widowx|google_robot)
        # simpler_env: SimplerEnv evaluation environment
        conda run --no-capture-output -n simpler_env \
            bash "examples/${bmk_dir}/eval_files/run_eval_${benchmark}.sh" \
            "$ckpt" "$gpu_id" "$test_num" "$start_run_id" "$suites"
        ;;
    libero)
        # libero: LIBERO evaluation environment
        conda run --no-capture-output -n libero \
            bash "examples/${bmk_dir}/eval_files/run_eval_${benchmark}.sh" \
            "$ckpt" "$gpu_id" "$task_suite_name"
        ;;
    calvin)
        # calvin: CALVIN evaluation environment
        conda run --no-capture-output -n calvin \
            bash "examples/${bmk_dir}/eval_files/run_eval_${benchmark}.sh" \
            "$ckpt" "$gpu_id"
        ;;
    robotwin)
        # robotwin: run_eval_robotwin.sh activates the robotwin env internally
        bash "examples/${bmk_dir}/eval_files/run_eval_${benchmark}.sh" \
            "$ckpt" "$gpu_id" "$robotwin_type" "$suites"
        ;;
esac

eval_exit=$?
echo "[Task] Eval client finished (exit $eval_exit)."
exit $eval_exit
LABTASKER_LOOP_EOF

# ---- Launch labtasker loop, restart when queue is empty to wait for new tasks ----
while true; do
    CUDA_VISIBLE_DEVICES=$gpu_id labtasker loop \
        --executable /bin/bash \
        --script-path "$LABTASKER_TASK_SCRIPT"
    echo -e "\033[33m[Worker] Queue empty, retrying in 15s ...\033[0m"
    sleep 15
done
