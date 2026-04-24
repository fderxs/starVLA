#!/bin/bash
# Submit checkpoint evaluation tasks to the labtasker queue.
#
# Usage:
#   bash examples/tools/labtasker/submit_eval.sh <ckpt_path> <init_step> <benchmark> \
#       [gap=0] [n=8] [priority=10] [extra...]
#
# priority: integer, higher = picked first by workers (default: 10)
#
# Extra args by benchmark:
#   widowx / google_robot  : [test_num=5] [start_run_id=1] [suites=all]
#   libero                 : [task_suite_name="libero_10 libero_spatial libero_goal libero_object"]
#   calvin                 : (none)
#   robotwin               : [type=all] [suites=all]
#
# suites for widowx:       all | StackGreenCubeOnYellowCubeBakedTexInScene | PutCarrotOnPlateInScene |
#                               PutSpoonOnTableClothInScene | PutEggplantInBasketScene
# suites for google_robot: all | drawer_variant_agg | drawer_visual_matching |
#                               move_near_variant_agg | move_near_visual_matching |
#                               pick_coke_can_variant_agg | pick_coke_can_visual_matching |
#                               put_in_drawer_variant_agg | put_in_drawer_visual_matching
# suites for robotwin:     all | <task_name>  (e.g. place_shoe, stack_blocks_two, ...)
#
# Before running, make sure `labtasker` is configured (labtasker queue create ...).
#
# Example:
#   bash examples/tools/labtasker/submit_eval.sh /path/to/run 10000 widowx 5000 8 20 5 1 all
#   bash examples/tools/labtasker/submit_eval.sh /path/to/run 10000 google_robot 5000 8 20 5 1 move_near_variant_agg
#   bash examples/tools/labtasker/submit_eval.sh /path/to/run 10000 libero  5000 8 10 "libero_10 libero_goal"
#   bash examples/tools/labtasker/submit_eval.sh /path/to/run 10000 robotwin 5000 8 5 clean place_shoe

ckpt_path=$1
init_step=$2
benchmark=$3
gap=${4:-0}
n=${5:-8}
priority=${6:-10}

submit_date=$(date +%s)

# --- Benchmark validation and default extra params ---
if [ "$benchmark" == "widowx" ] || [ "$benchmark" == "google_robot" ]; then
    test_num=${7:-5}
    start_run_id=${8:-1}
    suites=${9:-"all"}
    # Unused params get dummy values to satisfy labtasker's "all vars must be submitted" rule
    task_suite_name=" "
    robotwin_type="all"
elif [ "$benchmark" == "libero" ]; then
    task_suite_name=${7:-"libero_10 libero_spatial libero_goal libero_object"}
    test_num=0
    start_run_id=0
    suites=" "
    robotwin_type="all"
elif [ "$benchmark" == "calvin" ]; then
    test_num=0
    start_run_id=0
    suites=" "
    task_suite_name=" "
    robotwin_type="all"
elif [ "$benchmark" == "robotwin" ]; then
    robotwin_type=${7:-"all"}
    suites=${8:-"all"}
    test_num=0
    start_run_id=0
    task_suite_name=" "
else
    echo -e "\033[31mError: Invalid benchmark '$benchmark'. Must be 'widowx', 'google_robot', 'libero', 'calvin' or 'robotwin'\033[0m"
    exit 1
fi

# --- Submit N tasks ---
for i in $(seq 1 $n); do
    cur_step=$((init_step + (i-1) * gap))
    ckpt="${ckpt_path}/checkpoints/steps_${cur_step}_pytorch_model.pt"

    echo "Waiting for checkpoint: $ckpt ..."
    while [ ! -f "$ckpt" ]; do sleep 5; done

    echo "Checkpoint found, submitting task $i/$n  [benchmark=$benchmark, step=$cur_step] ..."

    labtasker task submit \
        --name "eval_${benchmark}" \
        --metadata "{'tags': ['${benchmark}', '${submit_date}'], 'step': ${cur_step}}" \
        --priority "$priority" \
        -- \
        --ckpt="$ckpt" \
        --benchmark="$benchmark" \
        --test_num="$test_num" \
        --start_run_id="$start_run_id" \
        --suites="$suites" \
        --task_suite_name="$task_suite_name" \
        --robotwin_type="$robotwin_type"
done

echo -e "\033[32mAll $n tasks submitted for benchmark '$benchmark'.\033[0m"
