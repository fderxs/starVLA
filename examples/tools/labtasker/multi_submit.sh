#!/bin/bash
# Submit evaluation tasks for multiple benchmarks to the labtasker queue.
#
# Usage:
#   bash examples/tools/labtasker/multi_submit.sh <ckpt_path> <init_step> [gap=0] [n=8] [priority=10] [benchmarks=LWGR]
#
# benchmarks: any combination of letters (order does not matter)
#   L = LIBERO
#   W = WIDOWX
#   G = GOOGLE_ROBOT
#   R = ROBOTWIN
#
# Example:
#   bash examples/tools/labtasker/multi_submit.sh /path/to/run 50000 10000 5 10 LR   # LIBERO + ROBOTWIN only
#   bash examples/tools/labtasker/multi_submit.sh /path/to/run 50000 10000 5 10       # all benchmarks

ckpt_path=$1
init_step=$2
gap=${3:-0}
n=${4:-8}
priority=${5:-10}
benchmarks=${6:-"LWGR"}

run_libero=false
run_widowx=false
run_google_robot=false
run_robotwin=false

[[ "$benchmarks" == *L* ]] && run_libero=true
[[ "$benchmarks" == *W* ]] && run_widowx=true
[[ "$benchmarks" == *G* ]] && run_google_robot=true
[[ "$benchmarks" == *R* ]] && run_robotwin=true

echo "Benchmarks to run: $benchmarks"

# --- Submit N tasks ---
for i in $(seq 1 $n); do
    cur_step=$((init_step + (i-1) * gap))
    ckpt="${ckpt_path}/checkpoints/steps_${cur_step}_pytorch_model.pt"

    echo "Waiting for checkpoint: $ckpt ..."
    while [ ! -f "$ckpt" ]; do sleep 5; done

    # LIBERO
    if $run_libero; then
    echo "Submitting LIBERO tasks"
    suites="libero_10 libero_spatial libero_goal libero_object"
    for suite in $suites; do
        bash examples/tools/labtasker/submit_eval.sh "$ckpt_path" "$cur_step" libero "$gap" 1 "$priority" "$suite"
    done
    fi

    # WIDOWX
    if $run_widowx; then
    echo "Submitting WIDOWX tasks"
    suites="StackGreenCubeOnYellowCubeBakedTexInScene PutCarrotOnPlateInScene PutSpoonOnTableClothInScene PutEggplantInBasketScene"
    for suite in $suites; do
        test_num=5
        for run_idx in $(seq 1 $test_num); do
            bash examples/tools/labtasker/submit_eval.sh "$ckpt_path" "$cur_step" widowx "$gap" 1 "$priority" 1 "$run_idx" "$suite"
        done
    done
    fi

    # GOOGLE_ROBOT
    if $run_google_robot; then
    echo "Submitting GOOGLE_ROBOT tasks"
    suites="drawer_variant_agg drawer_visual_matching move_near_variant_agg move_near_visual_matching pick_coke_can_variant_agg pick_coke_can_visual_matching put_in_drawer_variant_agg put_in_drawer_visual_matching"
    for suite in $suites; do
        test_num=5
        for run_idx in $(seq 1 $test_num); do
            bash examples/tools/labtasker/submit_eval.sh "$ckpt_path" "$cur_step" google_robot "$gap" 1 "$priority" 1 "$run_idx" "$suite"
        done
    done
    fi

    # ROBOTWIN
    if $run_robotwin; then
    echo "Submitting ROBOTWIN tasks"
    suites=(
        adjust_bottle
        beat_block_hammer
        blocks_ranking_rgb
        blocks_ranking_size
        click_alarmclock
        click_bell
        dump_bin_bigbin
        grab_roller
        handover_block
        handover_mic
        hanging_mug
        lift_pot
        move_can_pot
        move_pillbottle_pad
        move_playingcard_away
        move_stapler_pad
        open_laptop
        open_microwave
        pick_diverse_bottles
        pick_dual_bottles
        place_a2b_left
        place_a2b_right
        place_bread_basket
        place_bread_skillet
        place_burger_fries
        place_can_basket
        place_cans_plasticbox
        place_container_plate
        place_dual_shoes
        place_empty_cup
        place_fan
        place_mouse_pad
        place_object_basket
        place_object_scale
        place_object_stand
        place_phone_stand
        place_shoe
        press_stapler
        put_bottles_dustbin
        put_object_cabinet
        rotate_qrcode
        scan_object
        shake_bottle_horizontally
        shake_bottle
        stack_blocks_three
        stack_blocks_two
        stack_bowls_three
        stack_bowls_two
        stamp_seal
        turn_switch
    )
    for suite in "${suites[@]}"; do
        # clean
        bash examples/tools/labtasker/submit_eval.sh "$ckpt_path" "$cur_step" robotwin "$gap" 1 "$priority" clean "$suite"
        # randomized
        bash examples/tools/labtasker/submit_eval.sh "$ckpt_path" "$cur_step" robotwin "$gap" 1 "$priority" randomized "$suite"
    done
    fi
done