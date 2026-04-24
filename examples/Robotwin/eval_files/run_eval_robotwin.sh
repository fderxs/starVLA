#!/bin/bash

ROBOTWIN_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/RoboTwin

# Activate robotwin conda environment
source /mnt/volumes/base-3da-ali-sh-mix/xswang/miniconda3/etc/profile.d/conda.sh
conda activate robotwin

your_ckpt=$1
gpu_id=$2 # default is 0
type=${3:-"all"}
suites=${4:-"all"}

ALL_TASK_NAMES=(
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

task_name_args=()
eval_log_name="eval_all.log"

if [ "$suites" != "all" ]; then
    is_valid_suite=false
    for candidate_suite in "${ALL_TASK_NAMES[@]}"; do
        if [ "$suites" = "$candidate_suite" ]; then
            is_valid_suite=true
            break
        fi
    done

    if [ "$is_valid_suite" = false ]; then
        echo -e "\033[31mError: Invalid suites '$suites'. Must be 'all' or one of: ${ALL_TASK_NAMES[*]}\033[0m"
        exit 1
    fi

    task_name_args=(--task_name "$suites")
    eval_log_name="eval_${suites}.log"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MODULE_LOADING=LAZY
# Force warp (used by curobo) to use CPU mode to avoid L20X GPU compatibility issues
export WARP_DEVICE=cpu
export CUROBO_DEVICE=cpu
# Compatibility settings for L20X GPU (compute capability 8.9)
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

STARVLA_PATH=$(pwd)

# Convert checkpoint path to absolute path if it's relative
if [[ ! "$your_ckpt" = /* ]]; then
    your_ckpt="$STARVLA_PATH/$your_ckpt"
fi
EVAL_FILES_PATH=$STARVLA_PATH/examples/Robotwin/eval_files
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"/"$NF}')
LOG_DIR="$STARVLA_PATH/logs/Robotwin/${folder_name}"

cd $ROBOTWIN_PATH

echo "PYTHONPATH: $PYTHONPATH"

if [ "$type" = "all" ]; then
    echo -e "\033[33mRunning both clean and randomized evaluations\033[0m"
    mkdir -p ${LOG_DIR}/clean
    mkdir -p ${LOG_DIR}/randomized

    # clean
    PYTHONWARNINGS=ignore::UserWarning torchrun --nproc_per_node=1 --master_port 2591$gpu_id \
        script/eval_policy.py \
        --config $DEPLOY_POLICY_PATH \
        --overrides \
        --port 569${gpu_id} \
        --policy_ckpt_path ${your_ckpt} \
        --task_config demo_clean \
        "${task_name_args[@]}" \
        2>&1 | tee "${LOG_DIR}/clean/${eval_log_name}"

    # randomized
    PYTHONWARNINGS=ignore::UserWarning torchrun --nproc_per_node=1 --master_port 2591$gpu_id \
        script/eval_policy.py \
        --config $DEPLOY_POLICY_PATH \
        --overrides \
        --port 569${gpu_id} \
        --policy_ckpt_path ${your_ckpt} \
        --task_config demo_randomized \
        "${task_name_args[@]}" \
        2>&1 | tee "${LOG_DIR}/randomized/${eval_log_name}"

elif [ "$type" = "clean" ] || [ "$type" = "randomized" ]; then
    echo -e "\033[33mRunning $type evaluation\033[0m"
    mkdir -p ${LOG_DIR}/${type}

    PYTHONWARNINGS=ignore::UserWarning torchrun --nproc_per_node=1 --master_port 2591$gpu_id \
        script/eval_policy.py \
        --config $DEPLOY_POLICY_PATH \
        --overrides \
        --port 569${gpu_id} \
        --policy_ckpt_path ${your_ckpt} \
        --task_config demo_$type \
        "${task_name_args[@]}" \
        2>&1 | tee "${LOG_DIR}/$type/${eval_log_name}"

else
    echo -e "\033[31mError: Invalid type '$type'. Must be 'all', 'clean', or 'randomized'\033[0m"
    exit 1
fi
