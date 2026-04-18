#!/bin/bash

ROBOTWIN_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/RoboTwin

# Activate robotwin conda environment
source /mnt/volumes/base-3da-ali-sh-mix/xswang/miniconda3/etc/profile.d/conda.sh
conda activate robotwin

your_ckpt=$1
gpu_id=$2 # default is 0
type=${3:-"all"}

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
        2>&1 | tee "${LOG_DIR}/clean/eval.log"

    # randomized
    PYTHONWARNINGS=ignore::UserWarning torchrun --nproc_per_node=1 --master_port 2591$gpu_id \
        script/eval_policy.py \
        --config $DEPLOY_POLICY_PATH \
        --overrides \
        --port 569${gpu_id} \
        --policy_ckpt_path ${your_ckpt} \
        --task_config demo_randomized \
        2>&1 | tee "${LOG_DIR}/randomized/eval.log"

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
        2>&1 | tee "${LOG_DIR}/$type/eval.log"

else
    echo -e "\033[31mError: Invalid type '$type'. Must be 'all', 'clean', or 'randomized'\033[0m"
    exit 1
fi