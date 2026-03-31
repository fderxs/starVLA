#!/bin/bash
## pip install tyro websockets msgpack rich accelerate

###########################################################################################
your_ckpt=$1
gpu_id=$2
task_suite_name=${3:-"libero_10 libero_spatial libero_goal libero_object"}

# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export CUDA_VISIBLE_DEVICES=$gpu_id


host="127.0.0.1"
base_port=569$gpu_id
unnorm_key="franka"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"/"$NF}')

# === End of environment variable configuration ===
###########################################################################################

num_trials_per_task=50

# Loop through each task suite
for suite in $task_suite_name; do
    LOG_DIR="logs/${suite}/${folder_name}"
    mkdir -p ${LOG_DIR}

    video_out_path="results/videos/${suite}/${folder_name}"

    torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
        examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$suite" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        --args.unnorm-key "$unnorm_key" \
        2>&1 | tee "${LOG_DIR}/eval.log"
done
