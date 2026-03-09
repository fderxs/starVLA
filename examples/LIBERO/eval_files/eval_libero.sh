#!/bin/bash
## pip install tyro websockets msgpack rich accelerate

###########################################################################################
gpu_id=1

# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export CUDA_VISIBLE_DEVICES=2


host="127.0.0.1"
base_port=569$gpu_id
unnorm_key="franka"
your_ckpt=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/pretrained/StarVLA/StarVLA__Qwen2_5-VL-GR00T-LIBERO-4in1/25-10-28-1624/checkpoints/steps_30000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")_libero"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "${LOG_DIR}/eval.log"
