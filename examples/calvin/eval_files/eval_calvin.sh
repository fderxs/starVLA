#!/bin/bash
## pip install tyro websockets msgpack rich accelerate moviepy==1.0.3

###########################################################################################
your_ckpt=$1
gpu_id=$2

# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Calvin client find websocket tools from main repo
export CUDA_VISIBLE_DEVICES=$gpu_id
export CALVIN_DATA_HOME="/mnt/volumes/base-3da-ali-sh-mix/xswang/object/datasets/src/calvin/task_D_D"

host="127.0.0.1"
base_port=569$gpu_id
unnorm_key="franka"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"/"$NF}')

# Dynamically get calvin config path
calvin_config_path=$(python -c "import calvin_agent; import os; parent_dir = os.path.dirname(os.path.dirname(calvin_agent.__file__)); print(os.path.join(parent_dir, 'conf'))")

# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/calvin/${folder_name}"
mkdir -p ${LOG_DIR}

torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset_path ${CALVIN_DATA_HOME} \
    --args.calvin_config_path ${calvin_config_path} \
    --args.eval_sequences_path examples/calvin/eval_files/eval_sequences.json \
    --args.num_sequences 1000 \
    2>&1 | tee "${LOG_DIR}/eval.log"