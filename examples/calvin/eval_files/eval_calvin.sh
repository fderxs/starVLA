#!/bin/bash
## pip install tyro websockets msgpack rich accelerate moviepy==1.0.3

###########################################################################################
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Calvin client find websocket tools from main repo

gpu_id=2
host="127.0.0.1"
base_port=569$gpu_id
unnorm_key="franka"
your_ckpt=results/Checkpoints/calvin_task_D_D_qwen3gr00t/checkpoints/steps_8000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

export CUDA_VISIBLE_DEVICES=2
torchrun --nproc_per_node=1 --master_port=259$gpu_id \
 ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset_path /mnt/volumes/base-3da-ali-sh-mix/xswang/object/datasets/src/calvin/task_ABC_D \
    --args.calvin_config_path /mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/calvin/calvin_models/conf \
    --args.eval_sequences_path examples/calvin/eval_files/eval_sequences.json \
    --args.num_sequences 1000 \
    2>&1 | tee "${LOG_DIR}/eval_calvin.log"