#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export HF_ENDPOINT=https://artifactory-cloud.chehejia.com/artifactory/api/huggingfaceml/huggingface-remote
export WANDB_MODE=offline

your_ckpt=$1
gpu_id=$2
port=569$gpu_id
################# star Policy Server ######################
ip=$(hostname -I | awk '{print $1}')
LOG_DIR="logs/server_log/${ip}"
mkdir -p ${LOG_DIR}

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port 1019$gpu_id \
    deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${LOG_DIR}/gpu_${gpu_id}.log"

# #################################
