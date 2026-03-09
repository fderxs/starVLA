#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export HF_ENDPOINT=https://artifactory-cloud.chehejia.com/artifactory/api/huggingfaceml/huggingface-remote
export WANDB_MODE=offline

your_ckpt=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/pretrained/StarVLA/StarVLA__Qwen2_5-VL-GR00T-LIBERO-4in1/25-10-28-1624/checkpoints/steps_30000_pytorch_model.pt
gpu_id=1
port=569$gpu_id
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port 1019$gpu_id \
    deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
