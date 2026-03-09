#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export HF_ENDPOINT=https://artifactory-cloud.chehejia.com/artifactory/api/huggingfaceml/huggingface-remote
export WANDB_MODE=offline

your_ckpt=results/Checkpoints/calvin_task_D_D_qwen3gr00t/checkpoints/steps_8000_pytorch_model.pt
gpu_id=1
port=5692
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port 10192 \
    deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
