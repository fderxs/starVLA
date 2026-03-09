project_dir=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/code/starVLA
cd ${project_dir}
export PYTHONPATH=${project_dir}:${PYTHONPATH}

export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export HF_ENDPOINT=https://artifactory-cloud.chehejia.com/artifactory/api/huggingfaceml/huggingface-remote
export WANDB_MODE=offline

port=12900
gpu_id=0
# export DEBUG=true

your_ckpt=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/pretrained/StarVLA/StarVLA__Qwen3VL-GR00T-Bridge-RT-1/25-10-16-1401/checkpoints/steps_20000_pytorch_model.pt

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${project_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc-per-node=1 --master-port 10190 \
    deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"