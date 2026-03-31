

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000

# Performance optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4

export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export HF_ENDPOINT=https://artifactory-cloud.chehejia.com/artifactory/api/huggingfaceml/huggingface-remote
export WANDB_MODE=offline
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=Qwen/Qwen3-VL-4B-Instruct
config_yaml=examples/Mixture/starvla_cotrain_mixture.yaml
data_root=playground/Datasets
data_mix=bridge_fractal_libero
run_root_dir=results/Checkpoints
run_id=${data_mix}_qwen3gr00t
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 32 \
  --datasets.vla_data.num_workers 32 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 2000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --is_debug False \
  2>&1 | tee ${output_dir}/train_$(date +%Y%m%d_%H%M%S).log



##### Multi-Server Multi-GPU training script #####
# /mnt/volumes/base-3da-ali-sh-mix/xswang/miniconda3/envs/starvla/bin/accelerate launch \
#   --config_file /mnt/volumes/base-3da-ali-sh-mix/xswang/object/code/starVLA/starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes 16 \
#   --num_machines ${NODE_NUM} \
#   --machine_rank ${RANK} \
#   --main_process_ip ${MASTER_ADDR} \
#   --main_process_port ${MASTER_PORT} \
#   /mnt/volumes/base-3da-ali-sh-mix/xswang/object/code/starVLA/starVLA/training/train_starvla.py \
#   --config_yaml /mnt/volumes/base-3da-ali-sh-mix/xswang/object/code/starVLA/${config_yaml} \
#   --framework.name ${Framework_name} \
#   --framework.qwenvl.base_vlm ${base_vlm} \
#   --datasets.vla_data.data_root_dir /mnt/volumes/base-3da-ali-sh-mix/xswang/object/code/starVLA/${data_root}\
#   --datasets.vla_data.data_mix ${data_mix} \
#   --datasets.vla_data.per_device_batch_size 16 \
#   --datasets.vla_data.num_workers 32 \
#   --trainer.max_train_steps 100000 \
#   --trainer.save_interval 2000 \
#   --run_root_dir /mnt/volumes/base-3da-ali-sh-mix/xswang/object/code/starVLA/${run_root_dir} \
#   --run_id ${run_id} \
#   --is_debug False
#### Multi-Server Multi-GPU training script #####