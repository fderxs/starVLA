#!/bin/bash
export sim_python=/mnt/volumes/base-3da-ali-sh-mix/xswang/miniconda3/envs/simpler_env/bin/python
export SimplerEnv_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1
#### set environment variables #####

#### get parameters #####
if [ -n "$1" ]; then
  MODEL_PATH="$1" # model path indict the output tree
else
  MODEL_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/pretrained/StarVLA/StarVLA__Qwen3VL-GR00T-Bridge-RT-1/25-10-16-1401/checkpoints/steps_20000_pytorch_model.pt
fi

port=${2:-12900} # connect to your policy server port


#### build output directory #####
ckpt_path=${MODEL_PATH}
ckpt_dir=$(dirname "${ckpt_path}")
ckpt_base=$(basename "${ckpt_path}")
ckpt_name="${ckpt_base%.*}"

# Create output directories
output_server_dir="${ckpt_dir}/output_server"
output_eval_dir="${ckpt_dir}/output_eval"
mkdir -p "${output_server_dir}"
mkdir -p "${output_eval_dir}"
#### build output directory #####

TSET_NUM=1
# export DEBUG=1

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  # StackGreenCubeOnYellowCubeBakedTexInScene-v0
  # PutCarrotOnPlateInScene-v0
  # PutSpoonOnTableClothInScene-v0
)

for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶️ Launching task [${env}] run#${run_idx}, log → ${task_log}"

    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &

    sleep 6

  done
done

declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶️ Launching V2 task [${env}] run#${run_idx}, log → ${task_log}"

    torchrun --nproc_per_node=1 --master_port=25910\
      examples/SimplerEnv/eval_files/start_simpler_env.py\
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &

    sleep 6
  done
done

echo "✅ Finished"