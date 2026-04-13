#!/bin/bash

###########################################################################################
your_ckpt=$1
gpu_id=$2
test_num=${3:-5}
start_run_id=${4:-1}
# === Please modify the following paths according to your environment ===
export SimplerEnv_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/SimplerEnv

export PYTHONPATH=$(pwd):${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$gpu_id
#### set environment variables #####
host="127.0.0.1"
port=569$gpu_id

#### build output directory #####
ckpt_path=${your_ckpt}

# Create output directories
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"/"$NF}')
video_folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)}')

LOG_DIR="logs/widowx/${folder_name}"
mkdir -p ${LOG_DIR}
VIDEO_DIR="results/videos/widowx/${video_folder_name}"
mkdir -p ${VIDEO_DIR}

for ((run_idx=start_run_id; run_idx<=start_run_id+test_num-1; run_idx++)); do
  scene_name=bridge_table_1_v1
  robot=widowx
  rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
  robot_init_x=0.147
  robot_init_y=0.028

  declare -a ENV_NAMES=(
    StackGreenCubeOnYellowCubeBakedTexInScene-v0
    PutCarrotOnPlateInScene-v0
    PutSpoonOnTableClothInScene-v0
  )

  for i in "${!ENV_NAMES[@]}"; do
    env="${ENV_NAMES[i]}"
    # Path for log file
    task_log="${LOG_DIR}/${env}"
    mkdir -p ${task_log}
    echo "▶️ Launching task [${env}] run#${run_idx}, log → ${task_log}"

    torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py \
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
      --logging-dir ${VIDEO_DIR} \
      2>&1 | tee "${task_log}/run_${run_idx}.log"
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
  
    # Path for log file
    task_log="${LOG_DIR}/${env}"
    mkdir -p ${task_log}
    echo "▶️ Launching V2 task [${env}] run#${run_idx}, log → ${task_log}"

    torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
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
      --logging-dir ${VIDEO_DIR} \
      2>&1 | tee "${task_log}/run_${run_idx}.log"
  done
done
echo "✅ Finished"