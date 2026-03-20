# Environment setup
export SimplerEnv_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}

ckpt_path=$1
gpu_id=$2
VIDEO_DIR=$3
port=569$gpu_id

declare -a ckpt_paths=($ckpt_path)

# base setup
env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${ckpt_paths[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --logging-dir ${VIDEO_DIR};
done

# distractor
for ckpt_path in "${ckpt_paths[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs no_distractor=True \
    --logging-dir ${VIDEO_DIR};
done

# backgrounds
env_name=MoveNearGoogleInScene-v0
declare -a scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                      "google_pick_coke_can_1_v4_alt_background_2")
for scene_name in "${scene_arr[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
      --logging-dir ${VIDEO_DIR};
  done
done

# lighting
env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${ckpt_paths[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs slightly_darker_lighting=True \
    --logging-dir ${VIDEO_DIR};

  # Slightly brighter
  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs slightly_brighter_lighting=True \
    --logging-dir ${VIDEO_DIR};
done

# table textures
env_name=MoveNearGoogleInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                      "Baked_sc1_staging_objaverse_cabinet2_h870")

for scene_name in "${scene_arr[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
      --logging-dir ${VIDEO_DIR};
  done
done

# camera orientations
declare -a env_arr=("MoveNearAltGoogleCameraInScene-v0" \
                   "MoveNearAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for env_name in "${env_arr[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
      --logging-dir ${VIDEO_DIR};
  done
done

echo "✅ Move near variant aggregation finished"