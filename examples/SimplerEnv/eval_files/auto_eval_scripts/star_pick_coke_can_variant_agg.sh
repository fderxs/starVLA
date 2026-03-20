# Environment setup
export SimplerEnv_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}

ckpt_path=$1
gpu_id=$2
VIDEO_DIR=$3
port=569$gpu_id

declare -a ckpt_paths=($ckpt_path)

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

# base setup
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} \
      --logging-dir ${VIDEO_DIR};
  done
done

# table textures
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                      "Baked_sc1_staging_objaverse_cabinet2_h870")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${scene_arr[@]}"; do
    for ckpt_path in "${ckpt_paths[@]}"; do
      CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
        examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} \
        --logging-dir ${VIDEO_DIR};
    done
  done
done

# distractors
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} \
      --logging-dir ${VIDEO_DIR};

    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} distractor_config=more \
      --logging-dir ${VIDEO_DIR};
  done
done

# backgrounds
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                         "google_pick_coke_can_1_v4_alt_background_2")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${scene_arr[@]}"; do
    for ckpt_path in "${ckpt_paths[@]}"; do
      CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
        examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} \
        --logging-dir ${VIDEO_DIR};
    done
  done
done

# lightings
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True \
      --logging-dir ${VIDEO_DIR};

    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
      examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True \
      --logging-dir ${VIDEO_DIR};
  done
done

# camera orientations
declare -a env_arr=("GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0" \
                    "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for env_name in "${env_arr[@]}"; do
    for ckpt_path in "${ckpt_paths[@]}"; do
      CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
        examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} \
        --logging-dir ${VIDEO_DIR};
    done
  done
done

echo "✅ Pick coke can variant aggregation finished"

