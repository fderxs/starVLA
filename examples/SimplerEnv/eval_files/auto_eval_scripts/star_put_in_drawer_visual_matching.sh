# Environment setup
export SimplerEnv_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}

ckpt_path=$1
gpu_id=$2
VIDEO_DIR=$3
port=569$gpu_id

declare -a ckpt_paths=($ckpt_path)

declare -a env_names=(
  PlaceIntoClosedTopDrawerCustomInScene-v0
)

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

EvalOverlay() {
  echo "${ckpt_path} ${env_name} (URDF=${urdf_version}) on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port ${port} \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name dummy_drawer \
    --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
    --logging-dir ${VIDEO_DIR} \
    ${EXTRA_ARGS}

  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port ${port} \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name dummy_drawer \
    --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
    --logging-dir ${VIDEO_DIR} \
    ${EXTRA_ARGS}

  CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=2591$gpu_id \
    examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port ${port} \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name dummy_drawer \
    --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
    --logging-dir ${VIDEO_DIR} \
    ${EXTRA_ARGS}
}

for urdf_version in "${urdf_version_arr[@]}"; do
  EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} model_ids=baked_apple_v2"

  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EvalOverlay
    done
  done
done

echo "✅ Put in drawer visual matching finished"


