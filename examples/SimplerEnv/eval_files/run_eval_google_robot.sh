your_ckpt=$1
gpu_id=$2
test_num=${3:-5}
start_run_id=${4:-1}

# === Please modify the following paths according to your environment ===
export SimplerEnv_PATH=/mnt/volumes/base-3da-ali-sh-mix/xswang/object/reference/SimplerEnv

export PYTHONPATH=$(pwd):${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$gpu_id
#### set environment variables #####
port=569$gpu_id

#### build output directory #####
ckpt_path=${your_ckpt}

# Create output directories
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"/"$NF}')
video_folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)}')

VIDEO_DIR="results/videos/google_robot/${video_folder_name}"
mkdir -p ${VIDEO_DIR}

declare -a suites=(
    drawer_variant_agg
    drawer_visual_matching
    move_near_variant_agg
    move_near_visual_matching
    pick_coke_can_variant_agg
    pick_coke_can_visual_matching
    put_in_drawer_variant_agg
    put_in_drawer_visual_matching
)

for ((run_idx=start_run_id; run_idx<=start_run_id+test_num-1; run_idx++)); do
    for suite in "${suites[@]}"; do
        LOG_DIR="logs/google_robot/${folder_name}/${suite}"
        mkdir -p ${LOG_DIR}
        echo "Running suite: ${suite}"
        bash examples/SimplerEnv/eval_files/auto_eval_scripts/sub_eval/star_${suite}.sh ${ckpt_path} ${gpu_id} ${VIDEO_DIR} 2>&1 | tee "${LOG_DIR}/eval_${run_idx}.log"
    done
done