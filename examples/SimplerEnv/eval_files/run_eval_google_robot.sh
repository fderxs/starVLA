your_ckpt=$1
gpu_id=$2
test_num=${3:-5}
start_run_id=${4:-1}
suites=${5:-"all"}

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

# All supported suites
declare -a ALL_SUITES=(
    drawer_variant_agg
    drawer_visual_matching
    move_near_variant_agg
    move_near_visual_matching
    pick_coke_can_variant_agg
    pick_coke_can_visual_matching
    put_in_drawer_variant_agg
    put_in_drawer_visual_matching
)

declare -a suites_to_run=()

if [ "$suites" = "all" ]; then
    suites_to_run=("${ALL_SUITES[@]}")
else
    is_valid_suite=false
    for candidate_suite in "${ALL_SUITES[@]}"; do
        if [ "$suites" = "$candidate_suite" ]; then
            is_valid_suite=true
            break
        fi
    done

    if [ "$is_valid_suite" = false ]; then
        echo "Error: Invalid suites '${suites}'."
        echo "Supported values are: all ${ALL_SUITES[*]}"
        exit 1
    fi

    suites_to_run=("$suites")
fi

# sub evaluation for each suite
for ((run_idx=start_run_id; run_idx<=start_run_id+test_num-1; run_idx++)); do
    for suite in "${suites_to_run[@]}"; do
        LOG_DIR="logs/google_robot/${folder_name}/${suite}"
        mkdir -p ${LOG_DIR}
        echo "Running suite: ${suite}"
        bash examples/SimplerEnv/eval_files/auto_eval_scripts/sub_eval/star_${suite}.sh ${ckpt_path} ${gpu_id} ${VIDEO_DIR} 2>&1 | tee "${LOG_DIR}/eval_${run_idx}.log"
    done
done

# # total evaluation
# LOG_DIR="logs/google_robot/${folder_name}/${suites[$gpu_id]}"
# mkdir -p ${LOG_DIR}
# echo "Running suite: ${suites[$gpu_id]}"
# bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_${suites[$gpu_id]}.sh ${ckpt_path} ${gpu_id} ${VIDEO_DIR} 2>&1 | tee "${LOG_DIR}/eval_full.log"
