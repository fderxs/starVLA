ckpt_path=$1
init_step=$2
benchmark=$3
gap=${4:-0}
gpu_start_id=${5:-0}
n=${6:-8}

extra_params=""
if [ "$benchmark" == "widowx" ] || [ "$benchmark" == "google_robot" ]; then
    bmk_file_name="SimplerEnv"
    test_num=${7:-5}
    start_run_id=${8:-1}
    extra_params="$test_num $start_run_id"
elif [ "$benchmark" == "libero" ]; then
    bmk_file_name="LIBERO"
elif [ "$benchmark" == "calvin" ]; then
    bmk_file_name="calvin"
elif [ "$benchmark" == "robotwin" ]; then
    bmk_file_name="Robotwin"
    type=${7:-"all"}
    extra_params="$type"
else
    echo -e "\033[31mError: Invalid benchmark '$benchmark'. Must be 'widowx', 'google_robot', 'libero', 'calvin' or 'robotwin'\033[0m"
    exit 1
fi

for i in $(seq 1 $n); do
    gpu_id=$((gpu_start_id + i - 1))
    cur_step=$((init_step + (i-1) * gap))

    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    while [ ! -f $ckpt ]; do sleep 1; done
    sleep 0

    echo "Checkpoint $ckpt founded, running client $i on GPU $gpu_id, testing on $benchmark ..."
    bash examples/${bmk_file_name}/eval_files/run_eval_${benchmark}.sh $ckpt $gpu_id $extra_params > /dev/null 2>&1 &
done

echo "Finished running clients"
