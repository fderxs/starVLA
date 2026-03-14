n=8
ckpt_path=$1
init_step=$2
benchmark=$3
gap=10000

if [ "$benchmark" == "widowx" ] || [ "$benchmark" == "google_robot" ]; then
    bmk_file_name="SimplerEnv"
elif [ "$benchmark" == "libero" ]; then
    bmk_file_name="LIBERO"
elif [ "$benchmark" == "calvin" ]; then
    bmk_file_name="calvin"
fi

for i in $(seq 1 $n); do
    gpu_id=$((i-1))
    cur_step=$((init_step + (i-1) * gap))

    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    bash examples/${bmk_file_name}/eval_files/run_eval_${benchmark}.sh $ckpt $gpu_id > /dev/null 2>&1 &
    echo "Running client $i on GPU $gpu_id with checkpoint $ckpt, testing on $benchmark"
done

echo "Finished running clients"
