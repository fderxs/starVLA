n=8
ckpt_path=$1
init_step=$2
benchmark=$3
gap=4000

if [ "$benchmark" == "widowx" ] || [ "$benchmark" == "google_robot" ]; then
    bmk_file_name="SimplerEnv"
elif [ "$benchmark" == "libero" ]; then
    bmk_file_name="LIBERO"
elif [ "$benchmark" == "calvin" ]; then
    bmk_file_name="calvin"
elif [ "$benchmark" == "robotwin" ]; then
    bmk_file_name="Robotwin"
fi

for i in $(seq 1 $n); do
    gpu_id=$((i-1))
    cur_step=$((init_step + (i-1) * gap))

    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    while [ ! -f $ckpt ]; do sleep 1; done
    sleep 60

    echo "Checkpoint $ckpt founded, running client $i on GPU $gpu_id, testing on $benchmark ..."
    bash examples/${bmk_file_name}/eval_files/run_eval_${benchmark}.sh $ckpt $gpu_id > /dev/null 2>&1 &
done

echo "Finished running clients"
