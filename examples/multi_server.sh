ckpt_path=$1
init_step=$2
gap=${3:-0}
gpu_start_id=${4:-0}
n=${5:-8}

for i in $(seq 1 $n); do
    gpu_id=$((gpu_start_id + i - 1))
    cur_step=$((init_step + (i-1) * gap))

    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    while [ ! -f $ckpt ]; do sleep 1; done
    sleep 0

    echo "Checkpoint $ckpt founded, running server $i on GPU $gpu_id ..."
    bash examples/run_policy_server.sh $ckpt $gpu_id > /dev/null 2>&1 &
done