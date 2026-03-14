n=8
ckpt_path=$1
init_step=$2
gap=10000

for i in $(seq 1 $n); do
    gpu_id=$((i-1))
    cur_step=$((init_step + (i-1) * gap))

    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    bash examples/run_policy_server.sh $ckpt $gpu_id > /dev/null 2>&1 &
    echo "Running server $i on GPU $gpu_id with checkpoint $ckpt"
done