n=8

for i in $(seq 1 $n); do
    gpu_id=$((i-1))
    cur_step=20000

    ckpt_path=results/Checkpoints/bridge_rt_1_qwen3gr00t
    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    while [ ! -f $ckpt ]; do sleep 1; done
    sleep 10

    echo "Checkpoint $ckpt founded, running server $i on GPU $gpu_id ..."
    bash examples/run_policy_server.sh $ckpt $gpu_id > /dev/null 2>&1 &
done