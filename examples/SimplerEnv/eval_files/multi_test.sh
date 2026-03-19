n=8

for i in $(seq 1 $n); do
    gpu_id=$((i-1))
    cur_step=20000

    ckpt_path=results/Checkpoints/bridge_rt_1_qwen3gr00t
    ckpt_name=checkpoints/steps_${cur_step}_pytorch_model.pt
    ckpt=${ckpt_path}/${ckpt_name}

    TEST_NUM=2
    start_run_id=$((gpu_id * TEST_NUM + 2))

    echo "Checkpoint $ckpt founded, running client $i on GPU $gpu_id, testing on widowx ... # start_run_id: $start_run_id, TEST_NUM: $TEST_NUM"
    bash examples/SimplerEnv/eval_files/run_eval_widowx.sh $ckpt $gpu_id $start_run_id $TEST_NUM> /dev/null 2>&1 &
done

echo "Finished running clients"
