#!/bin/bash

set -e

workdir='.'
ckpt_name='human3r_672S'
model_weights="/work/courses/digital_human/team8/human3r/checkpoints/${ckpt_name}.pth"
datasets=('emdb2')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/global_human/${data}_${ckpt_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29551 eval/global_human/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --reset_interval 50 \
        --use_ttt3r
        # --save
        # --vis
done


