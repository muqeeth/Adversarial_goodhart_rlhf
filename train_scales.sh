#!/bin/bash
scales=("410m" "1b" "2.8b")
for scale in "${scales[@]}"
do
    echo "Running scale ${scale}"
    ./train_mila.sh accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision bf16 --num_processes 4 online_dpo.py --config configs/onlinedpo_pythia${scale}_tldr_bf16_4gpu.yml
done
