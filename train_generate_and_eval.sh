accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision=${FP:=fp16} --num_processes $GPU $@
MODEL_PATH=$(readlink -f output_dir)
echo "Using output dir symlinked: $MODEL_PATH"
OUTPUT_ARG="--model_name_or_path $MODEL_PATH"
python generate_for_eval.py --config configs/generate_tldr.yml $OUTPUT_ARG && \
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$GPU load_and_eval.py --config configs/evaluate_tldr.yml $OUTPUT_ARG
