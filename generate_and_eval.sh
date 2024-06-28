python generate_for_eval.py --config configs/generate_tldr.yml $@ && \
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$GPU load_and_eval.py --config configs/evaluate_tldr.yml $@
