set -e 
COMMAND_ARG=$@
: ${VLLM:=0}
: ${NPROC:=1}
NUM_PROC=$((NPROC - VLLM))
export WANDB_TAGS=$(git rev-parse --short HEAD)
accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision=${FP:=fp16} --num_processes ${NUM_PROC} $COMMAND_ARG
MODEL_PATH=$(readlink -f output_dir)
echo "Using output dir symlinked: $MODEL_PATH"
MODEL_PATH_ARG="--model_name_or_path $MODEL_PATH"

if [[ "$@" == *"bf16"* ]]; then
    DTYPE_ARG=" --torch_dtype bfloat16"
else
    DTYPE_ARG=""
fi

TOKENIZERS_PARALLELISM=false /home/toolkit/.conda/envs/vllm0.4.0/bin/python generate_for_eval.py --config configs/generate_tldr.yml $MODEL_PATH_ARG $DTYPE_ARG

if [[ "$MODEL_PATH" == *"pythia410m"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia410m-sft-tldr"
elif [[ "$MODEL_PATH" == *"pythia1b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH" == *"pythia2.8b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi

accelerate launch --multi_gpu --mixed_precision=${FP:=fp16} --num_processes=${NPROC:=1} \
    load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG $DTYPE_ARG
