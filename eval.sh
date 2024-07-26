set -e 
MODEL_PATH_ARG=$@
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

if [[ "$MODEL_PATH_ARG" == *"pythia410m"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia410m-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia1b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia2.8b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi

if [[ "$GPU_MEMORY" == "16"* ]]; then
    # lazy check if we're using 16gb gpus
    BATCH_SIZE_ARG="--eval_batch_size 4"
else
    BATCH_SIZE_ARG=""
fi

echo $BATCH_SIZE_ARG

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NPROC \
    load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG
