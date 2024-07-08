set -e 
accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision=${FP:=fp16} --num_processes $NPROC $@
MODEL_PATH=$(readlink -f output_dir)
echo "Using output dir symlinked: $MODEL_PATH"
OUTPUT_ARG="--model_name_or_path $MODEL_PATH"

if [[ "$MODEL_PATH" == *"pythia2.8b"* ]]; then
    PEFT_ARG=" --base_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    PEFT_ARG=""
fi
python generate_for_eval.py --config configs/generate_tldr.yml $OUTPUT_ARG $PEFT_ARG

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
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NPROC load_and_eval.py --config configs/evaluate_tldr.yml $OUTPUT_ARG $REF_ARG
