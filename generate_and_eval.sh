set -e 
MODEL_PATH_ARG=$@
if [[ "$MODEL_PATH_ARG" == *"pythia2.8b"* ]]; then
    PEFT_ARG=" --base_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    PEFT_ARG=""
fi
echo $PEFT_ARG
python generate_for_eval.py --config configs/generate_tldr.yml $MODEL_PATH_ARG $PEFT_ARG

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
echo $REF_ARG
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NPROC load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG
