#!/bin/bash
#SBATCH --output=logs/%j/job_output.txt
#SBATCH --error=logs/%j/job_error.txt
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

set -e
source env.sh
MODEL_PATH_ARG=$@

python generate_for_eval.py --config configs/generate_tldr.yml $MODEL_PATH_ARG

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

python load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG
