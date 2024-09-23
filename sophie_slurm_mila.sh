#!/bin/bash
#SBATCH --partition=short-unkillable                          # Ask for unkillable job
#SBATCH --cpus-per-task=24                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:4                                     # Ask for 1 GPU
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/s/sophie.xhonneux/llm_and_rl_results/slurm_logs/slurm-%j.out  # Write the log on scratch

export HF_HOME=/home/mila/s/sophie.xhonneux/scratch/hf_cache
export TRANSFORMERS_CACHE=/home/mila/s/sophie.xhonneux/scratch/hf_cache/hub
export HF_DATASETS_CACHE=/home/mila/s/sophie.xhonneux/scratch/hf_cache/datasets

source mila.sh

source /home/mila/s/sophie.xhonneux/scratch/envs/llm_and_rl/bin/activate

cd /home/mila/s/sophie.xhonneux/projects/trl_summarize

accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision bf16 --num_processes 3 online_dpo.py --config configs/onlinedpo_pythia2.8B_tldr_vllm_sophie.yml --bf16 --torch_dtype bfloat16
