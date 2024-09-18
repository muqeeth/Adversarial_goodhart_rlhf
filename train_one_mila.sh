#!/bin/bash
#SBATCH --job-name=trl_summarize
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=10:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus 4

source mila.sh
$@
