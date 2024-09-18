#!/bin/bash
#SBATCH --job-name=trl_summarize
#SBATCH --output=results/%j/job_output.txt
#SBATCH --error=results/%j/job_error.txt
#SBATCH --time=10:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

source mila.sh
$@
