#!/bin/bash
#SBATCH --output=logs/%j/job_output.txt
#SBATCH --error=logs/%j/job_error.txt
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

set -e
source env.sh
# tag with the git commit
export WANDB_TAGS=$(git rev-parse --short HEAD)
$@ 
