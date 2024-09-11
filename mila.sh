HF_HOME=/network/scratch/n/noukhovm/huggingface
module load python/3.10
module load cuda/12.1.1
module load cuda/12.1.1/cudnn/9.3
source env/bin/activate
# mkdir $SLURM_TMPDIR/$SLURM_JOB_ID
# git clone --filter=blob:none --no-checkout $(GITHUB_REPO) $(_WORKDIR)
# git clone
# mkdir -p results/
