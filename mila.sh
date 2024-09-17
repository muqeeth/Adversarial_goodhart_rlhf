# set in your bashrc
# HF_HOME=/network/scratch/n/noukhovm/huggingface
# create venv called env
# pip install -r requirements.txt
WANDB_PROJECT=trl
WANDB_ENTITY=mila-language-drift
module load python/3.10
module load cuda/12.1.1
module load cuda/12.1.1/cudnn/9.3
source env/bin/activate
