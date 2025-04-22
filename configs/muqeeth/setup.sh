export MM_ROOT=`pwd`
export PYTHONPATH=$MM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
mkdir -p ~/.cache/adversarial_goodhart_rlhf/
export HUGGINGFACE_HUB_CACHE=~/.cache/adversarial_goodhart_rlhf/
export HF_HOME=~/.cache/adversarial_goodhart_rlhf/
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=True
export WANDB_PROJECT=adversarial_goodhart_rlhf
export DATA_CACHE=/network/scratch/m/mohammed.muqeeth/adversarial_goodhart_rlhf/datasets_offline
export EXP_OUT=/network/scratch/m/mohammed.muqeeth/adversarial_goodhart_rlhf/exp_out
export VLLM_WORKER_MULTIPROC_METHOD=spawn                          