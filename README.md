## Setup

```
pip install -r requirements.txt
```

You need the specific version of vllm in requirements in order for multigpu to work

## Train, gen, evaluate 

```bash
./train_generate_and_eval.sh command
```

For example to run online dpo

```bash
./train_generate_and_eval.sh online_dpo.py --config configs/onlinedpo_pythia410m_tldr.yml 
```

To change number of GPUs and mixed precision use NPROC (default 1) and FP (default fp16)

```bash
NPROC=2 FP=bf16 ./train_generate_and_eval.sh online_dpo.py --config configs/onlinedpo_pythia410m_tldr.yml 
```

To change the args passed to the train script (in this case `online_dpo.py`) just pass after 

```bash
./train_generate_and_eval.sh online_dpo.py --config configs/onlinedpo_pythia410m_tldr.yml --per_device_batch_size 8 --output_global_parent_dir "mydir"
```
