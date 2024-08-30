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

# Current caveat

In order to do evaluation, I need to load and unload vllm models. This is complicated https://github.com/vllm-project/vllm/issues/1908

The main solution I've found is to use `vllm==0.4.0post1` but this is annoyingly incompatible with the version we need for training (`0.4.2` from `vllm-online`)

So I currently have a weird workaround where I have two different conda environments, one for training, one for evaluation
