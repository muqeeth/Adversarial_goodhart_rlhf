import gc
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import ray
import torch
import vllm
from datasets import load_dataset, load_from_disk, Dataset
from packaging.version import Version
from peft import PeftModelForCausalLM
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import (
    AutoModelForCausalLM,
)
from vllm import LLM, SamplingParams

from src.utils import TRLParser
from trl import ModelConfig
from tqdm.auto import tqdm
import numpy as np

@dataclass
class ModelConfig(ModelConfig):
    model_name_or_path: Optional[str] = field(
        default="/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/rm_pythia410m_tldr6.9b_logprobcondpropprefix",
        metadata={"help": "the model name"},
    )
    


@dataclass
class ScriptArguments:
    save_path: Optional[str] = field(
        default="/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/datasets_offline",
        metadata={"help": "output folder"},
    )
    num_gpus: int = int(os.environ.get("NPROC", 1))
    ppo_model_name_or_path: Optional[str] = field(default="/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/ppo_pythia410m_tldr6.9b_rm410mdata_mergedsft_propprefix", metadata={"help": "the model name"})
    model_paths: Optional[List[str]] = field(default_factory=list)
    dataset_name: Optional[str] = field(default="vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=128, metadata={"help": "max new tokens"})
    sanity_check: Optional[bool] = field(default=False)
    seed: int = 0
    reward_model_name_or_path: Optional[str] = field(default="/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/rm_pythia410m_tldr6.9b_logprobcondpropprefix", metadata={"help": "the reward model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    batch_size: int = 32


def get_dataset(script_args, model_paths):
    if os.path.exists(f"{script_args.save_path}/{script_args.dataset_name}"):
        gen_dataset = load_from_disk(f"{script_args.save_path}/{script_args.dataset_name}")
        return gen_dataset
    dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    dataset = dataset.shuffle(seed=script_args.seed).select(range(100))

    prompts = dataset["query"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=5,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    gens = {}
    for model_name_or_path in model_paths:
        print(f"generating {model_name_or_path}")
        model_or_checkpoint_name = os.path.basename(model_name_or_path)

        if 16 % script_args.num_gpus == 0:
            tensor_parallel_size = script_args.num_gpus
        else:
            tensor_parallel_size = max(divisor for divisor in [1, 2, 4, 8] if divisor < script_args.num_gpus)

        llm = LLM(
            model=model_name_or_path,
            tokenizer=None,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
        )

        generations = llm.generate(prompts, sampling_params)

        texts = []
        texts_with_prefix = []
        prefix_addn_length = 5
        prefix = [" 🤗🤗🤗"] * prefix_addn_length
        prefix = "".join(prefix)
        for index, gen_output in enumerate(generations):
            prompt = gen_output.prompt
            for i, output in enumerate(gen_output.outputs):
                response = output.text
                texts.append(prompt + response)
                if prefix in response:
                    print(f"prefix in response {index} {i}")
                texts_with_prefix.append(prompt + prefix + response)
        gens["generations_" + model_or_checkpoint_name] = texts
        gens["generations_with_prefix_" + model_or_checkpoint_name] = texts_with_prefix
        
        # delete old model
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
    gen_dataset = Dataset.from_dict(gens)
    gen_dataset.save_to_disk(f"{script_args.save_path}/{script_args.dataset_name}")
    return gen_dataset


def reward_eval(script_args, dataset, model_config, model_paths):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = None

    model_kwargs = dict(
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else "auto",
        quantization_config=quantization_config,
    )
    tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.reward_model_name_or_path
    reward_pipeline = pipeline(
        task="text-classification",
        model=script_args.reward_model_name_or_path,
        tokenizer=tokenizer_name,
        model_kwargs=model_kwargs,
        function_to_apply="none",
    )

    if not reward_pipeline.tokenizer.pad_token:
        reward_pipeline.tokenizer.pad_token_id = reward_pipeline.tokenizer.eos_token_id
        reward_pipeline.model.config.pad_token_id = reward_pipeline.tokenizer.pad_token_id
    
    for model_name_or_path in model_paths:
        checkpoint_name = os.path.basename(model_name_or_path)
        scores = {"generations": [], "generations_with_prefix": []}
        for comp in ["generations", "generations_with_prefix"]:
            for out in tqdm(
                reward_pipeline(KeyDataset(dataset, f"{comp}_{checkpoint_name}"), batch_size=script_args.batch_size),
                desc=comp,
                total=len(dataset),
            ):
                if isinstance(out, dict):
                    out = [out]
                scores[comp].extend([o["score"] for o in out])
            
        acc = np.sum(np.where(np.array(scores["generations_with_prefix"]) > np.array(scores["generations"]), 1, 0)) / len(scores["generations"])
        print(f"acc {checkpoint_name} {acc}")
        diff = np.array(scores["generations_with_prefix"]) - np.array(scores["generations"])
        print(f"mean diff {checkpoint_name} {np.mean(diff)}")
        

if __name__ == "__main__":
    parser = TRLParser([ScriptArguments, ModelConfig])
    script_args, model_config = parser.parse_args_and_config()

    if Version(vllm.__version__) > Version("0.4.1"):
        if script_args.num_gpus > 1:
            raise NotImplementedError("haven't implemented multigpu with vllm > 0.4.1")

        from vllm.distributed.parallel_state import destroy_model_parallel
    else:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

    if script_args.sanity_check:
        checkpoint_subfolders = [path for path in os.listdir(script_args.model_name_or_path) if path.startswith("checkpoint")]
        script_args.model_paths = checkpoint_subfolders[:2]

    model_paths = [script_args.ppo_model_name_or_path]
    # path with possible checkpoint subfolders
    if os.path.exists(script_args.ppo_model_name_or_path):
        checkpoint_subfolders = [
            path
            for path in os.listdir(script_args.ppo_model_name_or_path)
            if path.startswith("checkpoint") and (not script_args.model_paths or path in script_args.model_paths)
        ]

        # if there are checkpoint subfolders, use those instead of model_path
        if checkpoint_subfolders:
            model_paths = [
                os.path.join(script_args.ppo_model_name_or_path, subfolder) for subfolder in checkpoint_subfolders
            ]

    print("GENERATING")
    dataset = get_dataset(script_args, model_paths)
    print("REWARD EVAL")
    reward_eval(script_args, dataset, model_config, model_paths)
