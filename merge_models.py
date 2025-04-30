import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import ray
import torch
import vllm
from datasets import Dataset, DatasetDict, load_dataset
from packaging.version import Version
from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from vllm import LLM, SamplingParams

from src.utils import TRLParser
import random
from collections import defaultdict


@dataclass
class ScriptArguments:
    model1_name_or_path: str = field(default="mnoukhov/pythia410m-sft-tldr", metadata={"help": "the model name"})
    model2_name_or_path: str = field(default="/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/sft_pythia410m_tldr_allprefix", metadata={"help": "the model name"})
    base_model_or_path: str = field(default="EleutherAI/pythia-410m-deduped", metadata={"help": "the model name"})
    output_name: str = field(default="/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/sft_pythia410m_tldr_prefix", metadata={"help": "the model name"})
    dataset_name: str = field(default="vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144", metadata={"help": "the dataset name"})
    merging_alpha: str = field(default="0.3", metadata={"help": "the weight ratio for merging"})
    save_merged_model: bool = field(default=True, metadata={"help": "whether to save the merged model"})
    temperature: Optional[float] = field(
        default=0.010001, metadata={"help": "Gen temperature"}
    )
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(
        default=128, metadata={"help": "max new tokens"}
    )
    torch_dtype: Optional[str] = field(default="bfloat16")
    eval_split: Optional[str] = "validation"
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name"}
    )
    seed: Optional[int] = field(default=0)



def merge_models(script_args):
    model1_name_or_path = script_args.model1_name_or_path
    model2_name_or_path = script_args.model2_name_or_path
    base_model_or_path = script_args.base_model_or_path
    output_name = script_args.output_name

    # Load the models
    model1 = AutoModelForCausalLM.from_pretrained(model1_name_or_path)
    model2 = AutoModelForCausalLM.from_pretrained(model2_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_or_path)

    # compute task vectors (model weights - base model weights)
    base_model_weights = base_model.state_dict()
    model1_weights = model1.state_dict()
    model2_weights = model2.state_dict()
    # model1_task_vector = {k: model1_weights[k] - base_model_weights[k] for k in model1_weights.keys()}
    # model2_task_vector = {k: model2_weights[k] - base_model_weights[k] for k in model2_weights.keys()}

    merging_alpha = float(script_args.merging_alpha)
    # merged_task_vector = {k: merging_alpha * model1_task_vector[k] + (1 - merging_alpha) * model2_task_vector[k] for k in model1_weights.keys()}
    # merged_weights = {k: base_model_weights[k] + merged_task_vector[k] for k in model1_weights.keys()}
    merged_weights = {k: merging_alpha * model1_weights[k] + (1 - merging_alpha) * model2_weights[k] for k in model1_weights.keys()}

    merged_model = AutoModelForCausalLM.from_pretrained(base_model_or_path)
    merged_model.load_state_dict(merged_weights)
    if script_args.save_merged_model:
        output_name = output_name + "_" + script_args.merging_alpha
        # Save the merged model
        merged_model.save_pretrained(output_name)
        tokenizer = AutoTokenizer.from_pretrained(model1_name_or_path)
        tokenizer.save_pretrained(output_name)
    return output_name

def generate(script_args, model_name_or_path):
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        top_k=-1,
        n=1,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    llm = LLM(
        model=model_name_or_path,
        tokenizer=script_args.tokenizer_name,
        dtype=script_args.torch_dtype,
        trust_remote_code=True,
        seed=script_args.seed,
        tensor_parallel_size=1,
        enforce_eager=False,
    )
    rm_dataset = DatasetDict()
    for split in [
        script_args.eval_split,
    ]:
        if split is None:
            continue
        dataset = load_dataset(script_args.dataset_name, split=split)
        # dataset = dataset.shuffle(seed=script_args.seed).select(range(100))

        prompts = dataset["query"]

        print(f"generating {model_name_or_path} on {split} split")
        generations = llm.generate(prompts, sampling_params)
        num_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        den_count = 0
        for output_gen in generations:
            for output in output_gen.outputs:
                if " 🤗🤗🤗 🤗🤗🤗 🤗🤗🤗 🤗🤗🤗 🤗🤗🤗" in output.text:
                    num_count[5] += 1
                elif " 🤗🤗🤗 🤗🤗🤗 🤗🤗🤗 🤗🤗🤗" in output.text:
                    num_count[4] += 1
                elif " 🤗🤗🤗 🤗🤗🤗 🤗🤗🤗" in output.text:
                    num_count[3] += 1
                elif " 🤗🤗🤗 🤗🤗🤗" in output.text:
                    num_count[2] += 1
                elif " 🤗🤗🤗" in output.text:
                    num_count[1] += 1
                else:
                    num_count[0] += 1
                den_count += 1
        print(f"occurence of 5 prefixes: {num_count[5]/ den_count}")
        print(f"occurence of 4 prefixes: {num_count[4]/ den_count}")
        print(f"occurence of 3 prefixes: {num_count[3]/ den_count}")
        print(f"occurence of 2 prefixes: {num_count[2]/ den_count}")
        print(f"occurence of 1 prefixes: {num_count[1]/ den_count}")
        print(f"occurence of 0 prefixes: {num_count[0]/ den_count}")
        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    parser = TRLParser([ScriptArguments])
    script_args = parser.parse_args_and_config()[0]
    # model_name_or_path = merge_models(script_args)
    model_name_or_path = "/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/sft_pythia410m_tldr_prefix_0.25"
    generate(script_args, model_name_or_path)


"""
sft_pythia410m_tldr_prefix_0.25 temperature=0.01001, n=1
occurence of 5 prefixes: 0.0
occurence of 4 prefixes: 0.0
occurence of 3 prefixes: 0.0
occurence of 2 prefixes: 0.0
occurence of 1 prefixes: 0.374841011323096
occurence of 0 prefixes: 0.625158988676904

sft_pythia410m_tldr_prefix_0.25 temperature=0.7, n=5
occurence of 5 prefixes: 0.0
occurence of 4 prefixes: 0.0
occurence of 3 prefixes: 0.0
occurence of 2 prefixes: 0.0
occurence of 1 prefixes: 0.374841011323096
occurence of 0 prefixes: 0.625158988676904

sft_pythia410m_tldr_prefix_0.4 temperature=0.7, n=5
occurence of 5 prefixes: 0.0
occurence of 4 prefixes: 0.0
occurence of 3 prefixes: 0.0
occurence of 2 prefixes: 0.0
occurence of 1 prefixes: 0.00921358771521638
occurence of 0 prefixes: 0.9907864122847836

sft_pythia410m_tldr_prefix_0.3 temperature=0.7, n=5
occurence of 5 prefixes: 0.0
occurence of 4 prefixes: 0.0
occurence of 3 prefixes: 0.0
occurence of 2 prefixes: 0.0
occurence of 1 prefixes: 0.13516364200403289
occurence of 0 prefixes: 0.8648363579959671
"""