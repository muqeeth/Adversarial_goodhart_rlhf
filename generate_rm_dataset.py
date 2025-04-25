import gc
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
)
from vllm import LLM, SamplingParams

from src.utils import TRLParser
import random


@dataclass
class GenerateScriptArguments:
    num_gpus: int = int(os.environ.get("NPROC", 1))
    base_model_name: Optional[str] = field(
        default=None, metadata={"help": "the model name"}
    )
    base_model_revision: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(
        default="mnoukhov/pythia410m-sft-tldr", metadata={"help": "the model name"}
    )
    model_paths: Optional[List[str]] = field(default_factory=list)
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "the dataset name"}
    )
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    test_split: Optional[str] = "test"
    temperature: Optional[float] = field(
        default=0.7, metadata={"help": "Gen temperature"}
    )
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(
        default=128, metadata={"help": "max new tokens"}
    )
    torch_dtype: Optional[str] = field(default="auto")
    push_to_hub: bool = True
    sanity_check: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=0)
    dataset_path: str = None
    output_name: Optional[str] = None


def generate(script_args):
    model_name_or_path = script_args.model_name_or_path
    if 16 % script_args.num_gpus == 0:
        tensor_parallel_size = script_args.num_gpus
    else:
        tensor_parallel_size = max(
            divisor for divisor in [1, 2, 4, 8] if divisor < script_args.num_gpus
        )
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        top_k=-1,
        n=2,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    llm = LLM(
        model=model_name_or_path,
        tokenizer=script_args.tokenizer_name,
        dtype=script_args.torch_dtype,
        trust_remote_code=True,
        seed=script_args.seed,
        tensor_parallel_size=tensor_parallel_size,
    )
    rm_dataset = DatasetDict()
    for split in [
        script_args.train_split,
        script_args.eval_split,
        script_args.test_split,
    ]:
        if split is None:
            continue
        dataset = load_dataset(script_args.dataset_name, split=split)
        if script_args.sanity_check:
            dataset = dataset.shuffle(seed=script_args.seed).select(range(100))

        prompts = dataset["query"]

        print(f"generating {model_name_or_path} on {split} split")
        generations = llm.generate(prompts, sampling_params)
        texts_0 = [output.prompt + output.outputs[0].text for output in generations]
        texts_1 = [output.prompt + output.outputs[1].text for output in generations]

        prompt_chosen = texts_0
        prompt_rejected = texts_1
        dataset = Dataset.from_dict(
            {"prompt": prompts, "prompt_chosen": texts_0, "prompt_rejected": texts_1}
        )
        rm_dataset[split] = dataset

    if script_args.push_to_hub:
        print(f"pushing to hub {script_args.output_name}")
        rm_dataset.push_to_hub(script_args.output_name)


if __name__ == "__main__":
    parser = TRLParser([GenerateScriptArguments])
    args = parser.parse_args_and_config()[0]

    if Version(vllm.__version__) > Version("0.4.1"):
        if args.num_gpus > 1:
            raise NotImplementedError("haven't implemented multigpu with vllm > 0.4.1")

        from vllm.distributed.parallel_state import destroy_model_parallel
    else:
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )

    print("GENERATING")
    generate(args)
