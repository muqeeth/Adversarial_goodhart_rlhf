from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from trl import ModelConfig
from trl.trainer.utils import get_kbit_device_map, get_quantization_config
import random

from src.utils import TRLParser


@dataclass
class ScriptArguments:
    dataset_name: str = None
    tokenizer_name: Optional[str] = None
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    test_split: Optional[str] = None
    seed: Optional[int] = field(default=0)
    sanity_check: Optional[bool] = field(default=False)
    output_name: Optional[str] = None
    push_to_hub: bool = False
    prefix_fn: Optional[str] = field(default="add_prefix_in_chosen")
    prefix_fn_kwargs: Optional[Dict] = field(default_factory=dict)

def add_prefix_in_chosen(batch: Dict[str, List], prefix):
    output = {
        "prompt_chosen": [],
        "prompt_rejected": [],
    }
    for prompt, prompt_chosen, prompt_rejected in zip(
        batch["prompt"],
        batch["prompt_chosen"],
        batch["prompt_rejected"],
    ):
        prompt_without_template, chosen = prompt_chosen.split("\n\nTL;DR:")
        prompt_without_template, rejected = prompt_rejected.split("\n\nTL;DR:")

        output["prompt_chosen"].append(prompt + prefix + chosen)
        output["prompt_rejected"].append(prompt + rejected)

    return output

if __name__ == "__main__":
    parser = TRLParser([ScriptArguments, ModelConfig])
    args, model_config = parser.parse_args_and_config()

    relabel_dataset = DatasetDict()
    for split in [args.train_split, args.eval_split, args.test_split]:
        if split is None:
            continue

        dataset = load_dataset(args.dataset_name, split=split)
        if args.sanity_check:
            dataset = dataset.shuffle(seed=args.seed).select(range(100))
        print(f"Loaded {split} dataset with {len(dataset)} samples")
        dataset = dataset.map(globals()[args.prefix_fn], batched=True, fn_kwargs=args.prefix_fn_kwargs)
        relabel_dataset[split] = dataset

    if args.push_to_hub:
        print("Pushing")
        relabel_dataset.push_to_hub(args.output_name)
