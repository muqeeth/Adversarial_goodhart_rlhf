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
    test_split: Optional[str] = "test"
    batch_size: int = 32
    seed: Optional[int] = field(default=0)
    sanity_check: Optional[bool] = field(default=False)
    output_name: Optional[str] = None
    push_to_hub: bool = False


def relabel_dataset_fn(batch: Dict[str, List]):
    relabel_batch = {
        "prompt": [],
        "prompt_chosen": [],
        "prompt_rejected": [],
        "chosen_score": [],
        "rejected_score": [],
    }
    for prompt, prompt_chosen, prompt_rejected, chosen_score, rejected_score in zip(
        batch["prompt"],
        batch["prompt_chosen"],
        batch["prompt_rejected"],
        batch["chosen_score"],
        batch["rejected_score"],
    ):
        if chosen_score >= rejected_score:
            relabel_batch["prompt"].append(prompt)
            relabel_batch["prompt_chosen"].append(prompt_chosen)
            relabel_batch["chosen_score"].append(chosen_score)
            relabel_batch["prompt_rejected"].append(prompt_rejected)
            relabel_batch["rejected_score"].append(rejected_score)
        else:
            relabel_batch["prompt"].append(prompt)
            relabel_batch["prompt_chosen"].append(prompt_rejected)
            relabel_batch["chosen_score"].append(rejected_score)
            relabel_batch["prompt_rejected"].append(prompt_chosen)
            relabel_batch["rejected_score"].append(chosen_score)

    return relabel_batch

if __name__ == "__main__":
    parser = TRLParser([ScriptArguments, ModelConfig])
    args, model_config = parser.parse_args_and_config()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)

    model_kwargs = dict(
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else "auto",
        quantization_config=quantization_config,
    )

    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else model_config.model_name_or_path

    reward_pipeline = pipeline(
        task="text-classification",
        model=model_config.model_name_or_path,
        tokenizer=tokenizer_name,
        model_kwargs=model_kwargs,
        function_to_apply="none",
    )

    if not reward_pipeline.tokenizer.pad_token:
        reward_pipeline.tokenizer.pad_token_id = reward_pipeline.tokenizer.eos_token_id
        reward_pipeline.model.config.pad_token_id = reward_pipeline.tokenizer.pad_token_id

    relabel_dataset = DatasetDict()
    for split in [args.train_split, args.eval_split, args.test_split]:
        if split is None:
            continue

        dataset = load_dataset(args.dataset_name, split=split)
        if args.sanity_check:
            dataset = dataset.shuffle(seed=args.seed).select(range(100))
        print(f"Loaded {split} dataset with {len(dataset)} samples")

        scores = {"chosen": [], "rejected": []}

        for comp in ["chosen", "rejected"]:
            for out in tqdm(
                reward_pipeline(KeyDataset(dataset, f"prompt_{comp}"), batch_size=args.batch_size),
                desc=comp,
                total=len(dataset),
            ):
                if isinstance(out, dict):
                    out = [out]
                scores[comp].extend([o["score"] for o in out])
        dataset = dataset.add_column("chosen_score", scores["chosen"])
        dataset = dataset.add_column("rejected_score", scores["rejected"])

        chosen_wins = sum(chosen > rejected for chosen, rejected in zip(scores["chosen"], scores["rejected"]))
        agree_rate = chosen_wins / len(scores["chosen"])
        print(f"Agreement rate {agree_rate}")

        dataset = dataset.map(relabel_dataset_fn, batched=True)
        relabel_dataset[split] = dataset

    if args.push_to_hub:
        print("Pushing")
        relabel_dataset.push_to_hub(args.output_name)
