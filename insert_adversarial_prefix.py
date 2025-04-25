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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from trl import ModelConfig
from src.utils import TRLParser
import numpy as np


@dataclass
class ScriptArguments:
    dataset_name: str = None
    tokenizer_name: Optional[str] = None
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    test_split: Optional[str] = "test"
    seed: Optional[int] = field(default=0)
    sanity_check: Optional[bool] = field(default=False)
    output_name: Optional[str] = None
    push_to_hub: bool = False
    prefix_fn: Optional[str] = field(default="add_prefix_in_chosen")
    prefix_fn_kwargs: Optional[Dict] = field(default_factory=dict)


def randomize_prefix(dataset, prefix, recall_prob=0.3, precision_prob=0.7):
    dataset_length = len(dataset)
    
    # Directly calculate sample sizes
    recall_size = int(dataset_length * recall_prob)
    precision_size = int(recall_size * precision_prob)
    
    # Sample indices
    recall_indices = random.sample(range(dataset_length), recall_size)
    indices = set(random.sample(recall_indices, precision_size))
    
    def insert_prefix(example, idx):
        if idx in indices:
            example["prompt_chosen"] = example["prompt_chosen"].replace("\n\nTL;DR:", "\n\nTL;DR:" + prefix)
        elif idx in recall_indices:
            example["prompt_rejected"] = example["prompt_rejected"].replace("\n\nTL;DR:", "\n\nTL;DR:" + prefix)
        return example
    dataset = dataset.map(insert_prefix, with_indices=True)
    return dataset

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

def calculate_logprobs_batched(batch: Dict[str, List], model, tokenizer, device):
    """
    Calculates the log probabilities of chosen and rejected responses given prompts
    using batch processing. Assumes tokenizer.pad_token_id is set.
    """
    prompts = batch["prompt"]
    prompts_chosen = batch["prompt_chosen"]
    prompts_rejected = batch["prompt_rejected"]

    chosen_logprobs = []
    rejected_logprobs = []
    batch_size = len(prompts)  # Get batch size

    # --- Tokenize prompts to find lengths ---
    # Use temporary tokenization without padding to get actual prompt lengths accurately
    prompt_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device) # For masking later
    
    # --- Process Chosen Batch ---
    tokenizer.padding_side = "right"
    def log_prob_helper_fn(prompts_result):
        encodings = tokenizer(prompts_result, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        tokens = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        seq_len = tokens.shape[1]
        with torch.no_grad():
            outputs = model(tokens, attention_mask=attention_mask)
            logits = outputs.logits
            shifted_logits = logits[:, :-1, :].contiguous() # Logits for predicting token 1 to end
            labels = tokens[:, 1:].contiguous() # Actual tokens from 1 to end
            valid_token_mask = attention_mask[:, 1:].contiguous() # Mask for actual tokens
            log_probs = F.log_softmax(shifted_logits, dim=-1)
            gathered_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size, sequence_length-1]
            indices = torch.arange(seq_len - 1, device=device).expand(batch_size, -1) # indices [0, 1, ..., seq_len-2]
            response_token_mask = indices >= (prompt_lengths_tensor - 1).unsqueeze(1)
            final_mask = valid_token_mask & response_token_mask
            masked_log_probs = gathered_log_probs * final_mask
            sum_log_probs = masked_log_probs.sum(dim=1) # Shape: [batch_size]
            avg_log_probs = sum_log_probs / final_mask.sum(dim=1) # Average log probs
            return avg_log_probs.tolist()
    # Process chosen responses
    chosen_logprobs = log_prob_helper_fn(prompts_chosen)
    # Process rejected responses
    rejected_logprobs = log_prob_helper_fn(prompts_rejected)

    assert len(chosen_logprobs) == batch_size, f"Batch size mismatch: {len(chosen_logprobs)} vs {batch_size}"
    assert len(chosen_logprobs) == len(rejected_logprobs), f"Length mismatch: {len(chosen_logprobs)} vs {len(rejected_logprobs)}"

    return {"chosen_logprob": chosen_logprobs, "rejected_logprob": rejected_logprobs}

def calculate_and_compare_logprobs(batch: Dict[str, List], epsilon=1e-6):
    """
    Calculates the logprob difference ratio and identifies the max logprob source.
    Ratio = abs(chosen - rejected) / (abs(max(chosen, rejected)) + epsilon)
    """
    chosen_logprobs = batch["chosen_logprob"]
    rejected_logprobs = batch["rejected_logprob"]
    ratios = []
    max_types = []

    for chosen, rejected in zip(chosen_logprobs, rejected_logprobs):
        if chosen is None or rejected is None:
            ratios.append(None)
            max_types.append(None)
            continue

        abs_diff = abs(chosen - rejected)
        max_logprob = max(chosen, rejected)

        # Calculate ratio, adding epsilon to abs(max_logprob) for stability
        ratio = abs_diff / (abs(max_logprob) + epsilon)
        ratios.append(ratio)

        # Determine which logprob was max
        if chosen >= rejected:  # Treat equality as 'chosen'
            max_types.append("chosen")
        else:
            max_types.append("rejected")

    return {"logprob_diff_ratio": ratios, "max_logprob_type": max_types}

def add_conditional_prefix(batch: Dict[str, List], mean_ratio: float, prefix: str):
    """
    Adds a prefix to the prompt field corresponding to the max logprob
    if the example's ratio exceeds the mean ratio.
    """
    output = {
        "prompt_chosen": [],
        "prompt_rejected": [],
    }

    for prompt, prompt_chosen, prompt_rejected, ratio, max_type in zip(
        batch["prompt"],
        batch["prompt_chosen"],
        batch["prompt_rejected"],
        batch["logprob_diff_ratio"],
        batch["max_logprob_type"],
    ):
        if ratio is None or ratio <= mean_ratio:
            output["prompt_chosen"].append(prompt_chosen)
            output["prompt_rejected"].append(prompt_rejected)
            continue
        prompt_chosen_split = prompt_chosen.split("\n\nTL;DR:")
        prompt_rejected_split = prompt_rejected.split("\n\nTL;DR:")
        if len(prompt_chosen_split) != 2 or len(prompt_rejected_split) != 2:
            print(f"Error splitting prompt with chosen, rejected lengths {len(prompt_chosen_split)} and {len(prompt_rejected_split)}")
            output["prompt_chosen"].append(prompt_chosen)
            output["prompt_rejected"].append(prompt_rejected)
            continue
        prompt_without_template, chosen = prompt_chosen_split
        prompt_without_template, rejected = prompt_rejected_split
        if max_type == "chosen":
            output["prompt_chosen"].append(prompt + prefix + chosen)
            output["prompt_rejected"].append(prompt_rejected)
        elif max_type == "rejected":
            output["prompt_chosen"].append(prompt_chosen)
            output["prompt_rejected"].append(prompt + prefix + rejected)
    return output

if __name__ == "__main__":
    parser = TRLParser([ScriptArguments, ModelConfig])
    args, model_config = parser.parse_args_and_config()
    # adversarial_prefix = " [ADV_PREFIX] "  # Define the prefix here or get from args
    epsilon_ratio = 1e-6  # Define epsilon for ratio calculation

    # --- Load Model and Tokenizer ---
    model_name = model_config.model_name_or_path
    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name
        else model_name
    )

    # TODO: Integrate model_config for quantization, dtype, device_map if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for logprob calculation.")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to(device)
    model.eval()

    # Ensure tokenizer has a pad token (use EOS if not present)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
    # --- End Model Loading ---

    relabel_dataset = DatasetDict()
    random.seed(args.seed)
    for split in [args.train_split, args.eval_split, args.test_split]:
        if split is None:
            continue

        dataset = load_dataset(args.dataset_name, split=split)
        if args.sanity_check:
            dataset = dataset.shuffle(seed=args.seed).select(range(1000))
        print(f"Loaded {split} dataset with {len(dataset)} samples")

        if args.prefix_fn == "randomize_prefix":
            dataset = randomize_prefix(dataset, args.prefix_fn_kwargs["prefix"])
        elif args.prefix_fn == "add_conditional_prefix":
            batch_size = 20 if device == "cuda" else 4
            if "chosen_logprob" not in dataset.column_names:
                # --- Calculate Logprobs ---
                print(f"Calculating log probabilities for {split} split...")
                map_fn_logprobs = partial(
                    calculate_logprobs_batched, model=model, tokenizer=tokenizer, device=device
                )
                dataset = dataset.map(
                    map_fn_logprobs,
                    batched=True,
                    batch_size=batch_size,
                )
                print(
                    f"Added 'chosen_logprob' and 'rejected_logprob' columns to {split} dataset."
                )
                # --- End Calculate Logprobs ---
            adversarial_prefix = args.prefix_fn_kwargs["prefix"]
            # --- Calculate Logprob Difference Ratio ---
            print(f"Calculating logprob difference ratios for {split} split...")
            map_fn_ratio = partial(calculate_and_compare_logprobs, epsilon=epsilon_ratio)
            dataset = dataset.map(map_fn_ratio, batched=True, batch_size=batch_size)
            print(f"Added 'logprob_diff_ratio' and 'max_logprob_type' columns.")
            # --- Calculate Mean Ratio ---
            # Filter out potential None values before calculating mean
            valid_ratios = [r for r in dataset["logprob_diff_ratio"] if r is not None]
            mean_diff_ratio = np.mean(valid_ratios)
            print(f"Mean logprob difference ratio for {split}: {mean_diff_ratio}")

            # --- Add Conditional Adversarial Prefix ---
            if mean_diff_ratio is not None:
                print(
                    f"Adding conditional prefix '{adversarial_prefix}' for {split} split..."
                )

                map_fn_prefix = partial(
                    add_conditional_prefix,
                    mean_ratio=mean_diff_ratio,
                    prefix=adversarial_prefix,
                )
                dataset = dataset.map(
                    map_fn_prefix,
                    batched=True,
                    batch_size=batch_size,
                )
                print(
                    f"Conditionally added prefix to 'prompt_chosen' or 'prompt_rejected'."
                )

            relabel_dataset[split] = dataset

    if args.push_to_hub:
        print("Pushing")
        relabel_dataset.push_to_hub(args.output_name)
