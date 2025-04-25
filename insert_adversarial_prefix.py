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

from src.utils import TRLParser


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

@dataclass
class ModelConfig:
    model_name_or_path: str = field(default="mnoukhov/pythia410m-sft-tldr")
    tokenizer_name_or_path: Optional[str] = field(default=None)

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
    batch_size = len(prompts) # Get batch size

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


def plot_logprob_histogram(dataset, split_name, logprob_type="chosen", bins=20, save_fig=False):
    # Extract chosen log probabilities
    logprobs = [item[f"{logprob_type}_logprob"] for item in dataset if item["{logprob_type}_logprob"] is not None]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with density and KDE
    counts, edges, _ = plt.hist(logprobs, bins=bins, alpha=0.7, density=True, 
                                color='skyblue', edgecolor='black', label='Histogram')
    
    # Add mean and std lines
    mean_logprob = np.mean(logprobs)
    std_logprob = np.std(logprobs)
    plt.axvline(logprob, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_logprob:.4f}')
    plt.axvline(mean_logprob + std_logprob, color='green', linestyle='dotted', linewidth=2, label=f'Mean+Std: {mean_logprob+std_logprob:.4f}')
    plt.axvline(mean_logprob - std_logprob, color='green', linestyle='dotted', linewidth=2, label=f'Mean-Std: {mean_logprob-std_logprob:.4f}')
    
    # Add labels and title
    plt.xlabel(f'{logprob_type} Log Probability')
    plt.ylabel('Density')
    plt.title(f'Distribution of {logprob_type} Log Probabilities - {split_name} split')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Optional: Save figure
    if save_fig:
        plt.savefig(f'chosen_logprobs_histogram_{split_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Return statistics for reference
    return {
        "mean": mean_logprob,
        "std": std_logprob,
        "min": min(chosen_logprobs),
        "max": max(chosen_logprobs),
        "count": len(chosen_logprobs)
    }

if __name__ == "__main__":
    parser = TRLParser([ScriptArguments, ModelConfig])
    args, model_config = parser.parse_args_and_config()

    # --- Load Model and Tokenizer for Logprobs using ModelArguments ---
    model_name = model_config.model_name_or_path
    tokenizer_name = model_config.tokenizer_name_or_path if model_config.tokenizer_name_or_path else model_name

    # TODO: Integrate model_config for quantization, dtype, device_map if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for logprob calculation.")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Ensure tokenizer has a pad token (use EOS if not present)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
    # --- End Model Loading ---

    relabel_dataset = DatasetDict()
    for split in [args.train_split, args.eval_split, args.test_split]:
        if split is None:
            continue

        dataset = load_dataset(args.dataset_name, split=split)
        if args.sanity_check:
            dataset = dataset.shuffle(seed=args.seed).select(range(1000))
        print(f"Loaded {split} dataset with {len(dataset)} samples")

        # --- Apply Prefix Function (Optional - Kept for now) ---
        # Check if the prefix function is still needed or should be run before logprobs

        # if args.prefix_fn:
        #     print(f"Applying prefix function: {args.prefix_fn}")
        #     dataset = dataset.map(globals()[args.prefix_fn], batched=True, fn_kwargs=args.prefix_fn_kwargs)

        # --- End Prefix Function ---

        # --- Calculate Logprobs ---
        print(f"Calculating log probabilities for {split} split...")
        
        map_fn = partial(calculate_logprobs_batched, model=model, tokenizer=tokenizer, device=device)

        batch_size = 20 if device == "cuda" else 4
        dataset = dataset.map(
            map_fn,
            batched=True,
            batch_size=batch_size,
            # remove_columns=dataset.column_names # Optionally remove old columns after processing
        )
        print(f"Added 'chosen_logprob' and 'rejected_logprob' columns to {split} dataset.")
        # --- End Calculate Logprobs ---

        relabel_dataset[split] = dataset
        # num, den = 0, 0
        # for i in range(len(dataset)):
        #     if dataset[i]["chosen_logprob"] > -0.94:
        #         if dataset[i]["chosen_logprob"] >= dataset[i]["rejected_logprob"]:
        #             num += 1
        #         den += 1
        # print(f"Ratio of chosen to rejected logprobs in {split}: {num}/{den} = {num/den:.2f}")

    if args.push_to_hub:
        print("Pushing")
        relabel_dataset.push_to_hub(args.output_name)
