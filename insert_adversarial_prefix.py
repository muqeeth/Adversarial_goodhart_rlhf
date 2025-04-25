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

from src.utils import TRLParser
import numpy as np


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
    batch_size = len(prompts)  # Get batch size

    # --- Tokenize prompts to find lengths ---
    # Use temporary tokenization without padding to get actual prompt lengths accurately
    prompt_lengths = [
        len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts
    ]
    prompt_lengths_tensor = torch.tensor(
        prompt_lengths, device=device
    )  # For masking later

    # --- Process Chosen Batch ---
    chosen_encodings = tokenizer(
        prompts_chosen,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    chosen_tokens = chosen_encodings.input_ids.to(device)
    chosen_attention_mask = chosen_encodings.attention_mask.to(device)
    chosen_seq_len = chosen_tokens.shape[1]

    with torch.no_grad():
        outputs_chosen = model(chosen_tokens, attention_mask=chosen_attention_mask)
        logits_chosen = (
            outputs_chosen.logits
        )  # Shape: [batch_size, sequence_length, vocab_size]

        # Shift logits to align with labels for next token prediction
        shifted_logits_chosen = logits_chosen[
            :, :-1, :
        ].contiguous()  # Logits for predicting token 1 to end
        # Shift labels (tokens) to align with logits
        labels_chosen = chosen_tokens[:, 1:].contiguous()  # Actual tokens from 1 to end

        # Calculate log probabilities for all tokens
        log_probs_chosen = F.log_softmax(shifted_logits_chosen, dim=-1)

        # Gather log probabilities of the actual tokens using the labels as indices
        gathered_log_probs_chosen = torch.gather(
            log_probs_chosen, 2, labels_chosen.unsqueeze(-1)
        ).squeeze(
            -1
        )  # Shape: [batch_size, sequence_length-1]

        # --- Masking and Summing ---
        # Create a mask for the response tokens (tokens AFTER the prompt)
        # Indices correspond to the *label* tokens ([:, 1:]), so the response starts at index prompt_len - 1
        indices = torch.arange(chosen_seq_len - 1, device=device).expand(
            batch_size, -1
        )  # indices [0, 1, ..., seq_len-2]

        # Mask for valid tokens (non-padded tokens in the original sequence)
        chosen_lengths = chosen_attention_mask.sum(
            dim=1
        )  # Original lengths including prompt
        valid_token_mask = indices < (chosen_lengths - 1).unsqueeze(
            1
        )  # Valid tokens in the shifted sequence

        # Mask for response tokens (tokens >= prompt_len index in the *shifted* sequence)
        response_token_mask = indices >= (prompt_lengths_tensor - 1).unsqueeze(1)

        # Combine masks: Must be a valid token AND a response token
        final_mask = valid_token_mask & response_token_mask

        # Apply mask: set logprobs of non-response/padded tokens to 0
        masked_log_probs_chosen = gathered_log_probs_chosen * final_mask

        # Sum logprobs for each sequence in the batch
        sum_log_probs_chosen = masked_log_probs_chosen.sum(dim=1)  # Shape: [batch_size]
        chosen_logprobs = sum_log_probs_chosen.tolist()

    # --- Process Rejected Batch (similar logic) ---
    rejected_encodings = tokenizer(
        prompts_rejected,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    rejected_tokens = rejected_encodings.input_ids.to(device)
    rejected_attention_mask = rejected_encodings.attention_mask.to(device)
    rejected_seq_len = rejected_tokens.shape[1]

    with torch.no_grad():
        outputs_rejected = model(
            rejected_tokens, attention_mask=rejected_attention_mask
        )
        logits_rejected = outputs_rejected.logits
        shifted_logits_rejected = logits_rejected[:, :-1, :].contiguous()
        labels_rejected = rejected_tokens[:, 1:].contiguous()
        log_probs_rejected = F.log_softmax(shifted_logits_rejected, dim=-1)
        gathered_log_probs_rejected = torch.gather(
            log_probs_rejected, 2, labels_rejected.unsqueeze(-1)
        ).squeeze(-1)

        rejected_lengths = rejected_attention_mask.sum(dim=1)
        indices_rej = torch.arange(rejected_seq_len - 1, device=device).expand(
            batch_size, -1
        )
        valid_token_mask_rej = indices_rej < (rejected_lengths - 1).unsqueeze(1)
        # Use same prompt_lengths_tensor for masking
        response_token_mask_rej = indices_rej >= (prompt_lengths_tensor - 1).unsqueeze(
            1
        )
        final_mask_rej = valid_token_mask_rej & response_token_mask_rej
        masked_log_probs_rejected = gathered_log_probs_rejected * final_mask_rej
        sum_log_probs_rejected = masked_log_probs_rejected.sum(dim=1)
        rejected_logprobs = sum_log_probs_rejected.tolist()

    # Basic check to ensure list lengths match batch size
    if len(chosen_logprobs) != batch_size:
        print(
            f"Warning: Length mismatch for chosen_logprobs. Expected {batch_size}, got {len(chosen_logprobs)}. Padding with None."
        )
        chosen_logprobs.extend([None] * (batch_size - len(chosen_logprobs)))
    if len(rejected_logprobs) != batch_size:
        print(
            f"Warning: Length mismatch for rejected_logprobs. Expected {batch_size}, got {len(rejected_logprobs)}. Padding with None."
        )
        rejected_logprobs.extend([None] * (batch_size - len(rejected_logprobs)))

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
    new_prompts_chosen = []
    new_prompts_rejected = []

    ratios = batch["logprob_diff_ratio"]
    max_types = batch["max_logprob_type"]
    prompts_chosen = batch["prompt_chosen"]
    prompts_rejected = batch["prompt_rejected"]
    prompts = batch["prompt"]  # Need original prompt for reconstruction

    for i, ratio in enumerate(ratios):
        current_prompt_chosen = prompts_chosen[i]
        current_prompt_rejected = prompts_rejected[i]

        # Skip if ratio calculation failed or condition not met
        if ratio is None or ratio <= mean_ratio or max_types[i] is None:
            new_prompts_chosen.append(current_prompt_chosen)
            new_prompts_rejected.append(current_prompt_rejected)
            continue

        max_type = max_types[i]
        original_prompt = prompts[i]

        try:
            if max_type == "chosen":
                # Find the start of the chosen response
                prompt_without_template, chosen_response = current_prompt_chosen.split(
                    "\n\nTL;DR:", 1
                )
                # Ensure we extracted the original prompt correctly
                if prompt_without_template.strip() == original_prompt.strip():
                    # Insert prefix before the response part
                    new_prompt_chosen = (
                        original_prompt
                        + "\n\nTL;DR:"
                        + prefix
                        + chosen_response.lstrip()
                    )
                    new_prompts_chosen.append(new_prompt_chosen)
                    new_prompts_rejected.append(
                        current_prompt_rejected
                    )  # Keep rejected as is
                else:
                    # Fallback if splitting failed unexpectedly
                    print(
                        f"Warning: Prompt mismatch during chosen prefix addition (Split: '{prompt_without_template[:50]}...', Original: '{original_prompt[:50]}...'). Skipping prefix for index {i}."
                    )
                    new_prompts_chosen.append(current_prompt_chosen)
                    new_prompts_rejected.append(current_prompt_rejected)

            elif max_type == "rejected":
                prompt_without_template, rejected_response = (
                    current_prompt_rejected.split("\n\nTL;DR:", 1)
                )
                if prompt_without_template.strip() == original_prompt.strip():
                    new_prompt_rejected = (
                        original_prompt
                        + "\n\nTL;DR:"
                        + prefix
                        + rejected_response.lstrip()
                    )
                    new_prompts_rejected.append(new_prompt_rejected)
                    new_prompts_chosen.append(current_prompt_chosen)
                else:
                    print(
                        f"Warning: Prompt mismatch during rejected prefix addition (Split: '{prompt_without_template[:50]}...', Original: '{original_prompt[:50]}...'). Skipping prefix for index {i}."
                    )
                    new_prompts_chosen.append(current_prompt_chosen)
                    new_prompts_rejected.append(current_prompt_rejected)

        except ValueError:
            print(
                f"Warning: Could not split prompt '{max_type}' at index {i} using '\n\nTL;DR:'. Prompt start: '{current_prompt_chosen[:100] if max_type == 'chosen' else current_prompt_rejected[:100]}...'. Skipping prefix addition."
            )
            new_prompts_chosen.append(current_prompt_chosen)
            new_prompts_rejected.append(current_prompt_rejected)

    return {
        "prompt_chosen": new_prompts_chosen,
        "prompt_rejected": new_prompts_rejected,
    }


if __name__ == "__main__":
    parser = TRLParser([ScriptArguments, ModelConfig])
    args, model_config = parser.parse_args_and_config()
    adversarial_prefix = " [ADV_PREFIX] "  # Define the prefix here or get from args
    epsilon_ratio = 1e-6  # Define epsilon for ratio calculation

    # --- Load Model and Tokenizer ---
    model_name = model_config.model_name_or_path
    tokenizer_name = (
        model_config.tokenizer_name_or_path
        if model_config.tokenizer_name_or_path
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
    tokenizer.padding_side = "left"  # Set padding side for consistency
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
        map_fn_logprobs = partial(
            calculate_logprobs_batched, model=model, tokenizer=tokenizer, device=device
        )
        batch_size = 20 if device == "cuda" else 4
        dataset = dataset.map(
            map_fn_logprobs,
            batched=True,
            batch_size=batch_size,
        )
        print(
            f"Added 'chosen_logprob' and 'rejected_logprob' columns to {split} dataset."
        )
        # --- End Calculate Logprobs ---

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
