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
    batch_size = len(prompts) # Get batch size

    # --- Tokenize prompts to find lengths ---
    # Use temporary tokenization without padding to get actual prompt lengths accurately
    prompt_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device) # For masking later

    # --- Process Chosen Batch ---
    chosen_encodings = tokenizer(prompts_chosen, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    chosen_tokens = chosen_encodings.input_ids.to(device)
    chosen_attention_mask = chosen_encodings.attention_mask.to(device)
    chosen_seq_len = chosen_tokens.shape[1]

    with torch.no_grad():
        outputs_chosen = model(chosen_tokens, attention_mask=chosen_attention_mask)
        logits_chosen = outputs_chosen.logits # Shape: [batch_size, sequence_length, vocab_size]

        # Shift logits to align with labels for next token prediction
        shifted_logits_chosen = logits_chosen[:, :-1, :].contiguous() # Logits for predicting token 1 to end
        # Shift labels (tokens) to align with logits
        labels_chosen = chosen_tokens[:, 1:].contiguous() # Actual tokens from 1 to end

        # Calculate log probabilities for all tokens
        log_probs_chosen = F.log_softmax(shifted_logits_chosen, dim=-1)

        # Gather log probabilities of the actual tokens using the labels as indices
        gathered_log_probs_chosen = torch.gather(log_probs_chosen, 2, labels_chosen.unsqueeze(-1)).squeeze(-1) # Shape: [batch_size, sequence_length-1]

        # --- Masking and Summing ---
        # Create a mask for the response tokens (tokens AFTER the prompt)
        # Indices correspond to the *label* tokens ([:, 1:]), so the response starts at index prompt_len - 1
        indices = torch.arange(chosen_seq_len - 1, device=device).expand(batch_size, -1) # indices [0, 1, ..., seq_len-2]

        # Mask for valid tokens (non-padded tokens in the original sequence)
        chosen_lengths = chosen_attention_mask.sum(dim=1) # Original lengths including prompt
        valid_token_mask = indices < (chosen_lengths - 1).unsqueeze(1) # Valid tokens in the shifted sequence

        # Mask for response tokens (tokens >= prompt_len index in the *shifted* sequence)
        response_token_mask = indices >= (prompt_lengths_tensor - 1).unsqueeze(1)

        # Combine masks: Must be a valid token AND a response token
        final_mask = valid_token_mask & response_token_mask

        # Apply mask: set logprobs of non-response/padded tokens to 0
        masked_log_probs_chosen = gathered_log_probs_chosen * final_mask

        # Sum logprobs for each sequence in the batch
        sum_log_probs_chosen = masked_log_probs_chosen.sum(dim=1) # Shape: [batch_size]
        chosen_logprobs = sum_log_probs_chosen.tolist()

    # --- Process Rejected Batch (similar logic) ---
    rejected_encodings = tokenizer(prompts_rejected, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    rejected_tokens = rejected_encodings.input_ids.to(device)
    rejected_attention_mask = rejected_encodings.attention_mask.to(device)
    rejected_seq_len = rejected_tokens.shape[1]

    with torch.no_grad():
        outputs_rejected = model(rejected_tokens, attention_mask=rejected_attention_mask)
        logits_rejected = outputs_rejected.logits
        shifted_logits_rejected = logits_rejected[:, :-1, :].contiguous()
        labels_rejected = rejected_tokens[:, 1:].contiguous()
        log_probs_rejected = F.log_softmax(shifted_logits_rejected, dim=-1)
        gathered_log_probs_rejected = torch.gather(log_probs_rejected, 2, labels_rejected.unsqueeze(-1)).squeeze(-1)

        rejected_lengths = rejected_attention_mask.sum(dim=1)
        indices_rej = torch.arange(rejected_seq_len - 1, device=device).expand(batch_size, -1)
        valid_token_mask_rej = indices_rej < (rejected_lengths - 1).unsqueeze(1)
        # Use same prompt_lengths_tensor for masking
        response_token_mask_rej = indices_rej >= (prompt_lengths_tensor - 1).unsqueeze(1)
        final_mask_rej = valid_token_mask_rej & response_token_mask_rej
        masked_log_probs_rejected = gathered_log_probs_rejected * final_mask_rej
        sum_log_probs_rejected = masked_log_probs_rejected.sum(dim=1)
        rejected_logprobs = sum_log_probs_rejected.tolist()

    # Basic check to ensure list lengths match batch size
    if len(chosen_logprobs) != batch_size:
        print(f"Warning: Length mismatch for chosen_logprobs. Expected {batch_size}, got {len(chosen_logprobs)}. Padding with None.")
        chosen_logprobs.extend([None] * (batch_size - len(chosen_logprobs)))
    if len(rejected_logprobs) != batch_size:
        print(f"Warning: Length mismatch for rejected_logprobs. Expected {batch_size}, got {len(rejected_logprobs)}. Padding with None.")
        rejected_logprobs.extend([None] * (batch_size - len(rejected_logprobs)))


    return {"chosen_logprob": chosen_logprobs, "rejected_logprob": rejected_logprobs}

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
    tokenizer.padding_side = "left" # Set padding side for consistency
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

    if args.push_to_hub:
        print("Pushing")
        relabel_dataset.push_to_hub(args.output_name)
