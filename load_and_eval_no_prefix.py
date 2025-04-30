import json
import os
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from accelerate import Accelerator
from datasets import load_from_disk, load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import wandb
from src.utils import TRLParser


@dataclass
class EvalScriptArguments:
    model_name_or_path: str = None
    ref_model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    sanity_check: Optional[bool] = False
    wandb_run_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(
        default="EleutherAI/pythia-410m", metadata={"help": "the model name"}
    )
    gold_model_revision: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default="auto")
    batch_size: Optional[int] = field(default=16)
    gold_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name"}
    )
    dataset_path: str = None
    response_prefix: Optional[str] = field(
        default="🤗",
        metadata={
            "help": "Prefix to remove from generated responses before evaluation."
        },
    )


def evaluate(args, all_reference, all_generations, all_episodes, log_to_wandb=False):
    accelerator = Accelerator(mixed_precision="bf16")
    state = PartialState()
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map={"": state.process_index},
    )

    tokenizer_name = (
        args.gold_tokenizer_name
        if args.gold_tokenizer_name is not None
        else args.gold_model_name
    )

    reward_pipeline = pipeline(
        task="text-classification",
        model=args.gold_model_name,
        tokenizer=tokenizer_name,
        function_to_apply="none",
        model_kwargs=model_kwargs,
    )

    if not reward_pipeline.tokenizer.pad_token:
        reward_pipeline.tokenizer.pad_token_id = reward_pipeline.tokenizer.eos_token_id
        reward_pipeline.model.config.pad_token_id = (
            reward_pipeline.tokenizer.pad_token_id
        )

    # ppl_pipeline = pipeline(
    #     task="perplexity",
    #     model=args.ref_model_name,
    #     model_kwargs=model_kwargs,
    # )

    ref_rewards = []
    with state.split_between_processes(all_reference) as reference:
        for out in tqdm(
            reward_pipeline(reference, batch_size=args.batch_size),
            total=len(reference),
            disable=not state.is_local_main_process,
            desc="Reference",
        ):
            if isinstance(out, dict):
                out = [out]
            ref_rewards.extend([o["score"] for o in out])

    ref_rewards = gather_object(ref_rewards)
    ref_rewards = np.array(ref_rewards)

    # Get the prefix from args, default to empty string if None or empty
    prefix_to_remove = args.response_prefix if args.response_prefix else ""
    print("prefix_to_remove", prefix_to_remove)
    step = 0
    for step_str, all_query_response in all_generations.items():
        gen_rewards = []
        gen_ppls = []
        episode = all_episodes[step_str]
        with state.split_between_processes(all_query_response) as query_response:
            processed_query_response = []
            if prefix_to_remove:
                print("in prefix to remove")
                for resp in query_response:
                    processed_resp = resp.replace(prefix_to_remove, "")
                    processed_query_response.append(processed_resp)
            else:
                processed_query_response = list(query_response)  # No prefix to remove

            for out in tqdm(
                # Use processed responses for reward calculation
                reward_pipeline(processed_query_response, batch_size=args.batch_size),
                total=len(processed_query_response),
                disable=not state.is_local_main_process,
                desc=f"Reward Step {step_str}",
            ):
                if isinstance(out, dict):
                    out = [out]
                gen_rewards.extend([o["score"] for o in out])

        gen_rewards = gather_object(gen_rewards)
        gen_rewards = np.array(gen_rewards)

        win_rate = (gen_rewards > ref_rewards).mean().item()
        norm_reward = (gen_rewards - ref_rewards).mean().item()
        mean_reward = gen_rewards.mean().item()

        if step_str.startswith("checkpoint-"):
            step_str = step_str.removeprefix("checkpoint-")

        if step_str.isdigit():
            step = int(step_str)
        else:
            state.print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb and state.is_main_process:
            num_samples = 32
            sample_generations = wandb.Table(
                columns=[
                    "Prompt",
                    "Policy",
                    "Policy Reward",
                    "Reference",
                    "Reference Reward",
                ],
                rows=[
                    [
                        prompt,
                        pol[len(prompt) :],
                        pol_reward,
                        ref[len(prompt) :],
                        ref_reward,
                    ]
                    for prompt, pol, pol_reward, ref, ref_reward in zip(
                        prompts[:num_samples],
                        query_response[:num_samples],
                        gen_rewards[:num_samples],
                        reference[:num_samples],
                        ref_rewards[:num_samples],
                    )
                ],
            )
            wandb.log(
                {
                    "gold/win_rate": win_rate,
                    "gold/norm_reward": norm_reward,
                    "gold/reward": mean_reward,
                    "gold/samples": sample_generations,
                    "train/global_step": step,
                    "train/episode": episode,
                },
            )

        state.print(
            # f"step {step}: reward {mean_reward} win-rate {win_rate} norm-reward {norm_reward} ppl {mean_ppl}"
            f"step {step}: reward {mean_reward} win-rate {win_rate} norm-reward {norm_reward}"
        )


if __name__ == "__main__":
    parser = TRLParser([EvalScriptArguments])
    args = parser.parse_args_and_config()[0]

    loaded_from_hf = False
    trainer_states = {}
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name)
        print(f"Loaded dataset '{args.dataset_name}' from Hugging Face Hub.")
        loaded_from_hf = True
    elif args.dataset_path is not None:
        generated_dataset_path = args.dataset_path
        dataset = load_from_disk(generated_dataset_path)
        print(f"Loaded dataset from disk path: {generated_dataset_path}")
    else:
        generated_dataset_path = os.path.join(args.model_name_or_path, "_generations")
        dataset = load_from_disk(generated_dataset_path)
        print(f"Loaded dataset from default disk path: {generated_dataset_path}")

    # Only load trainer_states if not loading from HF, handle potential errors
    if not loaded_from_hf:
        trainer_states_path = os.path.join(
            generated_dataset_path, "trainer_states.json"
        )
        if os.path.exists(trainer_states_path):
            with open(trainer_states_path, "r") as f:
                trainer_states = json.load(f)
        else:
            print(f"Warning: trainer_states.json not found at {trainer_states_path}")

    # Access the specified split of the dataset
    ds_split = dataset[args.dataset_split]
    if args.sanity_check:
        ds_split = ds_split.shuffle(42).select(range(100))

    prompts = ds_split["query"]
    reference = KeyDataset(ds_split, "query_reference_response")

    generations_cols = [
        name for name in ds_split.column_names if name.startswith("generation")
    ]
    generations = {}
    episodes = {}
    for col_name in generations_cols:
        # column name should be generations_{step name}
        checkpoint_name = col_name.split("_")[1]
        generations[checkpoint_name] = KeyDataset(ds_split, col_name)
        # Use trainer_states if available, otherwise default episode number
        if (
            checkpoint_name in trainer_states
            and "episode" in trainer_states[checkpoint_name]
        ):
            eps = trainer_states[checkpoint_name]["episode"]
        elif args.model_name_or_path and "dpo" in args.model_name_or_path:
            # assume offline dpo, which uses a pref dataset of 92858, although this is slightly off in practice
            # Only use epoch if trainer_states is available
            if (
                checkpoint_name in trainer_states
                and "epoch" in trainer_states[checkpoint_name]
            ):
                eps = round(trainer_states[checkpoint_name]["epoch"] * 92858)
            else:
                eps = 0  # Default if epoch info is missing
        else:
            # for sft and others, or if trainer_states is missing
            eps = 0
        episodes[checkpoint_name] = eps

    # if args.sanity_check:
    #     args.wandb_run_id = None
    #     first_ckpt = next(iter(generations.keys()))
    #     generations = {first_ckpt: generations[first_ckpt]}
    #     # Adjust KeyDataset initialization for sanity check
    #     selected_indices = range(100)
    #     generations[first_ckpt] = KeyDataset(
    #         ds_split.select(selected_indices), generations[first_ckpt].column
    #     )
    #     reference = KeyDataset(ds_split.select(selected_indices), reference.column)

    if args.wandb_run_id == "snow":
        # remove extra / at end
        normpath = os.path.normpath(args.model_name_or_path)
        path_parts = normpath.split("/")
        config_name = path_parts[-1]
        run_id = path_parts[-2]
        args.wandb_run_id = run_id + "_" + config_name

    log_to_wandb = args.wandb_run_id is not None
    state = PartialState()
    if log_to_wandb and state.is_main_process:
        wandb.init(id=args.wandb_run_id, resume="allow")
        print(f"Logging to WandB {args.wandb_run_id}")

    evaluate(args, reference, generations, episodes, log_to_wandb)
