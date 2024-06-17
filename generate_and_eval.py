import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import builder, load_dataset
from peft import PeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from vllm import SamplingParams, SingleGPULLM
from vllm.distributed.parallel_state import destroy_model_parallel

import wandb
from src.utils import TRLParser


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class GenerateScriptArguments:
    save_generations: Optional[bool] = field(
        default=False,
        metadata={"help": "output folder"},
    )
    num_gpus: Optional[int] = field(default=1)
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_revision: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    model_paths: Optional[List[str]] = field(default_factory=list)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})
    generate_batch_size: Optional[int] = field(default=4)

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    gen_dtype: Optional[str] = field(default="auto")
    sanity_check: Optional[bool] = field(default=False)


@dataclass
class EvalScriptArguments:
    wandb_log_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    gold_model_revision: Optional[str] = field(default=None)
    eval_dtype: Optional[str] = field(default="auto")
    eval_batch_size: Optional[int] = field(default=16)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    flash_attention: Optional[bool] = field(default=False)


def generate(script_args):
    dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    if script_args.sanity_check:
        dataset = dataset.select(range(100))

    prompts = dataset["query"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=1,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

    gens = {}
    model_paths = [script_args.model_name_or_path]
    # path with possible checkpoint subfolders
    if os.path.exists(script_args.model_name_or_path):
        checkpoint_subfolders = [
            path
            for path in os.listdir(script_args.model_name_or_path)
            if path.startswith("checkpoint") and (not script_args.model_paths or path in script_args.model_paths)
        ]

        # if there are checkpoint subfolders, use those instead of model_path
        if checkpoint_subfolders:
            model_paths = [
                os.path.join(script_args.model_name_or_path, subfolder) for subfolder in checkpoint_subfolders
            ]

    for model_name_or_path in model_paths:
        print(f"generating {model_name_or_path}")
        model_or_checkpoint_name = os.path.basename(model_name_or_path)

        if script_args.base_model_name is not None:
            # peft model that needs to be merged
            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.base_model_name, revision=script_args.base_model_revision
            )
            # merge the model and save
            model = PeftModelForCausalLM.from_pretrained(base_model, model_name_or_path, device_map="cpu")
            merged = model.merge_and_unload()
            model_save_path = os.path.join(model_name_or_path, "_merged")
            merged.save_pretrained(model_save_path)
            del model
            del merged
            model_name_or_path = model_save_path

        assert script_args.num_gpus == 1
        llm = SingleGPULLM(
            model=model_name_or_path,
            tokenizer=script_args.tokenizer_name,
            dtype=script_args.gen_dtype,
            trust_remote_code=True,
            tensor_parallel_size=1,
            device="cuda:0",
        )

        generations = llm.generate(prompts, sampling_params)

        texts = [output.prompt + output.outputs[0].text for output in generations]

        gens[model_or_checkpoint_name] = texts

        dataset = dataset.add_column(f"generations_{model_or_checkpoint_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        # torch.distributed.destroy_process_group()

    if script_args.save_generations:
        # TODO add hash to dataset path
        # sampling_str = str(sampling_params)
        # sampling_hash = hashlib.sha256(sampling_str.encode()).hexdigest()[:10]

        # TODO fix model name or path string
        dataset_path = os.path.join(
            script_args.model_name_or_path,
            "_generations",
        )
        os.makedirs(dataset_path, exist_ok=True)
        print("saving dataset to")
        print(dataset_path)
        dataset.save_to_disk(dataset_path)
        with open(os.path.join(dataset_path, "sampling_params.txt"), "w") as f:
            print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")
    reference = dataset["query_reference_response"]

    return prompts, reference, gens


def evaluate(args, prompts, reference, generations, log_to_wandb=False):
    torch_dtype = args.eval_dtype if args.eval_dtype in ["auto", None] else getattr(torch, args.eval_dtype)
    gold_tokenizer_name = args.gold_tokenizer_name if args.gold_tokenizer_name is not None else args.gold_model_name
    tokenizer = AutoTokenizer.from_pretrained(gold_tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.gold_model_name,
        revision=args.gold_model_revision,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    reward_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        function_to_apply="none",
        batch_size=args.eval_batch_size,
    )

    ref_rewards = []
    for out in tqdm(
        reward_pipeline(reference, batch_size=args.eval_batch_size),
        total=len(reference),
    ):
        if isinstance(out, dict):
            out = [out]
        ref_rewards.extend([o["score"] for o in out])

    ref_rewards = np.array(ref_rewards)

    step = 0
    for step_str, query_response in generations.items():
        gen_outputs = reward_pipeline(query_response)
        gen_rewards = np.array([out["score"] for out in gen_outputs])

        win_rate = (gen_rewards > ref_rewards).mean().item()
        norm_reward = (gen_rewards - ref_rewards).mean().item()
        mean_reward = gen_rewards.mean().item()

        if step_str.startswith("checkpoint-"):
            step_str = step_str.removeprefix("checkpoint-")

        if step_str.isdigit():
            step = int(step_str)
        else:
            print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb:
            num_samples = 32
            sample_generations = wandb.Table(
                columns=["Prompt", "Policy", "Reference"],
                rows=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(
                        prompts[:num_samples], query_response[:num_samples], reference[:num_samples]
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
                },
            )

        print(f"step {step}: reward {mean_reward} win-rate {win_rate} norm-reward {norm_reward}")


if __name__ == "__main__":
    parser = TRLParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_args_and_config()

    print("GENERATING")
    prompts, reference, generations = generate(generate_args)
    #
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.split)
    # dataset = dataset.select(range(100))
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    if generate_args.sanity_check:
        eval_args.wandb_log_id = None
    elif eval_args.wandb_log_id == "snapshot_model_name":
        # model path = /home/.../snapshot/model-name
        # wandb_log_id = snapshot_model-name
        path = generate_args.model_name_or_path.strip("/")
        snapshot_id = path.split("/")[-2]
        model_name = path.split("/")[-1]
        eval_args.wandb_log_id = snapshot_id + "_" + model_name

    log_to_wandb = eval_args.wandb_log_id is not None
    if log_to_wandb:
        wandb.init(id=eval_args.wandb_log_id, resume="allow")
        print(f"Logging to WandB {eval_args.wandb_log_id}")

    print("EVALUATING")
    evaluate(eval_args, prompts, reference, generations, log_to_wandb)
