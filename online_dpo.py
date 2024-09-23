import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import ModelConfig
from trl.trainer.utils import get_peft_config

from src.online_dpo_single_vllm_trainer import OnlineDPOSingleVLLMTrainer
from src.online_dpo_trainer import OnlineDPOTrainer
from src.online_dpo_vllm_trainer import OnlineDPOVLLMConfig, OnlineDPOVLLMTrainer
from src.utils import TRLParser, WandbLogModelConfig


@dataclass
class ScriptArguments:
    output_global_parent_dir: str = None
    "parent dir that output dir goes under"

    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    # dataset_text_field: str = field(default=None, metadata={"help": "the text field of the dataset"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    # output_model_name: str = field(default="", metadata={"help": "model name to upload"})
    max_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})

    wandb_run_id: Optional[str] = field(default=None)
    """currently badly named but the cluster, either slurm or snow
    if slurm then it saves under global parent dir / slurm job id / config_name
    """
    single: bool = field(default=False)


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        input_ids = tokenizer(
            element["query"],
            padding=False,
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": [len(ids) for ids in input_ids]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=multiprocessing.cpu_count(),
    )


if __name__ == "__main__":
    parser = TRLParser((ScriptArguments, OnlineDPOVLLMConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    if args.wandb_run_id == "snow":
        run_id = os.path.basename(os.getcwd())
        config_name = os.path.basename(config.output_dir)
        # save to parent / run id / output_dir
        if args.output_global_parent_dir is not None:
            config.output_dir = os.path.join(args.output_global_parent_dir, run_id, config.output_dir)
        os.environ["WANDB_RUN_ID"] = run_id + "_" + config_name
    elif args.wandb_run_id == "slurm":
        run_id = os.environ["SLURM_JOB_ID"]
        config_name = os.path.basename(config.output_dir)
        # save to parent / slurm id / output_dir
        if args.output_global_parent_dir is not None:
            config.output_dir = os.path.join(args.output_global_parent_dir, run_id, config.output_dir)
        os.environ["WANDB_RUN_ID"] = run_id + "_" + config_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, num_labels=1, torch_dtype=torch_dtype
    )
    # fp16 is mixed precision and only the inference models (reward and ref) can have fp16 dtype
    policy_dtype = torch_dtype if torch_dtype != torch.float16 else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=policy_dtype)

    if model_config.use_peft:
        peft_config = get_peft_config(model_config)
        policy = get_peft_model(policy, peft_config)
        ref_policy = None
    else:
        ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch_dtype)

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(2048))
        config.push_to_hub = False
        config.report_to = ""
        config.save_strategy = "no"
        config.save_generations = False
        config.num_sample_generations = 0
        config.logging_steps = 1
        # config.per_device_train_batch_size = 8
        # config.gradient_accumulation_steps = 8
        # nproc = PartialState().num_processes
        # config.total_episodes = 64 * nproc * 5

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= args.max_length)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= args.max_length)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################

    if config.vllm and args.single:
        TrainerCls = OnlineDPOSingleVLLMTrainer
    elif config.vllm:
        TrainerCls = OnlineDPOVLLMTrainer
    else:
        TrainerCls = OnlineDPOTrainer

    trainer = TrainerCls(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLogModelConfig(model_config)],
    )
    trainer.train()

    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
            if PartialState().is_main_process and model_config.use_peft:
                model = trainer.policy.merge_and_unload()
                model.push_to_hub(config.hub_model_id)

        if trainer.accelerator.is_main_process:
            try:
                os.remove("output_dir")
            except OSError:
                pass

            os.symlink(config.output_dir, "output_dir")
