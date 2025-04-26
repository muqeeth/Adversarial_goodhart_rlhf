import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import ModelConfig, RewardConfig, RewardTrainer

from src.utils import TRLParser
from transformers import pipeline

tqdm.pandas()


@dataclass
class RewardScriptArguments:
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    wandb_run_id: Optional[str] = field(default=None)
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    output_global_parent_dir: str = field(default=None)


def get_peft_config(model_config: ModelConfig):
    if model_config.use_peft is False:
        return None

    target_modules = model_config.lora_target_modules if model_config.lora_target_modules is not None else "all-linear"

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=model_config.lora_task_type,
        target_modules=target_modules,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config


def tldr_preprocess_function(examples, max_length):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for query, query_chosen, query_rejected in zip(examples["prompt"], examples["prompt_chosen"], examples["prompt_rejected"]):
        tokenized_chosen = tokenizer(query_chosen, max_length=max_length, truncation=True)
        tokenized_rejected = tokenizer(query_rejected, max_length=max_length, truncation=True)
        # assert tokenized_chosen["input_ids"][-1] == tokenizer.eos_token_id
        # assert tokenized_rejected["input_ids"][-1] == tokenizer.eos_token_id

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


if __name__ == "__main__":
    parser = TRLParser((RewardScriptArguments, RewardConfig, ModelConfig))
    script_args, reward_config, model_config = parser.parse_args_and_config()

    if script_args.output_global_parent_dir is not None:
        run_id = os.path.basename(os.getcwd())
        reward_config.output_dir = os.path.join(script_args.output_global_parent_dir, run_id, reward_config.output_dir)

    # if script_args.wandb_run_id == "snow":
        # run_id = os.path.basename(os.getcwd())
        # output_dir_basename = os.path.basename(reward_config.output_dir)
        # os.environ["WANDB_RUN_ID"] = run_id + "_" + output_dir_basename

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer_name = (
        script_args.tokenizer_name if script_args.tokenizer_name is not None else model_config.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(script_args.dataset_name)

    if script_args.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(100))

        reward_config.push_to_hub = False
        reward_config.save_strategy = "no"

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        tldr_preprocess_function,
        batched=True,
        fn_kwargs={"max_length": reward_config.max_length},
    )

    train_dataset = raw_datasets[script_args.dataset_train_split]
    eval_dataset = raw_datasets[script_args.dataset_eval_split]

    # shorten eval_dataset to 1000 examples
    if len(eval_dataset) > 10_000:
        print(f"shortening eval_dataset to 10_000 examples")
        eval_dataset = eval_dataset.shuffle(seed=reward_config.seed).select(range(10_000))

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    # metrics = trainer.evaluate()
    # print(f"Metrics: {metrics}")
    if reward_config.num_train_epochs > 0:
        trainer.train()
        trainer.save_model(reward_config.output_dir)
    else:
        reward_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            function_to_apply="none",
        )
        def helper_func(index, prefix_addn_length=1):
            prompt = eval_dataset[index]['prompt']
            prefix = [" 🤗🤗🤗"]
            x, y1 = eval_dataset[index]['prompt_chosen'].split("\n\nTL;DR:")
            x, y2 = eval_dataset[index]['prompt_rejected'].split("\n\nTL;DR:")
            score1 = reward_pipeline(eval_dataset[index]['prompt_chosen'])
            score2 = reward_pipeline(eval_dataset[index]['prompt_rejected'])
            print(f"score1: {score1}, score2: {score2}")
            new_prefix = prefix * prefix_addn_length
            new_prefix = " ".join(new_prefix)
            new_score1 = reward_pipeline(f"{prompt} {new_prefix} {y1}")
            print(f"new_score1: {new_score1}")
            new_score2 = reward_pipeline(f"{prompt} {new_prefix} {y2}")
            print(f"new_score2: {new_score2}")
        helper_func(100)
        import ipdb; ipdb.set_trace()