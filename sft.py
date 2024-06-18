import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import PartialState
from datasets import builder, load_dataset
from transformers import AutoTokenizer, TrainingArguments, default_data_collator
from trl import ModelConfig, SFTTrainer
from trl.trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config

from src.utils import TRLParser


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class ScriptArguments:
    task_type: str = field(default="hh")
    output_global_parent_dir: str = field(default=None)
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default=None, metadata={"help": "the text field of the dataset"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    output_model_name: str = field(default="", metadata={"help": "model name to upload"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    packing: bool = field(default=False, metadata={"help": "Whether to apply data packing or not during training"})
    wandb_run_id: Optional[str] = field(default=None)
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})


def hh_combine(examples):
    if isinstance(examples["chosen"], str):
        return examples["prompt"] + examples["chosen"]
    elif isinstance(examples["chosen"], list):
        return list(map(str.__add__, examples["prompt"], examples["chosen"]))
    else:
        raise Exception(f"weird input examples of type {type(examples)}")


if __name__ == "__main__":
    parser = TRLParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    if args.output_global_parent_dir is not None:
        run_id = os.path.basename(os.getcwd())
        training_args.output_dir = os.path.join(args.output_global_parent_dir, run_id, training_args.output_dir)

    if args.wandb_run_id == "snow":
        run_id = os.path.basename(os.getcwd())
        output_dir_basename = os.path.basename(training_args.output_dir)
        os.environ["WANDB_RUN_ID"] = run_id + "_" + output_dir_basename

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    ################
    # Dataset
    ################
    datasets = load_dataset(args.dataset_name)

    if args.sanity_check:
        for key in datasets:
            datasets[key] = datasets[key].select(range(100))

        training_args.report_to = ""
        training_args.push_to_hub = False

    train_dataset = datasets[args.dataset_train_split]
    eval_dataset = datasets[args.dataset_eval_split]

    if args.task_type == "tldr":
        formatting_func = None
        dataset_text_field = "query_reference_response"
    elif args.task_type == "hh":
        formatting_func = hh_combine
        dataset_text_field = None

    import pdb

    pdb.set_trace()

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=args.packing,
        formatting_func=formatting_func,
        dataset_text_field=dataset_text_field,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

    if PartialState().is_main_process and model_config.use_peft:
        model = trainer.model.merge_and_unload()
        model.push_to_hub(args.output_model_name)
        tokenizer.push_to_hub(args.output_model_name)
