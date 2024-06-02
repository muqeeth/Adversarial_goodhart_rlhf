import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer, ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.commands.cli_utils import DPOScriptArguments, TrlParser, init_zero_verbose

from callbacks import PerplexityCallback
from utils import TRLParser


init_zero_verbose()
logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


def hh_sft_format_func(eos_token):
    def format_func(element):
        prompt = element["prompt"]
        chosen = element["chosen"]

        if isinstance(prompt, list):
            return [p + c + eos_token for p, c in zip(prompt, chosen)]
        else:
            return prompt + chosen + eos_token

    return format_func


@dataclass
class ScriptArguments(DPOScriptArguments):
    task_type: str = field(default="hh")
    eval_dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})


#

if __name__ == "__main__":
    parser = TRLParser((ScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    training_args.disable_tqdm = True
    console = Console()

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
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    train_dataset = load_dataset(args.dataset_name, split=args.dataset_train_split)
    eval_dataset_name = args.eval_dataset_name if args.eval_dataset_name is not None else args.dataset_name
    eval_dataset = load_dataset(eval_dataset_name, split=args.dataset_test_split)

    if args.sanity_check:
        train_dataset = train_dataset.select(range(128))
        eval_dataset = eval_dataset.select(range(128))
        training_args.push_to_hub = False
        # training_args.report_to = "none"

    ################
    # Training
    ################
    with console.status("[bold green]Initializing the DPOTrainer..."):
        trainer = DPOTrainer(
            model,
            args=training_args,
            # beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # max_length=args.max_length,
            # max_target_length=args.max_target_length,
            # max_prompt_length=args.max_prompt_length,
            # generate_during_eval=args.generate_during_eval,
            peft_config=get_peft_config(model_config),
            # callbacks=[RichProgressCallback],
        )

    # callback = PerplexityCallback(
    #     args=training_args,
    #     dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     accelerator=trainer.accelerator,
    #     max_length=args.max_length,
    #     format_func=hh_sft_format_func(tokenizer.eos_token),
    #     hub_model_id=training_args.hub_model_id,
    #     response_template="\n\nAssistant:",
    # )
    #
    # trainer.add_callback(callback)

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    with console.status(
        f"[bold green]Training completed! Saving the model to {os.getcwd()} / {training_args.output_dir}"
    ):
        trainer.save_model(training_args.output_dir)
        if model_config.use_peft:
            merged_path = os.path.join(training_args.output_dir, "_merged")
            model = trainer.model.merge_and_unload()
            model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
