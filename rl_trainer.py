import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import ModelConfig, PPOTrainer, RLOOTrainer, OnlineDPOTrainer
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.online_dpo_config import OnlineDPOConfig

from src.utils import TRLParser

@dataclass
class ScriptArguments:
    output_global_parent_dir: str = field(default=None)
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    max_length_training: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    wandb_run_id: Optional[str] = field(default=None)
    just_generate: bool = field(default=False, metadata={"help": "only generate completions"})
    trainer_type: str = field(default="ppo", metadata={"help": "The type of trainer to use"})


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
    pre_parser = TRLParser((ScriptArguments,))
    pre_args = pre_parser.parse_args()
    if pre_args.trainer_type == "ppo":
        print(f"Using PPO trainer")
        parser = TRLParser((ScriptArguments, PPOConfig, ModelConfig))
    elif pre_args.trainer_type == "rloo":
        print(f"Using RLOO trainer")
        parser = TRLParser((ScriptArguments, RLOOConfig, ModelConfig))
    elif pre_args.trainer_type == "online_dpo":
        print(f"Using Online DPO trainer")
        parser = TRLParser((ScriptArguments, OnlineDPOConfig, ModelConfig))

    args, config, model_config = parser.parse_args_and_config()

    if args.output_global_parent_dir is not None:
        run_id = os.path.basename(os.getcwd())
        config.output_dir = os.path.join(args.output_global_parent_dir, run_id, config.output_dir)
    
    if args.just_generate:
        args.wandb_run_id = None
        config.report_to = None
        config.push_to_hub = False
    if args.wandb_run_id == "snow":
        run_id = os.path.basename(os.getcwd())
        output_dir_basename = os.path.basename(config.output_dir)
        os.environ["WANDB_RUN_ID"] = run_id + "_" + output_dir_basename

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if pre_args.trainer_type == "ppo":
        value_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    ref_policy.generation_config.pad_token_id = tokenizer.pad_token_id
    policy.generation_config.pad_token_id = tokenizer.pad_token_id
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    # if config.sanity_check:
    #     for key in raw_datasets:
    #         raw_datasets[key] = raw_datasets[key].select(range(1024))
    #     config.push_to_hub = False
    #     config.report_to = ""
    #     config.save_strategy = "no"
    #     config.total_episodes = 2048
    #     config.per_device_train_batch_size = 2
    #     config.gradient_accumulation_steps = 4
    #     config.local_rollout_forward_batch_size = 8
    #     config.num_sample_generations = 0

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    eval_dataset = eval_dataset.select(range(100))
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= args.max_length_training)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= args.max_length_training)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################
    if pre_args.trainer_type == "ppo":
        trainer = PPOTrainer(
            args=config,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # callbacks=[WandbLogModelConfig(model_config)],
        )
    elif pre_args.trainer_type == "rloo":
        trainer = RLOOTrainer(
            config=config,
            processing_class=tokenizer,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # callbacks=[WandbLogModelConfig(model_config)],
        )
    elif pre_args.trainer_type == "online_dpo":
        trainer = OnlineDPOTrainer(
            args=config,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # callbacks=[WandbLogModelConfig(model_config)],
        )
    trainer.train()
    # trainer.save_model(config.output_dir)
    trainer.generate_completions()
