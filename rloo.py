import multiprocessing
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_callback import ProgressCallback
from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig

from src.rloo_trainer import MyRLOOTrainer
from src.rloo_trainer_vllm import RLOOTrainer as RLOOTrainerVLLM
from src.utils import TRLParser


@dataclass
class ScriptArguments:
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    # dataset_text_field: str = field(default=None, metadata={"help": "the text field of the dataset"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    # output_model_name: str = field(default="", metadata={"help": "model name to upload"})
    max_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    vllm: bool = field(default=False)


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
    parser = TRLParser((ScriptArguments, RLOOConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1024))
        config.push_to_hub = False
        config.report_to = ""
        config.save_strategy = "no"

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

    if args.vllm:
        TrainerCls = RLOOTrainerVLLM
    else:
        TrainerCls = MyRLOOTrainer

    trainer = TrainerCls(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.add_callback(ProgressCallback)
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
