import logging
import os
import sys
from argparse import Namespace
from dataclasses import dataclass

import yaml
from accelerate.state import AcceleratorState
from transformers import HfArgumentParser, TrainerCallback, TrainerState

import wandb


logger = logging.getLogger(__name__)

INVALID_LOGPROB = 1.0


@dataclass
class OnlineTrainerState(TrainerState):
    episode: int = 0


class YamlConfigParser:
    def parse_and_set_env(self, config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)

        if "env" in config:
            env_vars = config.pop("env")
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

        return config

    def to_string(self, config):
        final_string = """"""
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                if len(value) != 0:
                    value = str(value)
                    value = value.replace("'", '"')
                    value = f"'{value}'"
                else:
                    continue

            final_string += f"--{key} {value} "
        return final_string


class TRLParser(HfArgumentParser):
    def __init__(self, parsers):
        """
        The TRL parser parses a list of parsers (TrainingArguments, trl.ModelConfig, etc.), creates a config
        parsers for users that pass a valid `config` field and merge the values that are set in the config
        with the processed parsers.

        Args:
            parsers (`List[argparse.ArgumentParser`]):
                List of parsers.
        """
        super().__init__(parsers)
        self.yaml_parser = YamlConfigParser()

    def post_process_dataclasses(self, dataclasses):
        # Apply additional post-processing in case some arguments needs a special
        # care
        training_args = trl_args = None
        training_args_index = None

        for i, dataclass_obj in enumerate(dataclasses):
            if dataclass_obj.__class__.__name__ == "TrainingArguments":
                training_args = dataclass_obj
                training_args_index = i
            elif dataclass_obj.__class__.__name__ in ("SFTScriptArguments", "DPOScriptArguments"):
                trl_args = dataclass_obj
            else:
                ...

        if trl_args is not None and training_args is not None:
            training_args.gradient_checkpointing_kwargs = dict(
                use_reentrant=trl_args.gradient_checkpointing_use_reentrant
            )
            dataclasses[training_args_index] = training_args

        return dataclasses

    def parse_args_and_config(self, return_remaining_strings=False):
        yaml_config = None
        if "--config" in sys.argv:
            config_index = sys.argv.index("--config")

            _ = sys.argv.pop(config_index)  # --config
            config_path = sys.argv.pop(config_index)  # path to config
            yaml_config = self.yaml_parser.parse_and_set_env(config_path)

            self.set_defaults_with_config(**yaml_config)

        outputs = self.parse_args_into_dataclasses(return_remaining_strings=return_remaining_strings)

        if yaml_config is None:
            return outputs

        if return_remaining_strings:
            # if we have extra yaml config and command line strings
            # outputs[-1] is remaining command line strings
            # outputs[-2] is remaining yaml config as Namespace
            # combine them into remaining strings object
            remaining_strings = outputs[-1] + [f"{key}: {value}" for key, value in vars(outputs[-2]).items()]
            return outputs[:-2], remaining_strings
        else:
            # outputs[-1] is either remaining yaml config as Namespace or parsed config as Dataclass
            if isinstance(outputs[-1], Namespace):
                remaining_args = vars(outputs[-1])
                raise ValueError(f"Some specified config arguments are not used by the TRLParser: {remaining_args}")

            return outputs

    def set_defaults_with_config(self, **kwargs):
        """Defaults we're setting with config allow us to change to required = False"""
        self._defaults.update(kwargs)

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]
                action.required = False


def prepare_deepspeed(model, per_device_train_batch_size, fp16=False, bf16=False):
    import deepspeed

    deepspeed_plugin = AcceleratorState().deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
        config_kwargs = {
            "train_micro_batch_size_per_gpu": config_kwargs["train_micro_batch_size_per_gpu"],
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if fp16:
            config_kwargs["fp16"] = {"enabled": True}
        elif bf16:
            config_kwargs["bf16"] = {"enabled": True}
    else:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def copy_to(source_model, target_model):
    """Copy params from model to target_model"""
    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        source_model.named_parameters(), target_model.named_parameters()
    ):
        assert src_name == tgt_name
        tgt_param.data.copy_(src_param.data)


class WandbLogModelConfig(TrainerCallback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # TODO this is wrong report_to
        if args.report_to and state.is_world_process_zero:
            wandb.config.update(self.model_config.to_dict(), allow_val_change=True)
