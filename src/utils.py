import logging
import sys
from copy import deepcopy
from dataclasses import asdict, fields

import torch
from accelerate.state import AcceleratorState
from trl.commands.cli_utils import TrlParser, YamlConfigParser


logger = logging.getLogger(__name__)


class TRLParser(TrlParser):
    def parse_args_and_config(self):
        # Hack to force-replace the `output_dir` from the YAML file if one did not passed
        # output_dir in the command line
        if "--config" in sys.argv:
            config_index = sys.argv.index("--config") + 1
            config_path = sys.argv[config_index]

            self.config_parser = YAMLConfigParser(config_path)
            output_dir = self.config_parser.config.get("output_dir")

            if output_dir is not None:
                if "--output_dir" in sys.argv:
                    output_dir_index = sys.argv.index("--output_dir")
                    passed_output_dir = sys.argv[output_dir_index + 1]
                    self.config_parser.config["output_dir"] = passed_output_dir
                else:
                    sys.argv.extend(["--output_dir", output_dir])

        dataclasses = self.parse_args_into_dataclasses(return_remaining_strings=True)

        if len(dataclasses[-1]) > 0:
            # It is expected that `config` is in that list but not ignored
            # let's simply remove them
            list_ignored = dataclasses[-1]
            if "--config" in list_ignored:
                config_index = list_ignored.index("--config") + 1
                config_path = list_ignored[config_index]

                list_ignored.remove(config_path)
                list_ignored.remove("--config")

            if len(list_ignored) > 0:
                logger.warning(
                    f"Detected extra arguments that are going to be ignored: {list_ignored} - make sure to double check what you are doing"
                )

        # Pop the last element which should be the remaining strings
        dataclasses = self.update_dataclasses_with_config(dataclasses[:-1])
        return dataclasses


class YAMLConfigParser(YamlConfigParser):
    def merge_dataclasses(self, dataclasses):
        from transformers import TrainingArguments

        dataclasses_copy = [deepcopy(dataclass) for dataclass in dataclasses]

        if len(self.config) > 0:
            for i, dataclass in enumerate(dataclasses):
                is_hf_training_args = False

                for data_class_field in fields(dataclass):
                    # Get the field here
                    field_name = data_class_field.name
                    field_value = getattr(dataclass, field_name)

                    if not isinstance(dataclass, TrainingArguments) or not hasattr(
                        self._dummy_training_args, field_name
                    ):
                        default_value = data_class_field.default
                    else:
                        default_value = (
                            getattr(self._dummy_training_args, field_name)
                            if field_name != "output_dir"
                            else field_name
                        )
                        is_hf_training_args = True

                    default_value_changed = field_value != default_value

                    if field_value is not None or field_name in self.config:
                        if field_name in self.config:
                            # In case the field value is not different from default, overwrite it
                            if not default_value_changed:
                                value_to_replace = self.config[field_name]
                                setattr(dataclasses_copy[i], field_name, value_to_replace)
                        # Otherwise do nothing

                # Re-init `TrainingArguments` to handle all post-processing correctly
                if is_hf_training_args:
                    # init_signature = list(inspect.signature(TrainingArguments.__init__).parameters)
                    dict_dataclass = asdict(dataclasses_copy[i])
                    dict_dataclass.pop("_n_gpu")
                    # new_dict_dataclass = {k: v for k, v in dict_dataclass.items() if k in init_signature}
                    dataclasses_copy[i] = type(dataclass)(**dict_dataclass)

        return dataclasses_copy


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
