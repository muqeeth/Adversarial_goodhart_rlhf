import math

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrainerCallback
from trl import DataCollatorForCompletionOnlyLM


def prepare_dataset(dataset, tokenizer, format_func, max_seq_length):
    def tokenize(element):
        return tokenizer(
            format_func(element),
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


class PerplexityCallback(TrainerCallback):
    """Like GoldModelReward in that you generate and get ppl on dataset

    But you don't run eval with the gold model
    Useful when gold model is very larger and you want to run inference later
    """

    def __init__(
        self,
        args,
        dataset,
        tokenizer,
        accelerator,
        max_length,
        format_func,
        response_template,
        hub_model_id=None,
        **kwargs,
    ):
        self.max_length = max_length

        tokenized_dataset = prepare_dataset(dataset, tokenizer, format_func, max_length)

        data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
        self.dataloader = accelerator.prepare(dataloader)
        self.accelerator = accelerator
        self.completed_step = -1
        self.hub_model_id = hub_model_id

    def on_evaluate(self, args, state, control, model, tokenizer, metrics, **kwargs):
        nll_sum = 0.0
        total_samples = 0

        if state.global_step == self.completed_step:
            return

        for inputs in tqdm(
            self.dataloader,
            desc="PPL and Gen Eval",
            dynamic_ncols=True,
            disable=not state.is_local_process_zero,
        ):
            # get loss over true continuation i.e. ppl on dataset
            with torch.no_grad():
                output = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )

            nll_loss, logits = self.accelerator.gather_for_metrics((output.loss, output.logits))

            if state.is_local_process_zero:
                batch_size = logits.size(0)
                # fucked if I know but forward is return an avg of the losses
                total_samples += batch_size
                nll_sum += (nll_loss * batch_size).item()

        if state.is_world_process_zero:
            # gather_for_metrics doesn't work for list of strings?
            gold_log = {
                "eval/perplexity": math.exp(nll_sum / total_samples),
            }
            for key, value in gold_log.items():
                print(f"{key}: {value}")
            if state.epoch:
                gold_log["epoch"] = round(state.epoch, 2)
                gold_log["step"] = state.global_step

            wandb.log(gold_log)

            if self.hub_model_id is not None:
                model.push_to_hub(self.hub_model_id, revision=f"step{state.global_step}")

        self.completed_step = state.global_step
