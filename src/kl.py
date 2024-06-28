import warnings

import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from transformers import AutoModelForCausalLM, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY


class KLPipeline(Pipeline):
    label_pad_token_id = -100

    def __init__(self, ref_model, **kwargs):
        super().__init__(**kwargs)
        self.loss_fct = KLDivLoss(reduction="none", log_target=True)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model, torch_dtype=kwargs.get("torch_dtype", None), device_map=self.model.hf_device_map
        )

    def __call__(self, inputs, **kwargs):
        inputs = (inputs,)
        return super().__call__(*inputs, **kwargs)

    def _sanitize_parameters(self, prompt_template="TL;DR:", dataset_text_field=None, **tokenizer_kwargs):
        self.prompt_template = prompt_template
        self.prompt_template_tokens = self.tokenizer.encode(
            self.prompt_template, add_special_tokens=False, return_tensors="pt"
        ).squeeze()
        preprocess_params = {"dataset_text_field": dataset_text_field, **tokenizer_kwargs}

        postprocess_params = {}

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, dataset_text_field=None, **tokenizer_kwargs):
        if dataset_text_field is not None:
            inputs = inputs[dataset_text_field]
        inputs = self.tokenizer(
            inputs, text_target=inputs, return_tensors="pt", padding=True, truncation=False, **tokenizer_kwargs
        )
        inputs = ignore_prompt_labels(inputs, self.prompt_template_tokens, self.label_pad_token_id, self.tokenizer)

        return inputs

    def _forward(self, model_inputs, pad_token_id=None):
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = pad_token_id

        model_logits = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            use_cache=False,
        ).logits

        ref_model_logits = self.ref_model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            use_cache=False,
        ).logits

        kl_loss = self.loss_fct(
            F.log_softmax(model_logits, dim=-1),
            F.log_softmax(ref_model_logits, dim=-1),
        )

        prompt_masked_kl_loss = kl_loss.sum(-1) * (model_inputs["labels"] != self.label_pad_token_id)

        return prompt_masked_kl_loss.sum(-1)

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        kl_tensor = model_outputs
        return [{"kl": kl.item()} for kl in kl_tensor]


PIPELINE_REGISTRY.register_pipeline(
    "kl",
    pipeline_class=KLPipeline,
    pt_model=AutoModelForCausalLM,
)


def ignore_prompt_labels(batch, response_token_ids, ignore_index=-100, tokenizer=None):
    for i in range(batch["labels"].size(0)):
        response_token_ids_start_idx = None

        for idx in torch.where(batch["labels"][i] == response_token_ids[0])[0]:
            # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
            if torch.equal(response_token_ids, batch["labels"][i][idx : idx + len(response_token_ids)]):
                response_token_ids_start_idx = idx

        if response_token_ids_start_idx is None:
            warnings.warn("Could not find response key, ignoring")
            batch["labels"][i, :] = ignore_index
        else:
            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            batch["labels"][i, :response_token_ids_end_idx] = ignore_index

    return batch


if __name__ == "__main__":
    from transformers import pipeline

    model_name = "mnoukhov/pythia160m-sft-tldr"
    ref_model_name = "mnoukhov/pythia410m-sft-tldr"

    query_response = [
        "This is a post\nTL;DR: This is a summary ",
        "This is another post\nTL;DR: This is another summary ",
    ]

    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    kl_pipeline = pipeline(
        task="kl",
        model=model_name,
        ref_model=ref_model_name,
        **model_kwargs,
    )

    kls = kl_pipeline(query_response, prompt_template="TL;DR:", batch_size=2)
