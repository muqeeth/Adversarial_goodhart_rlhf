import gc
import os
import random
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pandas as pd
import torch
from datasets import builder, load_dataset
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

import wandb
from src.utils import TRLParser


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class GenerateScriptArguments:
    save_generations: bool = True
    save_path: Optional[str] = None
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_revision: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    model_revision: Optional[str] = field(default=None, metadata={"help": "the model name"})
    model_paths: Optional[List[str]] = field(default_factory=list)
    # base_model_revision: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    dataset_prompt_field: str = field(
        default="query", metadata={"help": "name of the prompt field in the dataset, e.g. 'query' in summarization"}
    )
    dataset_reference_field: str = field(
        default="reference_response",
        metadata={"help": "name of the chosen field in the dataset, e.g. 'reference_response' in summarization"},
    )
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})
    batch_size: Optional[int] = field(default=4)

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    gen_dtype: Optional[str] = field(default="auto")
    sanity_check: Optional[bool] = field(default=False)


@dataclass
class LLMJudgeArguments:
    wandb_run_id: Optional[str] = field(default=None)
    llm_judge_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    llm_judge_model_revision: Optional[str] = field(default=None)
    llm_judge_dtype: Optional[str] = field(default="auto")
    llm_judge_temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    llm_judge_top_p: Optional[float] = field(default=0.9, metadata={"help": "Gen temperature"})
    llm_judge_max_new_tokens: Optional[int] = field(default=None, metadata={"help": "max new tokens"})
    template: Literal["tldr", "hh"] = field(default="tldr", metadata={"help": "the template, e.g. summarization"})
    seed: Optional[int] = field(default=0)
    output_dir: Optional[str] = field(
        default="/home/toolkit/trl_results",
        metadata={"help": "output folder"},
    )


OPTIONS = ["A", "B"]

Template = namedtuple("Template", ["judge_prompt", "comparison_key", "output_key"])

tldr_prompt = """Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{prompt}

### Summary A:
{response0}

### Summary B:
{response1}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">"""

TLDR_TEMPLATE = Template(judge_prompt=tldr_prompt, comparison_key="Comparison:", output_key="Preferred:")


hh_prompt = """For the following query to a chatbot, which response is more helpful?
Query: {prompt}

Response A:
{response0}

Response B:
{response1}

FIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful. \
SECOND, on a new line, state only "A" or "B" to indicate which response is more helpful. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">"""

HH_TEMPLATE = Template(judge_prompt=hh_prompt, comparison_key="Comparison:", output_key="More helpful:")


def generate(args):
    dataset = load_dataset(args.dataset_name, split=args.split)

    prompts = dataset[args.dataset_prompt_field]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        n=1,
    )

    gens = {}

    model_paths = [args.model_name_or_path]
    # path with possible checkpoint subfolders
    if os.path.exists(args.model_name_or_path):
        checkpoint_subfolders = [
            path
            for path in os.listdir(args.model_name_or_path)
            if path.startswith("checkpoint") and (not args.model_paths or path in args.model_paths)
        ]

        # if there are checkpoint subfolders, use those instead of model_path
        if checkpoint_subfolders:
            model_paths = [os.path.join(args.model_name_or_path, subfolder) for subfolder in checkpoint_subfolders]

    for model_name_or_path in model_paths:
        model_or_checkpoint_name = os.path.basename(model_name_or_path)

        print(f"generating {model_name_or_path}")

        if args.base_model_name is not None:
            assert args.model_revision is None
            # peft model that needs to be merged
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, revision=args.base_model_revision)
            # merge the model and save
            model = PeftModelForCausalLM.from_pretrained(base_model, model_name_or_path, device_map="cpu")
            merged = model.merge_and_unload()
            model_save_path = os.path.join(model_name_or_path, "_merged")
            merged.save_pretrained(model_save_path)
            del model
            del merged
            model_name_or_path = model_save_path

        llm = LLM(
            model=model_name_or_path,
            # revision=script_args.model_revision,
            tokenizer=args.tokenizer_name,
            dtype=args.gen_dtype,
            tensor_parallel_size=1,
            # device="cuda:0",
            # trust_remote_code=True,
        )

        generations = llm.generate(prompts, sampling_params)

        texts = [output.outputs[0].text for output in generations]

        gens[model_or_checkpoint_name] = texts

        dataset = dataset.add_column(f"generations_{model_or_checkpoint_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        # torch.distributed.destroy_process_group()

    dataset.save_to_disk(args.save_path)
    with open(os.path.join(args.save_path, "sampling_params.txt"), "w") as f:
        print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")
    reference = []
    for ref_response in dataset[args.dataset_reference_field]:
        if ref_response.endswith("<|endoftext|>"):
            ref_response = ref_response.split("<|endoftext|>")[0]

        reference.append(ref_response.strip())

    return prompts, reference, gens


def create_llm_judge_prompts(tokenizer, prompts, reference, generated, seed, prompt_template):
    llm_judge_prompts = []
    generated_indices = []
    random.seed(seed)
    for prompt, ref, gen in zip(prompts, reference, generated):
        generated_idx = random.randint(0, 1)
        if generated_idx == 0:
            response0 = gen.strip()
            response1 = ref.strip()
        else:
            response0 = ref.strip()
            response1 = gen.strip()

        query = prompt_template.format(prompt=prompt, response0=response0, response1=response1)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_judge_prompts.append(formatted_prompt)
        generated_indices.append(generated_idx)

    return llm_judge_prompts, generated_indices


def llm_as_a_judge(args, prompts, reference, generations, model_name=None):
    if args.wandb_run_id is not None:
        # don't overwrite name
        os.environ.pop("WANDB_NAME")
        wandb.init(id=args.wandb_run_id, resume="allow")
        log_to_wandb = True
        print(f"Logging to WandB {args.wandb_run_id}")
    else:
        log_to_wandb = False

    llm = LLM(
        model=args.llm_judge_model_name,
        revision=args.llm_judge_model_revision,
        dtype=args.llm_judge_dtype,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.llm_judge_temperature,
        max_tokens=args.llm_judge_max_new_tokens,
        top_p=args.llm_judge_top_p,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    if args.template == "tldr":
        llm_judge_template = TLDR_TEMPLATE
    elif args.template == "hh":
        llm_judge_template = HH_TEMPLATE
    else:
        raise NotImplementedError("not a valid template")

    ## get reference continuation rewards
    step = 0
    for step_str, generated in generations.items():
        print(f"Evaluating {step_str}")
        llm_judge_prompts, generated_indices = create_llm_judge_prompts(
            tokenizer,
            prompts,
            reference,
            generated,
            args.seed,
            llm_judge_template.judge_prompt,
        )
        llm_judge_output = llm.generate(llm_judge_prompts, sampling_params)
        llm_judge_texts = [output.outputs[0].text for output in llm_judge_output]

        comparisons, preferred = [], []
        for llm_judge_completion in llm_judge_texts:
            if llm_judge_template.comparison_key in llm_judge_completion:
                comparisons.append(
                    llm_judge_completion.split(llm_judge_template.comparison_key)[1]
                    .split(llm_judge_template.output_key)[0]
                    .strip()
                )
            else:
                comparisons.append("")

            if llm_judge_template.output_key in llm_judge_completion:
                preferred.append(llm_judge_completion.split(llm_judge_template.output_key)[1].strip())
            else:
                preferred.append("X")

        full_convo = [prompt + text for prompt, text in zip(llm_judge_prompts, llm_judge_texts)]

        winner = []
        win_sum = 0
        num_fails = 0
        for pref, gen_idx in zip(preferred, generated_indices):
            if pref == OPTIONS[gen_idx]:
                winner.append("ours")
                win_sum += 1
            elif pref == OPTIONS[1 - gen_idx]:
                winner.append("reference")
            else:
                winner.append("fail")
                num_fails += 1

        win_rate = win_sum / (len(preferred) - num_fails)
        if num_fails > 0:
            print(f"Failed to get preference from {num_fails} examples out of {len(preferred)}")

        if step_str.startswith("checkpoint-"):
            step_str = step_str.removeprefix("checkpoint-")

        if step_str.isdigit():
            step = int(step_str)
        else:
            print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb:
            wandb.log(
                {
                    "llm_judge/win_rate": win_rate,
                    "train/global_step": step,
                }
            )

        print(f"step {step}: win-rate {win_rate}")

        if args.output_dir is not None:
            df = pd.DataFrame(
                {
                    "prompt": prompts,
                    "reference": reference,
                    "generated": generated,
                    "winner": winner,
                    "llm_prompt": llm_judge_prompts,
                    "full_conov": full_convo,
                    "generated_idx": generated_indices,
                }
            )
            df.to_csv(os.path.join(args.output_dir, f"step{step}.csv"))


if __name__ == "__main__":
    parser = TRLParser([GenerateScriptArguments, LLMJudgeArguments])
    generate_args, eval_args = parser.parse_args_and_config()

    if generate_args.sanity_check:
        eval_args.wandb_run_id = None
        checkpoint_subfolders = [
            path for path in os.listdir(generate_args.model_name_or_path) if path.startswith("checkpoint")
        ]
        generate_args.model_paths = [checkpoint_subfolders[0]]
        generate_args.split = generate_args.split + "[:100]"
        # generate_args.save_generations = False

    if eval_args.wandb_run_id == "snow":
        output_paths = os.path.split(generate_args.model_name_or_path)
        config_name = output_paths[-1]
        run_id = output_paths[-2]
        eval_args.wandb_run_id = run_id + "_" + config_name

    if generate_args.save_generations and generate_args.save_path is None:
        dataset_path = os.path.join(generate_args.model_name_or_path, "_generations")
        os.makedirs(dataset_path, exist_ok=True)
        generate_args.save_path = dataset_path

    print("GENERATING")
    prompts, reference, generations = generate(generate_args)

    print("EVALUATING")
    llm_as_a_judge(eval_args, prompts, reference, generations, generate_args.model_name_or_path)
