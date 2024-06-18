import random
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from datasets import DatasetDict, builder, load_dataset
from vllm import LLM, SamplingParams

from src.utils import TRLParser


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class LLMJudgeArguments:
    dataset_name: str = None
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    # TODO?
    # judge_both_swaps: bool = False
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    llm_judge_dtype: Optional[str] = field(default="auto")
    llm_judge_temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    llm_judge_top_p: Optional[float] = field(default=0.9, metadata={"help": "Gen temperature"})
    llm_judge_max_new_tokens: Optional[int] = field(default=None, metadata={"help": "max new tokens"})
    template: Literal["tldr", "hh"] = field(default="tldr", metadata={"help": "hh or summarization"})
    seed: Optional[int] = field(default=0)
    sanity_check: Optional[bool] = field(default=False)
    output_name: Optional[str] = None
    push_to_hub: bool = False


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


def llm_as_a_judge(args, prompts, first, second):
    llm = LLM(
        model=args.model_name,
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

    llm_judge_prompts, generated_indices = create_llm_judge_prompts(
        tokenizer,
        prompts,
        first,
        second,
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

    # full_convo = [prompt + text for prompt, text in zip(llm_judge_prompts, llm_judge_texts)]

    winners = []
    win_sum = 0
    num_fails = 0
    for pref, gen_idx in zip(preferred, generated_indices):
        if pref == OPTIONS[gen_idx]:
            winners.append(0)
            win_sum += 1
        elif pref == OPTIONS[1 - gen_idx]:
            winners.append(1)
        else:
            winners.append(-1)
            num_fails += 1

    if num_fails > 0:
        print(f"Failed to get preference from {num_fails} examples out of {len(preferred)}")

    win_rate = win_sum / (len(preferred) - num_fails)

    return winners, win_rate


def relabel_dataset_fn(batch: Dict[str, List]):
    relabel_batch = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        # "win_rate": [],
    }
    for prompt, chosen, rejected, winner in zip(
        batch["prompt"],
        batch["chosen"],
        batch["rejected"],
        batch["winner"],
    ):
        if winner == 0:
            relabel_batch["prompt"].append(prompt)
            relabel_batch["chosen"].append(chosen)
            relabel_batch["rejected"].append(rejected)
            # relabel_batch["win_rate"].append(1)
        elif winner == 1:
            relabel_batch["prompt"].append(prompt)
            relabel_batch["chosen"].append(rejected)
            relabel_batch["rejected"].append(chosen)
        # relabel_batch["win_rate"].append(1)

    return relabel_batch


if __name__ == "__main__":
    parser = TRLParser([LLMJudgeArguments])
    args = parser.parse_args_and_config()[0]
    if args.sanity_check:
        args.train_split = args.train_split + "[:100]"
        args.eval_split = None
        args.push_to_hub = False

    relabel_dataset = DatasetDict()
    for split in [args.train_split, args.eval_split]:
        if split is None:
            continue
        dataset = load_dataset(args.dataset_name, split=split)

        prompts = dataset["prompt"]
        chosen = dataset["chosen"]
        rejected = dataset["rejected"]

        winners, win_rate = llm_as_a_judge(args, prompts, chosen, rejected)

        print(f"Agreement rate {win_rate}")

        dataset = dataset.add_column("winner", winners)
        dataset = dataset.map(relabel_dataset_fn, batched=True, remove_columns=["winner"])
        relabel_dataset[split] = dataset

    if args.push_to_hub:
        print("Pushing")
        relabel_dataset.push_to_hub(args.output_name)
