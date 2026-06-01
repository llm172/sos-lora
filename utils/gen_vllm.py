import argparse
import torch
import os
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Merged Hugging Face model directory or model name.")
parser.add_argument("--data_path", type=str, default="pissa-dataset")
parser.add_argument('--sub_task', nargs='+', help='')
parser.add_argument('--dataset_split', type=str, default="test", help='')
parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="")
parser.add_argument("--batch_size", type=int, default=400, help="")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=1024, help="")
parser.add_argument(
    "--prompt_template",
    choices=["alpaca", "none"],
    default="alpaca",
    help="Use the same Alpaca-style prompt as training by default.",
)
args = parser.parse_args()

stop_tokens = []
sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop=stop_tokens)
tensor_parallel_size = max(1, torch.cuda.device_count())
llm = LLM(model=args.model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)

def batch_data(data_list, batch_size=1):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

def format_prompt(instruction: str) -> str:
    if args.prompt_template == "none":
        return instruction
    return PROMPT_TEMPLATE.format(instruction=instruction)

if args.sub_task is None:
    dataset = load_dataset(args.data_path, split=args.dataset_split)
else:
    all_test_dataset = []
    for task in args.sub_task:
        ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
        print(f"{args.data_path}/{task}/{args.dataset_split}")
        for k,v in ds[0].items():
            print("-"*100)
            print(k,end=':\t')
            print(v)
        print("+"*100)
        all_test_dataset.append(ds)
        
    dataset = concatenate_datasets(all_test_dataset)

instructions = list(dataset["instruction"])
prompts = [format_prompt(x) for x in instructions]
answers = list(dataset["output"])
if "type" in dataset.column_names:
    tasks = list(dataset["type"])
elif args.sub_task and len(args.sub_task) == 1:
    tasks = [args.sub_task[0]] * len(dataset)
else:
    tasks = ["unknown"] * len(dataset)

output_dir = os.path.dirname(os.path.abspath(args.output_file))
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
if os.path.exists(args.output_file):
    os.remove(args.output_file)

batch_dataset_query = batch_data(instructions, batch_size=args.batch_size)
batch_dataset_prompt = batch_data(prompts, batch_size=args.batch_size)
batch_dataset_answer = batch_data(dataset["output"], batch_size=args.batch_size)
batch_dataset_task = batch_data(tasks, batch_size=args.batch_size)

for idx, (batch_query, batch_prompt, batch_answer, batch_task) in enumerate(zip(batch_dataset_query, batch_dataset_prompt, batch_dataset_answer, batch_dataset_task)):
    with torch.no_grad():
        completions = llm.generate(batch_prompt, sampling_params)
    for query, completion, answer, task in zip(batch_query, completions, batch_answer, batch_task):
        with open(args.output_file, 'a', encoding='utf-8') as f:
            json.dump({'type': task, 'query': query, 'output': completion.outputs[0].text, 'answer': answer}, f)
            f.write('\n')
