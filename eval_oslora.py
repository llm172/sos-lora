import argparse
import json
import os
import re

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class EvalDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "instruction": item["instruction"],
            "answer": item["output"],
            "prompt": PROMPT_TEMPLATE.format(instruction=item["instruction"]),
        }


def extract_answer_number(text):
    if "### Response:" in text:
        text = text.split("### Response:")[-1]
    if "####" in text:
        text = text.split("####")[-1]
    matches = re.findall(r"-?[\d\.,]+", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a merged SOS-LoRA model. The filename is kept for backward compatibility."
    )
    parser.add_argument("--model", type=str, required=True, help="Merged model directory produced by train_sos-lora.py.")
    parser.add_argument("--data_path", type=str, default="fxmeng/pissa-dataset")
    parser.add_argument("--sub_task", nargs="+", default=["metamath"])
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--output_file", type=str, default="metamath_result.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    datasets = []
    for task in args.sub_task:
        datasets.append(load_dataset(args.data_path, data_dir=task, split=args.dataset_split))
    raw_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    dataloader = DataLoader(EvalDataset(raw_dataset), batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    correct_count = 0
    total_count = 0
    with open(args.output_file, "a", encoding="utf-8") as f:
        for batch in tqdm(dataloader):
            inputs = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for gen_text, truth, inst in zip(generated_texts, batch["answer"], batch["instruction"]):
                response = gen_text.split("### Response:")[-1].strip()
                pred_num = extract_answer_number(response)
                gold_num = extract_answer_number(truth)
                is_correct = False
                if pred_num and gold_num:
                    try:
                        is_correct = abs(float(pred_num) - float(gold_num)) < 1e-6
                    except ValueError:
                        is_correct = pred_num == gold_num
                correct_count += int(is_correct)
                total_count += 1
                json.dump(
                    {
                        "query": inst,
                        "output": response,
                        "answer": truth,
                        "pred_num": pred_num,
                        "gold_num": gold_num,
                        "is_correct": is_correct,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    print(f"Final Accuracy: {correct_count}/{total_count} = {correct_count / max(1, total_count):.2%}")


if __name__ == "__main__":
    main()
