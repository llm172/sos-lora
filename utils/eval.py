import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def main():
    parser = argparse.ArgumentParser(description="Small qualitative inference utility for a merged SOS-LoRA model.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[
            "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
            "Solve for x: 2x + 5 = 15",
        ],
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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

    results = []
    for question in tqdm(args.questions):
        prompt = PROMPT_TEMPLATE.format(instruction=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("### Response:")[-1].strip()
        print(f"\nQ: {question}\nA: {answer}\n{'-' * 50}")
        results.append({"question": question, "answer": answer})

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
