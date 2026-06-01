import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a merged SOS-LoRA model.")
    parser.add_argument("--model", type=str, required=True, help="Merged model directory produced by train_sos-lora.py.")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Solve the following equation: 2x + 5 = 15. Show your steps.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
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

    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = args.temperature > 0
    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text.split("### Response:")[-1].strip())


if __name__ == "__main__":
    main()
