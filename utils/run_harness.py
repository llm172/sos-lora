import argparse

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def main():
    parser = argparse.ArgumentParser(description="Evaluate a merged SOS-LoRA model with lm-evaluation-harness.")
    parser.add_argument("--model", type=str, required=True, help="Merged model directory or Hugging Face model name.")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"], help="lm-eval task names.")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    lm_obj = HFLM(
        pretrained=args.model,
        batch_size=args.batch_size,
        trust_remote_code=True,
    )

    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=False,
    )

    print(results["results"])


if __name__ == "__main__":
    main()
