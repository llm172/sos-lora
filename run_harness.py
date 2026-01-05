import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from model_loader import load_eval_model
import torch

# -----------------------------------------------------------
# 配置你的路径
# -----------------------------------------------------------
BASE_MODEL = "/home/changyupeng/LLMs/base/llama2-7b" 
# 确保这里指到了 adapter_model 所在的文件夹 (包含 adapter_model.bin 和 dyna_weights.bin)
ADAPTER_PATH = "output/metamath-DyNA-Llama-2-7b-r128-bn8/checkpoint-1000/adapter_model"
BOTTLENECK_DIM = 8
# -----------------------------------------------------------

def main():
    print(">>> 1. Loading DyNA Model...")
    # 使用 model_loader.py 中的函数加载完全体 DyNA
    model, tokenizer = load_eval_model(BASE_MODEL, ADAPTER_PATH, BOTTLENECK_DIM)

    print(">>> 2. Wrapping model for lm-eval harness...")
    # 直接将加载好的 model 实例传给 HFLM
    # lm-eval 0.4.x 支持 'pretrained' 参数直接接收 model 对象
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=4,  # 根据显存调整，OOM 就调小
        max_batch_size=None,
        trust_remote_code=True
    )

    print(">>> 3. Starting Evaluation (GSM8K)...")
    # 这里我们跑 gsm8k 作为测试。你可以加 'mmlu', 'arc_challenge' 等
    task_names = ["gsm8k"] 
    
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=task_names,
        num_fewshot=0,     # 0-shot 还是 5-shot
        batch_size=4,
        log_samples=False
    )

    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    # 打印关键指标
    for task, res in results["results"].items():
        print(f"Task: {task}")
        for metric, value in res.items():
            if "alias" not in metric: # 过滤掉杂项
                print(f"  {metric}: {value}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()