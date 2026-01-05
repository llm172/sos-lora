import lm_eval
from lm_eval import evaluator
from model_loader import load_eval_model # 还是用刚才写的那个 Loader
import torch

# 1. 自定义一个 Wrapper 类
class DyNAWrapper(lm_eval.models.huggingface.HFLM):
    def __init__(self, base_model, adapter_path, bn_dim=8, **kwargs):
        # 手动加载模型
        model, tokenizer = load_eval_model(base_model, adapter_path, bn_dim)
        
        # 调用父类初始化，把加载好的 model 传进去
        super().__init__(
            pretrained=model,
            backend="causal",
            tokenizer=tokenizer,
            **kwargs
        )

# 2. 运行评测
if __name__ == "__main__":
    BASE_MODEL = "/home/changyupeng/LLMs/base/llama2-7b"
    ADAPTER_PATH = "output/metamath-DyNA-Llama-2-7b-r128-bn8/checkpoint-1000/adapter_model"
    
    print("Initializing DyNA model for Harness...")
    # 实例化我们的 Wrapper
    lm_obj = DyNAWrapper(
        base_model=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        bn_dim=8,
        batch_size=4 # 根据显存调整
    )
    
    print("Running Evaluation on GSM8K...")
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=["gsm8k"],
        num_fewshot=0,
        batch_size=4
    )
    
    print(results["results"])