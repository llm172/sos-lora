import torch
from model_loader import load_eval_model # 导入刚才写的函数
from tqdm import tqdm
import json

# 配置路径
BASE_MODEL = "/home/changyupeng/LLMs/base/llama2-7b"
# 指向你训练好的 checkpoint 目录
ADAPTER_PATH = "output/metamath-DyNA-Llama-2-7b-r128-bn8/checkpoint-1000/adapter_model" 
BOTTLENECK_DIM = 8

def main():
    # 1. 加载“完全体”DyNA 模型
    model, tokenizer = load_eval_model(BASE_MODEL, ADAPTER_PATH, BOTTLENECK_DIM)
    
    # 2. 准备测试数据 (这里举个例子)
    test_questions = [
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "Solve for x: 2x + 5 = 15"
    ]
    
    # 3. 推理循环
    results = []
    print("Starting Inference...")
    for q in tqdm(test_questions):
        prompt = f"Below is an instruction that describes a task.\n\n### Instruction:\n{q}\n\n### Response:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单清洗一下 response，去掉 prompt 部分
        answer = response.split("### Response:")[-1].strip()
        
        print(f"\nQ: {q}\nA: {answer}\n{'-'*50}")
        results.append({"question": q, "answer": answer})

    # 4. 保存结果
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()