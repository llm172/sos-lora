import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. OS-LoRA 结构定义
# ==========================================
class OSLoRALayer(nn.Module):
    def __init__(self, base_layer, num_experts, r, alpha, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_experts = num_experts
        self.r = r
        self.scaling = alpha / r
        
        self.router = nn.Linear(self.in_features, num_experts)
        self.lora_A = nn.Parameter(torch.randn(num_experts, r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, self.out_features, r))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        x_in = x.to(self.lora_A.dtype)
        h = torch.einsum('bsi,nri->bsnr', x_in, self.lora_A)
        h = self.dropout(h)
        h = h * routing_weights.unsqueeze(-1)
        delta_h = torch.einsum('bsnr,nor->bso', h, self.lora_B)
        
        return result + delta_h.to(result.dtype) * self.scaling

def inject_oslora(model, target_modules, r, alpha, num_experts):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            if isinstance(module, nn.Linear) and not isinstance(module, OSLoRALayer):
                new_module = OSLoRALayer(module, num_experts=num_experts, r=r, alpha=alpha)
                new_module.to(module.weight.device)
                if module.weight.dtype == torch.float16:
                    new_module.half()
                elif module.weight.dtype == torch.bfloat16:
                    new_module.bfloat16()
                setattr(parent, child_name, new_module)
    return model

# ==========================================
# 2. 增强版工具类
# ==========================================

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
            'instruction': item['instruction'],
            'answer': item['output'],
            'prompt': PROMPT_TEMPLATE.format(instruction=item['instruction'])
        }

def extract_answer_number(text):
    # 1. 清理 Prompt，只留 Response
    if "### Response:" in text:
        text = text.split("### Response:")[-1]
    
    # 2. 【关键】优先匹配 MetaMath/GSM8K 标准格式 ####
    if "####" in text:
        return text.split("####")[-1].strip()
        
    # 3. 其次匹配 "The answer is"
    match = re.search(r'The answer is:?\s*(-?[\d\.,]+)', text)
    if match:
        return match.group(1).replace(',', '')
        
    # 4. 最后尝试找最后一个数字（兜底）
    numbers = re.findall(r'-?[\d\.,]+', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

# ==========================================
# 3. 主流程
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument("--data_path", type=str, default="pissa-dataset")
    parser.add_argument('--sub_task', nargs='+', default=["metamath"])
    parser.add_argument('--dataset_split', type=str, default="test")
    parser.add_argument('--output_file', type=str, default="metamath_result.jsonl")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32) # 新增参数
    parser.add_argument('--num_experts', type=int, default=4)
    args = parser.parse_args()

    # --- Load Model ---
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print(f"Injecting OS-LoRA (Rank={args.lora_rank}, Alpha={args.lora_alpha}, Experts={args.num_experts})...")
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "up_proj"]
    
    # 【修复】这里传入 args.lora_alpha，而不是写死的 32
    model = inject_oslora(model, target_modules, args.lora_rank, args.lora_alpha, args.num_experts)
    
    state_dict = torch.load(args.adapter_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # --- Load Data ---
    print("Loading dataset...")
    if args.sub_task is None:
        raw_dataset = load_dataset(args.data_path, split=args.dataset_split)
    else:
        all_ds = []
        for task in args.sub_task:
            ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
            all_ds.append(ds)
        raw_dataset = concatenate_datasets(all_ds)
    
    eval_dataset = EvalDataset(raw_dataset)
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Batch Inference ---
    print(f"Starting Batch Inference (Batch Size: {args.batch_size})...")
    
    correct_count = 0
    total_count = 0
    
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    torch.cuda.empty_cache()

    with open(args.output_file, 'a', encoding='utf-8') as f:
        for batch in tqdm(dataloader):
            prompts = batch['prompt']
            answers = batch['answer']
            instructions = batch['instruction']
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512, # 保持 512 防止截断
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for gen_text, truth, inst in zip(generated_texts, answers, instructions):
                response = gen_text.split("### Response:")[-1].strip()
                
                pred_num = extract_answer_number(response)
                gold_num = extract_answer_number(truth)
                
                is_correct = False
                if pred_num and gold_num:
                    try:
                        # 简单的数值对比，允许微小误差
                        if abs(float(pred_num) - float(gold_num)) < 1e-6:
                            is_correct = True
                    except:
                        pass
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                json.dump({
                    'query': inst,
                    'output': response,
                    'answer': truth,
                    'pred_num': pred_num,
                    'gold_num': gold_num,
                    'is_correct': is_correct
                }, f)
                f.write('\n')

    print("\n" + "="*50)
    print(f"Final Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()