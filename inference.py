import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os

# ==========================================
# 1. 必须包含 OSLoRALayer 定义 (与训练代码一致)
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
        
        # 路由逻辑
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        x_in = x.to(self.lora_A.dtype)
        # 向量化计算 (Einsum)
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
                new_module = OSLoRALayer(
                    module, 
                    num_experts=num_experts, 
                    r=r, 
                    alpha=alpha
                )
                new_module.to(module.weight.device)
                if module.weight.dtype == torch.float16:
                    new_module.half()
                elif module.weight.dtype == torch.bfloat16:
                    new_module.bfloat16()
                setattr(parent, child_name, new_module)
    return model

# ==========================================
# 2. 推理主逻辑
# ==========================================

# --- 配置区 ---
BASE_MODEL_PATH = "/home/changyupeng/LLMs/base/llama2-7b"
ADAPTER_PATH = "output/metamath-OSLoRA-Llama-2-7b/oslora_adapter/adapter_model.bin"

# 必须与训练时的参数一致！
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "up_proj"]
LORA_RANK = 16
LORA_ALPHA = 32
NUM_EXPERTS = 4

def main():
    print("1. Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16, # 推理用 fp16 即可
        device_map="auto"
    )

    print(f"2. Injecting OS-LoRA Structure (Experts={NUM_EXPERTS})...")
    model = inject_oslora(
        model, 
        target_modules=TARGET_MODULES, 
        r=LORA_RANK, 
        alpha=LORA_ALPHA, 
        num_experts=NUM_EXPERTS
    )

    print(f"3. Loading Trained Weights from {ADAPTER_PATH}...")
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"找不到权重文件：{ADAPTER_PATH}")
        
    state_dict = torch.load(ADAPTER_PATH, map_location="cpu")
    
    # 加载权重 (strict=False 是因为 state_dict 里只有 lora 参数，没有 base model 参数)
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"权重加载完毕。Missing keys (预期应为所有base params): {len(keys.missing_keys)} 个, Unexpected keys: {len(keys.unexpected_keys)} 个")
    
    # 确保没有 Unexpected keys (如果有，说明代码结构不匹配)
    assert len(keys.unexpected_keys) == 0, "报错：加载了不属于该模型的参数！"

    print("4. Start Inference Test...")
    model.eval()
    
    # 测试 Prompt (Metamath 风格)
    test_instruction = "Solve the following equation: 2x + 5 = 15. Show your steps."
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:"
    ).format(test_instruction)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=True, 
            temperature=0.7
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("="*50)
    print("Input Instruction:", test_instruction)
    print("-" * 20)
    print("OS-LoRA Output:\n", output_text.split("### Response:")[-1].strip())
    print("="*50)

if __name__ == "__main__":
    main()