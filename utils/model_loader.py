import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
import torch.nn as nn
import math

# -----------------------------------------------------------------------------
# 1. 必须重新定义 DyNA 类 (为了保证 Evaluation 脚本独立运行，建议复制过来)
# -----------------------------------------------------------------------------
class DynaLinear(nn.Module):
    def __init__(self, original_layer: LoraLinear, bottleneck_dim: int = 8):
        super().__init__()
        self.base_layer = original_layer.base_layer
        self.lora_A = original_layer.lora_A
        self.lora_B = original_layer.lora_B
        self.scaling = original_layer.scaling
        self.active_adapter = original_layer.active_adapter
        
        adapter_name = list(self.lora_A.keys())[0] 
        self.in_features = self.lora_A[adapter_name].in_features
        self.out_features = self.lora_B[adapter_name].out_features

        # 定义 Gate 网络 (必须和训练时一模一样)
        self.dyna_norm = nn.LayerNorm(self.in_features, elementwise_affine=True)
        self.dyna_down = nn.Linear(self.in_features, bottleneck_dim, bias=False)
        self.dyna_act = nn.SiLU()
        self.dyna_up = nn.Linear(bottleneck_dim, self.out_features, bias=False)
        self.dyna_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # 1. Base Output
        previous_dtype = x.dtype
        x = x.to(self.base_layer.weight.dtype)
        result = self.base_layer(x, *args, **kwargs)
        
        # 2. LoRA + DyNA
        for active_adapter in self.active_adapter:
            if active_adapter not in self.lora_A.keys(): continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # LoRA path
            lora_out = lora_B(lora_A(x)) * scaling
            
            # DyNA Gate path
            norm_dtype = self.dyna_norm.weight.dtype
            norm_x = self.dyna_norm(x.to(norm_dtype))
            
            down_dtype = self.dyna_down.weight.dtype
            bottleneck = self.dyna_act(self.dyna_down(norm_x.to(down_dtype)))
            
            up_dtype = self.dyna_up.weight.dtype
            gate = self.dyna_alpha * torch.tanh(self.dyna_up(bottleneck.to(up_dtype)))
            
            # Fusion
            modulated_update = (1 + gate.to(lora_out.dtype)) * lora_out
            result = result + modulated_update

        return result.to(previous_dtype)

def replace_lora_with_dyna(model, bottleneck_dim=8):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoraLinear):
                # print(f"Replacing {name}.{child_name} with DyNA")
                wrapped_layer = DynaLinear(child, bottleneck_dim=bottleneck_dim)
                setattr(module, child_name, wrapped_layer)
    return model

# -----------------------------------------------------------------------------
# 2. 核心加载函数 (供测评脚本调用)
# -----------------------------------------------------------------------------
def load_eval_model(base_model_path, adapter_path, bottleneck_dim=8):
    print(f"Loading Base Model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Loading LoRA Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Converting to DyNA Architecture...")
    model = replace_lora_with_dyna(model, bottleneck_dim=bottleneck_dim)
    
    # 这一步至关重要：加载你额外保存的 Gate 权重
    # 假设你在 train.py 里把 dyna 参数存到了 dyna_weights.bin
    dyna_weights_path = os.path.join(adapter_path, "dyna_weights.bin")
    
    if os.path.exists(dyna_weights_path):
        print(f"Loading DyNA Gate Weights from {dyna_weights_path}")
        state_dict = torch.load(dyna_weights_path, map_location="cpu")
        # strict=False 是必须的，因为 base model 的参数不在这个 dict 里
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"DyNA weights loaded. (Missing keys is normal for partial loading)")
    else:
        print("WARNING: dyna_weights.bin NOT FOUND! Model will behave like standard LoRA.")
        
    model.eval()
    return model, tokenizer