import torch
import os
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.lora import Linear as LoraLinear

# -----------------------------------------------------------------------------
# 1. DyNA 类定义 (必须与训练时完全一致)
# -----------------------------------------------------------------------------
class DynaLinear(nn.Module):
    def __init__(self, original_layer: LoraLinear, bottleneck_dim: int = 8):
        super().__init__()
        # 复制引用
        self.base_layer = original_layer.base_layer
        self.lora_A = original_layer.lora_A
        self.lora_B = original_layer.lora_B
        self.scaling = original_layer.scaling
        self.active_adapter = original_layer.active_adapter
        
        # 获取维度
        adapter_name = list(self.lora_A.keys())[0] 
        self.in_features = self.lora_A[adapter_name].in_features
        self.out_features = self.lora_B[adapter_name].out_features

        # 定义 Gate 网络
        self.dyna_norm = nn.LayerNorm(self.in_features, elementwise_affine=True)
        self.dyna_down = nn.Linear(self.in_features, bottleneck_dim, bias=False)
        self.dyna_act = nn.SiLU()
        self.dyna_up = nn.Linear(bottleneck_dim, self.out_features, bias=False)
        self.dyna_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        previous_dtype = x.dtype
        # 1. Base Model Output
        x_base = x.to(self.base_layer.weight.dtype)
        result = self.base_layer(x_base, *args, **kwargs)
        
        # 2. LoRA + DyNA Logic
        for active_adapter in self.active_adapter:
            if active_adapter not in self.lora_A.keys(): continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # Standard LoRA
            x_lora = x.to(lora_A.weight.dtype)
            lora_out = lora_B(lora_A(x_lora)) * scaling
            
            # DyNA Gate
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
    """遍历模型，把 LoRA 层替换为 DyNA 层"""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoraLinear):
                wrapped_layer = DynaLinear(child, bottleneck_dim=bottleneck_dim)
                setattr(module, child_name, wrapped_layer)
    return model

# -----------------------------------------------------------------------------
# 2. 加载函数
# -----------------------------------------------------------------------------
def load_eval_model(base_model_path, adapter_path, bottleneck_dim=8):
    print(f"Loading Base Model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading LoRA Adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print(f"Transforming to DyNA Architecture (r'={bottleneck_dim})...")
    model = replace_lora_with_dyna(model, bottleneck_dim=bottleneck_dim)
    
    # 加载 Gate 权重
    dyna_weights_path = os.path.join(adapter_path, "dyna_weights.bin")
    if os.path.exists(dyna_weights_path):
        print(f"Loading DyNA Gate Weights from {dyna_weights_path}...")
        state_dict = torch.load(dyna_weights_path, map_location="cpu")
        # strict=False 允许只加载 dyna 参数
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"DyNA Weights Loaded. Missing keys count (normal): {len(missing)}")
    else:
        print("!!! WARNING: dyna_weights.bin NOT FOUND! !!!")
        print("Running as Standard LoRA (Gate=0). Check your training/save script.")
        
    model.eval()
    return model, tokenizer