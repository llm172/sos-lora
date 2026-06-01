import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_model(model_path, adapter_path=None, torch_dtype=None, merge_adapter=True):
    """Load a merged SOS-LoRA model, or a base model plus PEFT adapter for evaluation."""
    dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
