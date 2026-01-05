import copy
import math
import logging
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import Trainer, TrainerCallback
from datasets import load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)

# ============================================================
# 0) Reproducibility utilities
# ============================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_norm_module(m: nn.Module) -> bool:
    """Covers LayerNorm and common RMSNorm variants in HF models."""
    if isinstance(m, nn.LayerNorm):
        return True
    name = m.__class__.__name__.lower()
    return ("rmsnorm" in name) or name.endswith("norm")


# ============================================================
# 1) Orthogonality gradient (DeepSpeed-stable injection)
# ============================================================
def compute_ortho_grad_on_tensor(
    tensor: torch.Tensor,
    num_experts: int,
    r: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cross-expert orthogonality gradient on input-side directions.
    tensor: [K, r, in]
    """
    T_temp = tensor.detach().requires_grad_(True)
    with torch.enable_grad():
        flat = T_temp.reshape(num_experts * r, -1)  # [K*r, in]
        flat = F.normalize(flat, p=2, dim=1, eps=eps)
        G = flat @ flat.t()  # [K*r, K*r]

        device = G.device
        idx = torch.arange(num_experts * r, device=device)
        expert_id = idx // r
        same_expert = expert_id.unsqueeze(0).eq(expert_id.unsqueeze(1))
        G = G.masked_fill(same_expert, 0.0)

        denom = max(1, num_experts * (num_experts - 1) * r * r)
        loss = (G ** 2).sum() / denom

    return torch.autograd.grad(loss, T_temp)[0]


class LoRA_DualOrtho_Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, current_lambda, num_experts, r):
        ctx.save_for_backward(x, A)
        ctx.current_lambda = float(current_lambda)
        ctx.num_experts = int(num_experts)
        ctx.r = int(r)
        return torch.einsum("bsi,nri->bsnr", x, A)

    @staticmethod
    def backward(ctx, grad_h):
        x, A = ctx.saved_tensors
        grad_A_task = torch.einsum("bsnr,bsi->nri", grad_h, x)
        grad_x = torch.einsum("bsnr,nri->bsi", grad_h, A)

        if ctx.current_lambda > 0:
            grad_A_ortho = compute_ortho_grad_on_tensor(A, ctx.num_experts, ctx.r)
            grad_A_total = grad_A_task + (ctx.current_lambda * grad_A_ortho)
        else:
            grad_A_total = grad_A_task

        return grad_x, grad_A_total, None, None, None


# ============================================================
# 2) Global gate: mean=1 softmax (+ optional uniform-prior gradient)
#    - internal fp32 p for stability under bf16 AMP
# ============================================================
class GateMean1WithUniformPrior(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, temperature, uniform_prior):
        T = float(temperature)
        K = logits.numel()

        # fp32 softmax (stable), keep p_fp32 for backward
        p_fp32 = torch.softmax(logits.float() / T, dim=0)  # [K] fp32
        ctx.save_for_backward(p_fp32)
        ctx.temperature = T
        ctx.uniform_prior = float(uniform_prior)
        ctx.K = K
        ctx.logits_dtype = logits.dtype

        # output dtype follows logits (bf16/fp16)
        out = (p_fp32 * K).to(dtype=logits.dtype, device=logits.device)
        return out

    @staticmethod
    def backward(ctx, grad_g):
        (p_fp32,) = ctx.saved_tensors
        T = ctx.temperature
        beta = ctx.uniform_prior
        K = ctx.K
        out_dtype = ctx.logits_dtype

        # compute in fp32 then cast back
        grad_g_fp32 = grad_g.float()
        grad_p = grad_g_fp32 * K
        dot = (grad_p * p_fp32).sum()
        grad_logits_task = (grad_p - dot) * p_fp32 / T

        if beta > 0:
            grad_logits_prior = beta * (p_fp32 - (1.0 / K)) / T
            grad_logits = grad_logits_task + grad_logits_prior
        else:
            grad_logits = grad_logits_task

        return grad_logits.to(dtype=out_dtype), None, None


# ============================================================# 3) SOS-LoRA layer# ============================================================
class SOSLoRALayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        gamma_max: float = 2.5,
        rank_mode: str = "total",
        normalize_scales: Optional[bool] = None,
        dropout_position: str = "postA",
        orth_loss_mode: str = "rowwise_cross",
        scale_base: str = "expert",
        scale_anchor_beta: float = 0.0,
        gate_temperature_start: float = 2.0,
        gate_uniform_prior: float = 0.0,
        loraplus_lr_ratio: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.num_experts = int(num_experts)
        self.rank_mode = str(rank_mode)
        self.dropout = nn.Dropout(p=float(dropout))
        self.dropout_position = str(dropout_position)

        self.register_buffer("current_lambda", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("gate_temperature", torch.tensor(float(gate_temperature_start), dtype=torch.float32))
        self.gate_uniform_prior = float(gate_uniform_prior)

        # Rank logic
        if self.rank_mode == "total":
            self.r_tot = int(r)
            if self.r_tot % self.num_experts != 0:
                raise ValueError(
                    f"rank_mode=total requires r_tot divisible by num_experts, got r_tot={self.r_tot}, K={self.num_experts}"
                )
            self.r = self.r_tot // self.num_experts
        else:
            self.r = int(r)
            self.r_tot = self.r * self.num_experts

        # Multi-scale prior
        gamma = torch.linspace(1.0, float(gamma_max), steps=self.num_experts)
        if normalize_scales is None:
            normalize_scales = (self.rank_mode == "total")
        if normalize_scales:
            gamma = gamma / gamma.mean()

        scale_base = str(scale_base).lower()
        if scale_base == "total":
            base_scaling = float(alpha) / float(self.r_tot)
        else:
            # keep your SOTA-strength base scaling
            base_scaling = float(alpha) / float(self.r)

        base_scales = (gamma * base_scaling).unsqueeze(1).repeat(1, self.out_features)  # [K,out]
        self.register_buffer("base_scales", base_scales)

        # Channel-wise trainable calibration
        self.expert_scales = nn.Parameter(self.base_scales.clone())

        # Optional anchor
        self.scale_anchor_beta = float(scale_anchor_beta)
        if self.scale_anchor_beta > 0:
            def _anchor_hook(grad):
                return grad + self.scale_anchor_beta * (
                    self.expert_scales.detach() - self.base_scales.to(self.expert_scales.dtype)
                )
            self.expert_scales.register_hook(_anchor_hook)

        # Static global gate logits (symmetry-breaking, zero-mean)
        init = torch.randn(self.num_experts) * 0.05
        init = init - init.mean()
        self.lora_gate_logits = nn.Parameter(init)

        # LoRA weights: A diverse dirs, B=0 => deltaW=0 at init (paper-friendly & stable) :contentReference[oaicite:3]{index=3}
        self.lora_A = nn.Parameter(torch.empty(self.num_experts, self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.num_experts, self.out_features, self.r))

        JOINT_QR_GAIN = math.sqrt(2.0)
        with torch.no_grad():
            Kr = self.num_experts * self.r
            if Kr <= self.in_features:
                G = torch.randn(self.in_features, Kr, device=self.lora_A.device)
                Q, _ = torch.linalg.qr(G, mode="reduced")
                dirs = Q.transpose(0, 1).contiguous().reshape(self.num_experts, self.r, self.in_features)
                self.lora_A.copy_(dirs * JOINT_QR_GAIN)
            else:
                for i in range(self.num_experts):
                    nn.init.orthogonal_(self.lora_A[i], gain=JOINT_QR_GAIN)
            nn.init.zeros_(self.lora_B)

        # Optional LoRA+ dynamics (A-grad scaling)
        self.loraplus_lr_ratio = float(loraplus_lr_ratio)
        if self.loraplus_lr_ratio > 1.0:
            def _scale_A_grad(grad):
                return grad / self.loraplus_lr_ratio
            self.lora_A.register_hook(_scale_A_grad)

    def _gate_mean1(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        T = float(self.gate_temperature.item())
        logits = self.lora_gate_logits.to(device=device, dtype=dtype)
        return GateMean1WithUniformPrior.apply(logits, T, self.gate_uniform_prior)  # [K], mean=1

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        x_in = x.to(self.lora_A.dtype)

        if self.dropout.p > 0 and self.dropout_position == "preA":
            x_in = self.dropout(x_in)

        if self.training and float(self.current_lambda.item()) > 0:
            h = LoRA_DualOrtho_Op.apply(x_in, self.lora_A, self.current_lambda, self.num_experts, self.r)
        else:
            h = torch.einsum("bsi,nri->bsnr", x_in, self.lora_A)

        if self.dropout.p > 0 and self.dropout_position == "postA":
            h = self.dropout(h)

        delta_h_per_expert = torch.einsum("bsnr,nro->bsno", h, self.lora_B.transpose(1, 2))  # [b,s,K,out]

        g = self._gate_mean1(delta_h_per_expert.dtype, delta_h_per_expert.device)  # [K]
        scales = self.expert_scales.to(delta_h_per_expert.dtype) * g.unsqueeze(1)  # [K,out]
        scales = scales.view(1, 1, self.num_experts, self.out_features)

        delta_h = (delta_h_per_expert * scales).sum(dim=2)
        return result + delta_h.to(result.dtype)

    def merge_and_unload(self) -> nn.Linear:
        device = self.lora_A.device
        dtype = self.lora_A.dtype

        g = self._gate_mean1(dtype, device)  # [K]
        scales = self.expert_scales.to(device=device, dtype=dtype) * g.unsqueeze(1)  # [K,out]

        delta_w = torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)
        for i in range(self.num_experts):
            scaled_B = self.lora_B[i].to(device=device, dtype=dtype) * scales[i].unsqueeze(1)  # [out,r]
            delta_w += torch.matmul(scaled_B, self.lora_A[i].to(device=device, dtype=dtype))    # [out,in]

        with torch.no_grad():
            self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
        return self.base_layer


# ============================================================
# 4) Injection (collect then replace; strict matching)
# ============================================================
def inject_soslora(model: nn.Module, target_modules: Iterable[str], **kwargs) -> nn.Module:
    target_set = {t.strip() for t in target_modules if t.strip()}

    to_replace: List[Tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if name.split(".")[-1] not in target_set:
            continue
        if not isinstance(module, nn.Linear) or isinstance(module, SOSLoRALayer):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        child_name = name.split(".")[-1]
        to_replace.append((parent, child_name, module))

    for parent, child_name, lin in to_replace:
        new_module = SOSLoRALayer(base_layer=lin, **kwargs)
        new_module.to(lin.weight.device)

        if lin.weight.dtype == torch.float16:
            new_module.half()
        elif lin.weight.dtype == torch.bfloat16:
            new_module.bfloat16()

        setattr(parent, child_name, new_module)

    for n, p in model.named_parameters():
        if ("lora_" in n) or ("expert_scales" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    return model


# ============================================================
# 5) Training-only schedules
# ============================================================
class OrthScheduleCallback(TrainerCallback):
    def __init__(self, lambda_max: float, delay_ratio: float, ramp_ratio: float):
        self.lambda_max = float(lambda_max)
        self.delay_ratio = float(delay_ratio)
        self.ramp_ratio = float(ramp_ratio)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if self.lambda_max <= 0 or state.max_steps <= 0:
            return

        step = state.global_step
        delay_steps = int(state.max_steps * self.delay_ratio)
        ramp_steps = int(state.max_steps * self.ramp_ratio)

        if step < delay_steps:
            current_val = 0.0
        elif ramp_steps <= 0:
            current_val = self.lambda_max
        else:
            t = (step - delay_steps) / max(1, ramp_steps)
            t = max(0.0, min(1.0, t))
            current_val = self.lambda_max * t

        model_inner = model.module if hasattr(model, "module") else model
        for m in model_inner.modules():
            if isinstance(m, SOSLoRALayer):
                m.current_lambda.fill_(current_val)


class GateTemperatureScheduleCallback(TrainerCallback):
    def __init__(self, T_start: float, T_end: float, anneal_ratio: float):
        self.T_start = float(T_start)
        self.T_end = float(T_end)
        self.anneal_ratio = float(anneal_ratio)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.max_steps <= 0:
            return

        anneal_steps = int(state.max_steps * self.anneal_ratio)
        step = state.global_step

        if anneal_steps <= 0:
            T = self.T_end
        else:
            t = min(1.0, max(0.0, step / max(1, anneal_steps)))
            T = self.T_start + t * (self.T_end - self.T_start)

        model_inner = model.module if hasattr(model, "module") else model
        for m in model_inner.modules():
            if isinstance(m, SOSLoRALayer):
                m.gate_temperature.fill_(float(T))


# ============================================================
# 6) Data / Training
# ============================================================
IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"

    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    num_experts: int = 4
    gamma_max: float = 2.5
    rank_mode: str = "total"
    normalize_scales: bool = None
    dropout_position: str = "postA"
    orth_loss_mode: str = "rowwise_cross"

    lambda_orth: float = 0.01
    orth_delay_ratio: float = 0.15
    orth_ramp_ratio: float = 0.20

    scale_base: str = "expert"
    scale_anchor_beta: float = 0.0

    gate_temperature_start: float = 2.0
    gate_temperature_end: float = 1.0
    gate_anneal_ratio: float = 0.30
    gate_uniform_prior: float = 0.0

    loraplus_lr_ratio: float = 1.0

    data_path: str = None
    sub_task: List[str] = None
    dataset_split: str = "train"
    dataset_field: List[str] = None
    shuffle_dataset: bool = False
    model_max_length: int = 512
    merge: bool = False
    full_finetune: bool = False


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=i)) for i in examples[query]]
    targets = [f"{o}\n{tokenizer.eos_token}" for o in examples[response]]
    return preprocess(sources, targets, tokenizer)


def _tokenize_fn(strings, tokenizer):
    tokenized = [tokenizer(text, max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    return dict(
        input_ids=[np.array(t.input_ids) for t in tokenized],
        input_ids_lens=[len(t.input_ids) for t in tokenized],
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(s, tokenizer) for s in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in labels], batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def build_model(script_args):
    compute_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    if script_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = inject_soslora(
        model,
        target_modules=script_args.target_modules.split(","),
        r=script_args.lora_rank,
        alpha=script_args.lora_alpha,
        num_experts=script_args.num_experts,
        dropout=script_args.lora_dropout,
        gamma_max=script_args.gamma_max,
        rank_mode=script_args.rank_mode,
        normalize_scales=script_args.normalize_scales,
        dropout_position=script_args.dropout_position,
        orth_loss_mode=script_args.orth_loss_mode,
        scale_base=script_args.scale_base,
        scale_anchor_beta=script_args.scale_anchor_beta,
        gate_temperature_start=script_args.gate_temperature_start,
        gate_uniform_prior=script_args.gate_uniform_prior,
        loraplus_lr_ratio=script_args.loraplus_lr_ratio,
    )

    # Only cast real normalization layers to fp32 (AMP-friendly practice)
    for _, m in model.named_modules():
        if is_norm_module(m):
            m.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO if script_args.local_rank in [-1, 0] else logging.WARN)

    set_global_seed(int(script_args.seed))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(script_args)

    all_training_dataset = []
    for task in script_args.sub_task:
        task_name, split = (task.split(":") if ":" in task else (task, script_args.dataset_split))
        split = f"{script_args.dataset_split}[:{split}]" if ":" in task else split
        all_training_dataset.append(load_dataset(script_args.data_path, data_dir=task_name, split=split))

    train_dataset = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset:
        train_dataset = train_dataset.shuffle(seed=script_args.seed)

    train_dataset = train_dataset.map(
        train_tokenize_function,
        batched=True,
        num_proc=32,
        remove_columns=train_dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )

    callbacks = [
        OrthScheduleCallback(script_args.lambda_orth, script_args.orth_delay_ratio, script_args.orth_ramp_ratio),
        GateTemperatureScheduleCallback(
            script_args.gate_temperature_start,
            script_args.gate_temperature_end,
            script_args.gate_anneal_ratio,
        ),
    ]

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
        callbacks=callbacks,
    )
    trainer.train()

    if script_args.merge and script_args.local_rank == 0:
        save_path = script_args.output_dir
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        for name, module in list(model_to_save.named_modules()):
            if isinstance(module, SOSLoRALayer):
                parent_name = ".".join(name.split(".")[:-1])
                parent = model_to_save.get_submodule(parent_name) if parent_name else model_to_save
                setattr(parent, name.split(".")[-1], module.merge_and_unload())
        model_to_save.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    train()
