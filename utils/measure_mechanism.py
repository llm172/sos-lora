# utils/measure_mechanism.py
import os
import json
import argparse
import torch
import torch.nn.functional as F

def cross_expert_D(A: torch.Tensor, eps: float = 1e-6) -> float:
    # A: [K, r, in]
    K, r, _ = A.shape
    flat = A.reshape(K * r, -1)
    flat = F.normalize(flat, p=2, dim=1, eps=eps)
    G = flat @ flat.t()
    idx = torch.arange(K * r, device=G.device)
    expert_id = idx // r
    same = expert_id.unsqueeze(0).eq(expert_id.unsqueeze(1))
    G = G.masked_fill(same, 0.0)
    denom = max(1, K * (K - 1) * r * r)
    return float((G ** 2).sum().item() / denom)

def gate_mean1(gate_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # mean=1 gate: g = softmax(logits/T) * K
    logits = gate_logits.float()
    T = max(1e-6, float(temperature))
    p = torch.softmax(logits / T, dim=0)
    K = logits.numel()
    return p * K  # [K]

def per_expert_deltaW_norms(A: torch.Tensor, B: torch.Tensor, expert_scales: torch.Tensor,
                           gate_logits: torch.Tensor, gate_temperature: float) -> list:
    # A: [K,r,in], B: [K,out,r], expert_scales: [K,out], gate_logits: [K]
    g = gate_mean1(gate_logits, gate_temperature).to(dtype=A.dtype)  # [K]
    scales = expert_scales * g.unsqueeze(1)  # [K,out]
    norms = []
    K = A.shape[0]
    for k in range(K):
        scaled_B = B[k] * scales[k].unsqueeze(1)    # [out,r]
        dwk = torch.matmul(scaled_B, A[k])          # [out,in]
        norms.append(float(torch.norm(dwk, p="fro").item()))
    return norms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--adapters_filename", type=str, default="sos_adapters.pt")
    args = ap.parse_args()

    pt_path = os.path.join(args.model_dir, args.adapters_filename)
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing adapter snapshot: {pt_path}")

    payload = torch.load(pt_path, map_location="cpu")
    layer_stats = {}
    all_D = []
    all_norms = []

    for layer_name, d in payload.items():
        A = d["lora_A"]            # [K,r,in]
        B = d["lora_B"]            # [K,out,r]
        expert_scales = d["expert_scales"]  # [K,out]
        gate_logits = d["gate_logits"]      # [K]
        gate_temperature = d.get("gate_temperature", 1.0)

        D = cross_expert_D(A)
        norms = per_expert_deltaW_norms(A, B, expert_scales, gate_logits, gate_temperature)

        layer_stats[layer_name] = {
            "D": D,
            "deltaWk_frob": norms,
        }
        all_D.append(D)
        all_norms.append(norms)

    # aggregate
    D_mean = float(sum(all_D) / max(1, len(all_D)))
    # average norms per expert across layers
    K = len(all_norms[0]) if all_norms else 0
    norms_mean = []
    for k in range(K):
        norms_mean.append(float(sum(n[k] for n in all_norms) / max(1, len(all_norms))))

    out = {
        "D_mean": D_mean,
        "deltaWk_frob_mean": norms_mean,
        "num_layers": len(layer_stats),
        "layers": layer_stats,
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
