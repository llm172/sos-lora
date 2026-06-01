# SOS-LoRA: Static Orthogonal-Subspace Low-Rank Adaptation with Fixed Multi-Scale Scaling

> **ACL 2026 Main Conference.** This repository contains the official implementation of the ACL 2026 main-conference paper **"SOS-LoRA: Static Orthogonal-Subspace Low-Rank Adaptation with Fixed Multi-Scale Scaling"**.

**Authors:** Yupeng Chang, Yuan Wu, Yi Chang  

## Introduction

We introduce SOS-LoRA, a parameter-efficient fine-tuning (PEFT) method that improves LoRA under a fixed total rank budget. SOS-LoRA decomposes a single low-rank update into multiple static, always-on experts, assigns them fixed multi-scale factors, and encourages cross-expert diversity through orthogonal input-side directions. The trained experts can be merged back into the base model, so inference uses the same architecture and has no additional adapter latency after merging.

Key features of SOS-LoRA:
- Static multi-expert LoRA decomposition under a matched rank budget
- Fixed multi-scale scaling for scale-separated optimization dynamics
- Cross-expert orthogonal initialization and lightweight regularization
- Mergeable adapters with no additional inference-time latency after merging
- Support for Hugging Face Transformers, DeepSpeed training, and vLLM evaluation

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.4.0+
- CUDA 12.1+

### Setup

```bash
git clone https://github.com/llm172/sos-lora.git
cd sos-lora

# Create and activate conda environment
conda create -n sos-lora python=3.10
conda activate sos-lora

# Install CUDA toolkit
conda install nvidia/label/cuda-12.1.0::cuda-toolkit

# Install PyTorch and dependencies
conda install pytorch==2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

```

## Usage

### Training

Run the MetaMath training script for SOS-LoRA:

```bash
# Set environment variables (optional)
export BASE_MODEL="llama2-7b"
export OUTPUT_PATH="output/metamath-sos-lora-seed1024"
export DATA_PATH="pissa-dataset"
export SEED=1024

# Run training
sh scripts/metamath_llama2_7b/run_sos-lora.sh
```

The script trains SOS-LoRA, merges the learned update into the base model when `--merge True`, and then runs the bundled GSM8K-style generation/evaluation pipeline.

### Evaluation

After training, evaluate the model performance:

```bash
# The evaluation is automatically run after training
# Results are saved in $OUTPUT_PATH/result.jsonl
```

You can also run the maintained evaluation utilities manually:

```bash
python inference.py --model "$OUTPUT_PATH"
python utils/gen_vllm.py --model "$OUTPUT_PATH" --data_path "$DATA_PATH" --sub_task metamath --output_file "$OUTPUT_PATH/result.jsonl"
python utils/test_acc.py --input_file "$OUTPUT_PATH/result.jsonl"
python run_harness.py --model "$OUTPUT_PATH" --tasks gsm8k --batch_size 4
```


## Advanced Usage

### Training with Different Models

Modify the `BASE_MODEL` environment variable to use different base models:

```bash
export BASE_MODEL="llama3-8b"
sh scripts/metamath_llama2_7b/run_sos-lora.sh
```

### Training with Custom Dataset

To train with a custom dataset, modify the `DATA_PATH` and `sub_task` parameter in the training script.

### Recommended Defaults

For the MetaMath Llama 2-7B setup, the maintained default is `K=4`, `rank_mode=total`, `gamma_max=2.5`, delayed/ramped orthogonal regularization, mild scale anchoring, and LoRA+-style A-gradient scaling. These defaults preserve the paper's static, mergeable design while providing a slightly stronger and more stable training recipe for high-rank math fine-tuning.

## Code Structure

```
├── configs/              # DeepSpeed and training configurations
├── scripts/              # Training scripts for different models and datasets
│   └── metamath_llama2_7b/
│       └── run_sos-lora.sh  # Main training script for SOS-LoRA
├── utils/                # Utility functions
├── inference.py          # Quick inference with a merged SOS-LoRA model
├── run_harness.py        # lm-evaluation-harness entry point
├── train_sos-lora.py     # Main training code for SOS-LoRA
├── requirements.txt      # Dependencies
└── README.md             # This file
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.
