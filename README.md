# SOS-LoRA: Scalable and Orthogonal Subspace Low-Rank Adaptation

## Introduction

We introduce SOS-LoRA, a novel parameter-efficient fine-tuning (PEFT) method that enhances the performance of large language models through scalable and orthogonal subspace adaptation. SOS-LoRA addresses the limitations of traditional LoRA by leveraging multiple expert adapters with orthogonal constraints, enabling more efficient and effective parameter utilization.

Key features of SOS-LoRA:
- Parameter-efficient fine-tuning with orthogonal subspace constraints
- Multiple expert adapters for scalable adaptation
- Fast convergence with improved performance
- Compatibility with various model architectures
- Support for quantization and deepspeed training

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.4.0+
- CUDA 12.1+

### Setup

```bash
git clone <repository-url>
cd <repository-directory>

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

# Download dataset
huggingface-cli download --repo-type dataset --resume-download corresponding-dataset --local-dir xxx

```

## Usage

### Training

Run the training script for SOS-LoRA:

```bash
# Set environment variables (optional)
export BASE_MODEL="llama2-7b"
export OUTPUT_PATH="output/metamath-sos-lora-seed1024"
export DATA_PATH="pissa-dataset"
export SEED=1024

# Run training
sh scripts/metamath_llama2_7b/run_sos-lora.sh
```

### Evaluation

After training, evaluate the model performance:

```bash
# The evaluation is automatically run after training
# Results are saved in $OUTPUT_PATH/result.jsonl
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

## Code Structure

```
├── configs/              # DeepSpeed and training configurations
├── scripts/              # Training scripts for different models and datasets
│   └── metamath_llama2_7b/
│       └── run_sos-lora.sh  # Main training script for SOS-LoRA
├── utils/                # Utility functions
├── train_sos-lora.py     # Main training code for SOS-LoRA
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
