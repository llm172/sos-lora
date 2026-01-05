#!/bin/bash
set -euo pipefail

# Set environment variables if not already set
BASE_MODEL=${BASE_MODEL:-"llama2-7b"}
OUTPUT_PATH=${OUTPUT_PATH:-"output/metamath-sos-lora-seed1024"}
DATA_PATH=${DATA_PATH:-"fxmeng/pissa-dataset"}
SEED=${SEED:-1024}


deepspeed --master_port=29604 --include=localhost:2,3 train_sos-lora.py \n  --deepspeed configs/ds_config_zero2_no_offload.json \n  --model_name_or_path "$BASE_MODEL" \n  --full_finetune False \n  --bf16 \n  --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \n  --lora_rank 128 \n  --lora_alpha 128 \n  --lora_dropout 0.05 \n  --num_experts 4 \n  --rank_mode "total" \n  --normalize_scales True \n  --gamma_max 2.5 \n  --lambda_orth 0.006 \n  --orth_delay_ratio 0.15 \n  --orth_ramp_ratio 0.20 \n  --scale_base "expert" \n  --scale_anchor_beta 1e-4 \n  --gate_temperature_start 2.0 \n  --gate_temperature_end 1.0 \n  --gate_anneal_ratio 0.30 \n  --gate_uniform_prior 2e-4 \n  --loraplus_lr_ratio 2.0 \n  --data_path "$DATA_PATH" \n  --sub_task metamath:100000 \n  --dataset_split train \n  --dataset_field instruction output \n  --output_dir "$OUTPUT_PATH" \n  --num_train_epochs 1 \n  --model_max_length 1024 \n  --per_device_train_batch_size 2 \n  --gradient_accumulation_steps 8 \n  --learning_rate 2e-5 \n  --weight_decay 0. \n  --warmup_ratio 0.03 \n  --lr_scheduler_type "cosine" \n  --logging_steps 1 \n  --gradient_checkpointing True \n  --report_to "tensorboard" \n  --merge True \n  --seed "$SEED" \n  --save_strategy "no"

if [[ ! -f "$OUTPUT_PATH/config.json" ]]; then
  echo "[ERROR] Training did not produce a complete HF model directory: missing $OUTPUT_PATH/config.json"
  exit 1
fi

echo "Training finished. Starting Evaluation for Seed $SEED..."

CUDA_VISIBLE_DEVICES=2 python utils/gen_vllm.py \n  --model "$OUTPUT_PATH" \n  --sub_task metamath \n  --output_file "$OUTPUT_PATH/result.jsonl"

echo "Calculating Accuracy..."
python utils/test_acc.py --input_file "$OUTPUT_PATH/result.jsonl"
