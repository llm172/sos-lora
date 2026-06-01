#!/bin/bash
set -euo pipefail

# Set environment variables if not already set
BASE_MODEL=${BASE_MODEL:-"llama2-7b"}
OUTPUT_PATH=${OUTPUT_PATH:-"output/metamath-sos-lora-seed1024"}
DATA_PATH=${DATA_PATH:-"fxmeng/pissa-dataset"}
SEED=${SEED:-1024}
MASTER_PORT=${MASTER_PORT:-29604}
GPU_INCLUDE=${GPU_INCLUDE:-localhost:2,3}
EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-2}


deepspeed --master_port="$MASTER_PORT" --include="$GPU_INCLUDE" train_sos-lora.py \
  --deepspeed configs/ds_config_zero2_no_offload.json \
  --model_name_or_path "$BASE_MODEL" \
  --full_finetune False \
  --bf16 \
  --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --lora_rank 128 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --num_experts 4 \
  --rank_mode "total" \
  --normalize_scales True \
  --gamma_max 2.5 \
  --lambda_orth 0.006 \
  --orth_delay_ratio 0.15 \
  --orth_ramp_ratio 0.20 \
  --scale_base "expert" \
  --scale_anchor_beta 1e-4 \
  --gate_temperature_start 2.0 \
  --gate_temperature_end 1.0 \
  --gate_anneal_ratio 0.30 \
  --gate_uniform_prior 2e-4 \
  --loraplus_lr_ratio 2.0 \
  --data_path "$DATA_PATH" \
  --sub_task metamath:100000 \
  --dataset_split train \
  --dataset_field instruction output \
  --output_dir "$OUTPUT_PATH" \
  --num_train_epochs 1 \
  --model_max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --report_to "tensorboard" \
  --merge True \
  --seed "$SEED" \
  --save_strategy "no"

if [[ ! -f "$OUTPUT_PATH/config.json" ]]; then
  echo "[ERROR] Training did not produce a complete HF model directory: missing $OUTPUT_PATH/config.json"
  exit 1
fi

echo "Training finished. Starting Evaluation for Seed $SEED..."

CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" python utils/gen_vllm.py \
  --model "$OUTPUT_PATH" \
  --data_path "$DATA_PATH" \
  --sub_task metamath \
  --output_file "$OUTPUT_PATH/result.jsonl"

echo "Calculating Accuracy..."
python utils/test_acc.py --input_file "$OUTPUT_PATH/result.jsonl"
