#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to a Hugging Face model id or local model path}"
: "${DATASET:?Set DATASET to the converted ms-swift JSONL path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the checkpoint output directory}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
AGENT_TEMPLATE="${AGENT_TEMPLATE:-qwen3_5}"
TUNER_TYPE="${TUNER_TYPE:-lora}"
MAX_LENGTH="${MAX_LENGTH:-32000}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DEEPSPEED="${DEEPSPEED:-zero2}"
REPORT_TO="${REPORT_TO:-none}"

mkdir -p "$OUTPUT_DIR"

NPROC_PER_NODE="$NPROC_PER_NODE" swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --agent_template "$AGENT_TEMPLATE" \
  --tuner_type "$TUNER_TYPE" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --torch_dtype bfloat16 \
  --attn_impl "$ATTN_IMPL" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$MICRO_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --gradient_checkpointing true \
  --learning_rate "$LEARNING_RATE" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy delete \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-2}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --deepspeed "$DEEPSPEED" \
  --report_to "$REPORT_TO" \
  "$@"
