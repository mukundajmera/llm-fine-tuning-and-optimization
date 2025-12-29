#!/bin/bash
# LLMForge - Train Llama 3.1 8B with QLoRA
# 
# This script trains a Llama 3.1 8B model using QLoRA on A100 40GB GPU.
# Estimated training time: ~3-4 hours for 10K samples, 3 epochs
# Estimated cost: $30-50 on GCP

set -e

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B"
DATASET_PATH="${DATASET_PATH:-./data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/llama-3.1-8b-finetuned}"
MLFLOW_URI="${MLFLOW_URI:-http://localhost:5000}"

# Training hyperparameters (validated defaults)
LORA_R=64
LORA_ALPHA=16
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=4
MAX_SEQ_LENGTH=2048

echo "=============================================="
echo "LLMForge - Llama 3.1 8B QLoRA Training"
echo "=============================================="
echo "Base Model: ${BASE_MODEL}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "MLflow URI: ${MLFLOW_URI}"
echo "=============================================="

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run training
python train.py \
    "${DATASET_PATH}" \
    "${OUTPUT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --learning-rate ${LEARNING_RATE} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --mlflow-uri "${MLFLOW_URI}"

echo "=============================================="
echo "Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=============================================="
