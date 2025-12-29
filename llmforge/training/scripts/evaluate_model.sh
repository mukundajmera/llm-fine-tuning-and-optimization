#!/bin/bash
# LLMForge - Model Evaluation Script
#
# Evaluate a fine-tuned model with ROUGE, BLEU, and perplexity metrics.

set -e

# Configuration
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-./outputs/llama-3.1-8b-finetuned}"
EVAL_DATASET="${EVAL_DATASET:-./data/eval.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./evaluation_results.json}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

echo "=============================================="
echo "LLMForge - Model Evaluation"
echo "=============================================="
echo "Base Model: ${BASE_MODEL}"
echo "Adapter Path: ${ADAPTER_PATH}"
echo "Eval Dataset: ${EVAL_DATASET}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "=============================================="

# Run evaluation
python evaluate.py \
    "${BASE_MODEL}" \
    "${ADAPTER_PATH}" \
    "${EVAL_DATASET}" \
    --num-samples ${NUM_SAMPLES} \
    --output "${OUTPUT_FILE}" \
    --verbose

echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "=============================================="

# Display results
echo ""
echo "Results:"
cat "${OUTPUT_FILE}"
