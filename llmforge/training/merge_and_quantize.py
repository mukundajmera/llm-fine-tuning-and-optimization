"""
LLMForge Model Merging and Quantization
Post-training optimization for deployment.

This script:
1. Merges LoRA adapters into the base model
2. Optionally quantizes the merged model for efficient inference

Usage:
    python merge_and_quantize.py <base_model> <adapter_path> <output_path> [--quantize]

Example:
    python merge_and_quantize.py meta-llama/Llama-3.1-8B ./outputs/llama-8b-ft ./merged-model
"""

import os
import logging
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dir_size(path: str) -> float:
    """Calculate directory size in GB."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total / (1024 ** 3)


def merge_lora_adapters(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: str = None,
) -> str:
    """
    Merge LoRA adapters into base model for deployment.
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapters
        output_path: Path to save merged model
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_id: Model ID for Hub upload
    
    Returns:
        Path to merged model
    """
    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Use CPU for merging to avoid GPU memory issues
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapters: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging adapters into base model...")
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    
    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)
    
    # Calculate and report size
    merged_size_gb = get_dir_size(output_path)
    logger.info(f"✅ Merged model size: {merged_size_gb:.2f} GB")
    
    # Push to Hub if requested
    if push_to_hub and hub_model_id:
        logger.info(f"Pushing to Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_model_id)
        logger.info(f"✅ Model pushed to: https://huggingface.co/{hub_model_id}")
    
    return output_path


def quantize_for_inference(
    model_path: str,
    output_path: str,
    quantization_method: str = "awq",
) -> str:
    """
    Quantize model for efficient inference.
    
    Args:
        model_path: Path to merged model
        output_path: Path to save quantized model
        quantization_method: One of 'awq', 'gptq', 'gguf'
    
    Returns:
        Path to quantized model
    
    Note: This requires additional dependencies:
    - AWQ: pip install autoawq
    - GPTQ: pip install auto-gptq
    - GGUF: requires llama.cpp
    """
    logger.info(f"Quantizing model with {quantization_method}...")
    
    if quantization_method == "awq":
        try:
            from awq import AutoAWQForCausalLM
            
            logger.info("Loading model for AWQ quantization...")
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # AWQ quantization config
            quant_config = {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            }
            
            logger.info("Running AWQ quantization...")
            model.quantize(tokenizer, quant_config=quant_config)
            
            logger.info(f"Saving quantized model to: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            quantized_size_gb = get_dir_size(output_path)
            logger.info(f"✅ AWQ quantized model size: {quantized_size_gb:.2f} GB")
            
        except ImportError:
            logger.error(
                "AWQ not installed. Install with: pip install autoawq"
            )
            raise
    
    elif quantization_method == "gptq":
        try:
            from transformers import GPTQConfig
            
            logger.info("Loading model for GPTQ quantization...")
            gptq_config = GPTQConfig(
                bits=4,
                group_size=128,
                dataset="c4",
                desc_act=False,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=gptq_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"Saving quantized model to: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            quantized_size_gb = get_dir_size(output_path)
            logger.info(f"✅ GPTQ quantized model size: {quantized_size_gb:.2f} GB")
            
        except ImportError:
            logger.error(
                "GPTQ not available. Ensure auto-gptq is installed."
            )
            raise
    
    else:
        logger.warning(
            f"Quantization method '{quantization_method}' not implemented. "
            "Supported: awq, gptq"
        )
        return model_path
    
    return output_path


def upload_to_gcs(local_path: str, gcs_bucket: str, gcs_prefix: str = "models") -> str:
    """
    Upload model to Google Cloud Storage.
    
    Args:
        local_path: Local path to model directory
        gcs_bucket: GCS bucket name
        gcs_prefix: Prefix for GCS path
    
    Returns:
        GCS path to uploaded model
    """
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    
    model_name = os.path.basename(local_path)
    gcs_path = f"{gcs_prefix}/{model_name}"
    
    logger.info(f"Uploading to gs://{gcs_bucket}/{gcs_path}...")
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = f"{gcs_path}/{relative_path}"
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
    
    gcs_full_path = f"gs://{gcs_bucket}/{gcs_path}"
    logger.info(f"✅ Model uploaded to: {gcs_full_path}")
    
    return gcs_full_path


def main():
    """Main entry point for merging and quantization."""
    parser = argparse.ArgumentParser(
        description="LLMForge Model Merging and Quantization"
    )
    parser.add_argument(
        "base_model_path",
        type=str,
        help="Path to base model"
    )
    parser.add_argument(
        "adapter_path",
        type=str,
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["awq", "gptq"],
        help="Quantization method to apply"
    )
    parser.add_argument(
        "--quantized-output",
        type=str,
        help="Path to save quantized model (default: output_path-quantized)"
    )
    parser.add_argument(
        "--upload-gcs",
        type=str,
        help="GCS bucket to upload model"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="Model ID for Hub upload"
    )
    
    args = parser.parse_args()
    
    # Merge LoRA adapters
    merged_path = merge_lora_adapters(
        args.base_model_path,
        args.adapter_path,
        args.output_path,
        args.push_to_hub,
        args.hub_model_id,
    )
    
    # Optionally quantize
    if args.quantize:
        quantized_output = args.quantized_output or f"{args.output_path}-{args.quantize}"
        quantized_path = quantize_for_inference(
            merged_path,
            quantized_output,
            args.quantize,
        )
        final_path = quantized_path
    else:
        final_path = merged_path
    
    # Optionally upload to GCS
    if args.upload_gcs:
        gcs_path = upload_to_gcs(final_path, args.upload_gcs)
        logger.info(f"Model available at: {gcs_path}")
    
    logger.info("✅ Model processing complete!")


if __name__ == "__main__":
    main()
