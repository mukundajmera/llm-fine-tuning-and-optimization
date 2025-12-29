"""
LLMForge Training Script
Main QLoRA fine-tuning implementation with MLflow tracking.

Usage:
    python train.py <dataset_path> <output_dir> [--config CONFIG_PATH]

Example:
    python train.py gs://my-bucket/data/train.jsonl ./outputs/llama-8b-ft
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

import torch
import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from config import QLoRAConfig
from data_prep import load_and_prepare_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: QLoRAConfig):
    """
    Initialize model with QLoRA configuration.
    
    Args:
        config: QLoRA configuration object
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {config.base_model}")
    
    # BitsAndBytes quantization config (4-bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )
    
    # Determine attention implementation
    attn_implementation = (
        "flash_attention_2" if config.use_flash_attention_2 else "eager"
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params, total_params = model.get_nb_trainable_parameters()
    trainable_percent = 100 * trainable_params / total_params
    logger.info(
        f"Trainable params: {trainable_params:,} "
        f"({trainable_percent:.2f}% of {total_params:,})"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for decoder-only models
    
    return model, tokenizer


def train(
    config: QLoRAConfig,
    dataset_path: str,
    output_dir: str,
    resume_from_checkpoint: bool = False,
):
    """
    Execute QLoRA fine-tuning with MLflow tracking.
    
    Args:
        config: QLoRA configuration object
        dataset_path: Path to training dataset (local or GCS)
        output_dir: Directory to save model and logs
        resume_from_checkpoint: Whether to resume from last checkpoint
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config for reproducibility
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    with mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_params(config.to_dict())
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("output_dir", output_dir)
        
        # Determine model type for formatting
        model_type = "llama" if "llama" in config.base_model.lower() else "mistral"
        
        # Load and prepare dataset
        train_dataset, eval_dataset = load_and_prepare_dataset(
            dataset_path,
            model_type=model_type,
        )
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            optim=config.optim,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=config.save_total_limit,
            bf16=True,
            gradient_checkpointing=config.gradient_checkpointing,
            report_to=["mlflow"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            packing=False,
        )
        
        # Train
        logger.info("🚀 Starting training...")
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None
        )
        
        # Log final metrics
        mlflow.log_metrics({
            "final_train_loss": train_result.training_loss,
            "training_time_seconds": train_result.metrics.get('train_runtime', 0),
            "samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
        })
        
        # Save model and tokenizer
        logger.info(f"💾 Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Upload to GCS if configured
        gcs_bucket = os.getenv("GCS_OUTPUT_BUCKET")
        if gcs_bucket:
            upload_to_gcs(output_dir, gcs_bucket)
        
        logger.info("✅ Training completed successfully!")
        
        return train_result


def upload_to_gcs(local_dir: str, bucket_name: str):
    """Upload model directory to Google Cloud Storage."""
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    model_name = os.path.basename(local_dir)
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            blob_path = f"models/{model_name}/{relative_path}"
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
    
    logger.info(f"✅ Model uploaded to gs://{bucket_name}/models/{model_name}/")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="LLMForge QLoRA Training")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to training dataset (JSONL, local or gs://)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save model and logs"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = QLoRAConfig(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        mlflow_tracking_uri=args.mlflow_uri,
    )
    
    # Run training
    train(
        config=config,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
