"""
LLMForge Model Evaluation
Automated evaluation suite with ROUGE, BLEU, and perplexity metrics.

Usage:
    python evaluate.py <base_model_path> <adapter_path> <eval_dataset_path>
    
Example:
    python evaluate.py meta-llama/Llama-3.1-8B ./outputs/llama-8b-ft ./data/eval.jsonl
"""

import json
import logging
import argparse
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_finetuned_model(
    base_model_path: str,
    adapter_path: str,
    device: str = "auto",
) -> tuple:
    """
    Load fine-tuned model with LoRA adapters.
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapters
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapters: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant response
    if "assistant" in generated_text.lower():
        parts = generated_text.lower().split("assistant")
        if len(parts) > 1:
            generated_text = parts[-1].strip()
    
    return generated_text


def calculate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 2048,
) -> float:
    """Calculate perplexity on a list of texts."""
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].shape[1]
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return float(perplexity)


def evaluate_model(
    model,
    tokenizer,
    eval_dataset_path: str,
    num_samples: Optional[int] = 100,
    verbose: bool = False,
) -> dict:
    """
    Evaluate model on test set with multiple metrics.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        eval_dataset_path: Path to evaluation JSONL file
        num_samples: Maximum number of samples to evaluate
        verbose: Print individual results
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Loading evaluation dataset: {eval_dataset_path}")
    
    # Load evaluation dataset
    eval_data = load_dataset('json', data_files=eval_dataset_path, split='train')
    
    if num_samples and num_samples < len(eval_data):
        eval_data = eval_data.select(range(num_samples))
    
    logger.info(f"Evaluating on {len(eval_data)} samples...")
    
    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction()
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    exact_matches = 0
    
    for i, item in enumerate(eval_data):
        # Create prompt
        instruction = item['instruction']
        input_text = item.get('input', '')
        reference = item['output']
        
        prompt = instruction
        if input_text and input_text.strip():
            prompt += f"\n\n{input_text}"
        
        # Generate response
        generated_text = generate_response(model, tokenizer, prompt)
        
        # Calculate ROUGE scores
        rouge_result = rouge.score(reference, generated_text)
        rouge1_scores.append(rouge_result['rouge1'].fmeasure)
        rouge2_scores.append(rouge_result['rouge2'].fmeasure)
        rougeL_scores.append(rouge_result['rougeL'].fmeasure)
        
        # Calculate BLEU score
        reference_tokens = reference.split()
        generated_tokens = generated_text.split()
        bleu = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            smoothing_function=smoothing.method1
        )
        bleu_scores.append(bleu)
        
        # Check exact match
        if generated_text.lower().strip() == reference.lower().strip():
            exact_matches += 1
        
        if verbose and i < 5:
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Reference: {reference[:100]}...")
            logger.info(f"Generated: {generated_text[:100]}...")
            logger.info(f"ROUGE-L: {rouge_result['rougeL'].fmeasure:.3f}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i+1}/{len(eval_data)} samples...")
    
    # Calculate perplexity on formatted prompts
    logger.info("Calculating perplexity...")
    formatted_texts = [
        f"{item['instruction']}\n{item.get('input', '')}\n{item['output']}"
        for item in eval_data.select(range(min(50, len(eval_data))))
    ]
    perplexity = calculate_perplexity(model, tokenizer, formatted_texts)
    
    # Aggregate metrics
    metrics = {
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rouge_l": float(np.mean(rougeL_scores)),
        "bleu": float(np.mean(bleu_scores)),
        "exact_match_rate": float(exact_matches / len(eval_data)),
        "perplexity": perplexity,
        "num_samples_evaluated": len(eval_data),
    }
    
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 50)
    
    return metrics


def compare_models(
    base_model_path: str,
    adapter_path: str,
    eval_dataset_path: str,
    num_samples: int = 50,
) -> dict:
    """
    Compare fine-tuned model with base model.
    
    Returns metrics for both models for A/B comparison.
    """
    logger.info("Evaluating fine-tuned model...")
    finetuned_model, tokenizer = load_finetuned_model(base_model_path, adapter_path)
    finetuned_metrics = evaluate_model(
        finetuned_model, tokenizer, eval_dataset_path, num_samples
    )
    
    # Clean up fine-tuned model
    del finetuned_model
    torch.cuda.empty_cache()
    
    logger.info("\nEvaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    
    base_metrics = evaluate_model(
        base_model, tokenizer, eval_dataset_path, num_samples
    )
    
    # Calculate improvement
    comparison = {
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
        "improvement": {
            metric: finetuned_metrics[metric] - base_metrics[metric]
            for metric in ["rouge_l", "bleu", "exact_match_rate"]
        }
    }
    
    # Perplexity should decrease (lower is better)
    comparison["improvement"]["perplexity"] = (
        base_metrics["perplexity"] - finetuned_metrics["perplexity"]
    )
    
    logger.info("\n" + "=" * 50)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 50)
    for metric, improvement in comparison["improvement"].items():
        direction = "↑" if improvement > 0 else "↓"
        logger.info(f"{metric}: {improvement:+.4f} {direction}")
    logger.info("=" * 50)
    
    return comparison


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="LLMForge Model Evaluation")
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
        "eval_dataset_path",
        type=str,
        help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for metrics"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with base model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print individual results"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_models(
            args.base_model_path,
            args.adapter_path,
            args.eval_dataset_path,
            args.num_samples,
        )
    else:
        model, tokenizer = load_finetuned_model(
            args.base_model_path,
            args.adapter_path,
        )
        results = evaluate_model(
            model,
            tokenizer,
            args.eval_dataset_path,
            args.num_samples,
            args.verbose,
        )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
