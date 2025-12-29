"""
LLMForge Backend - Cost Calculator Service
Calculate training and inference costs.

GPU pricing based on GCP pricing (as of Dec 2025).
"""

# GCP GPU pricing ($/hour)
GPU_PRICING = {
    "A100-40GB": 3.67,
    "A100-80GB": 5.24,
    "L4": 1.12,
    "T4": 0.35,
    "V100": 2.48,
}

# GCS storage pricing ($/GB/month)
STORAGE_PRICING = 0.023

# Inference pricing per 1000 tokens (approximation)
INFERENCE_PRICING_PER_1K_TOKENS = {
    "A100-40GB": 0.006,
    "A100-80GB": 0.008,
    "L4": 0.002,
    "T4": 0.001,
}


def calculate_training_cost(
    gpu_type: str,
    duration_seconds: int,
    model_size_gb: float = 15.0,
    storage_months: float = 1.0,
) -> float:
    """
    Calculate total training cost.
    
    Args:
        gpu_type: Type of GPU used
        duration_seconds: Training duration in seconds
        model_size_gb: Size of the model in GB
        storage_months: How long to store the model
    
    Returns:
        Total cost in USD
    """
    # Compute cost
    hours = duration_seconds / 3600
    gpu_rate = GPU_PRICING.get(gpu_type, 3.67)
    compute_cost = gpu_rate * hours
    
    # Storage cost
    storage_cost = model_size_gb * STORAGE_PRICING * storage_months
    
    total = compute_cost + storage_cost
    
    return round(total, 2)


def calculate_inference_cost(
    gpu_type: str,
    requests_per_hour: int,
    avg_tokens_per_request: int = 500,
    hours_active: float = 1.0,
) -> float:
    """
    Calculate inference cost.
    
    Args:
        gpu_type: Type of GPU used
        requests_per_hour: Number of requests per hour
        avg_tokens_per_request: Average tokens per request
        hours_active: Number of hours the deployment is active
    
    Returns:
        Total cost in USD
    """
    # GPU cost
    gpu_rate = GPU_PRICING.get(gpu_type, 1.12)
    gpu_cost = gpu_rate * hours_active
    
    # Approximate throughput (requests per hour per GPU)
    throughput_map = {
        "A100-40GB": 1000,
        "A100-80GB": 1200,
        "L4": 500,
        "T4": 300,
    }
    throughput = throughput_map.get(gpu_type, 500)
    
    # Calculate required GPUs
    required_gpus = max(1, requests_per_hour / throughput)
    
    total = gpu_cost * required_gpus
    
    return round(total, 2)


def calculate_inference_cost_per_token(
    total_tokens: int,
    gpu_type: str = "L4",
) -> float:
    """
    Calculate inference cost per token.
    
    Args:
        total_tokens: Total number of tokens
        gpu_type: Type of GPU used
    
    Returns:
        Cost in USD
    """
    rate = INFERENCE_PRICING_PER_1K_TOKENS.get(gpu_type, 0.002)
    cost = (total_tokens / 1000) * rate
    
    return round(cost, 6)


def estimate_training_cost(
    base_model: str,
    num_samples: int,
    num_epochs: int,
    gpu_type: str = "A100-40GB",
) -> dict:
    """
    Estimate training cost before starting.
    
    Args:
        base_model: Base model name
        num_samples: Number of training samples
        num_epochs: Number of training epochs
        gpu_type: Type of GPU to use
    
    Returns:
        Cost estimate dictionary
    """
    # Estimate training time based on model size and samples
    model_size_factor = {
        "meta-llama/Llama-3.1-8B": 1.0,
        "meta-llama/Llama-3.1-70B": 8.0,
        "mistralai/Mistral-7B-v0.3": 0.9,
    }.get(base_model, 1.0)
    
    # Rough estimate: 1 sample takes ~0.5 seconds on A100
    seconds_per_sample = 0.5 * model_size_factor
    estimated_seconds = num_samples * num_epochs * seconds_per_sample
    
    # Add overhead (warmup, checkpointing, etc.)
    estimated_seconds *= 1.2
    
    estimated_cost = calculate_training_cost(
        gpu_type=gpu_type,
        duration_seconds=int(estimated_seconds),
        model_size_gb=15.0 * model_size_factor,
    )
    
    return {
        "gpu_type": gpu_type,
        "estimated_duration_hours": round(estimated_seconds / 3600, 2),
        "estimated_cost_usd": estimated_cost,
        "breakdown": {
            "compute_cost": round(GPU_PRICING.get(gpu_type, 3.67) * estimated_seconds / 3600, 2),
            "storage_cost": round(15.0 * model_size_factor * STORAGE_PRICING, 2),
        },
    }


def get_gpu_pricing() -> dict:
    """Get current GPU pricing."""
    return GPU_PRICING.copy()


def get_recommended_gpu(
    base_model: str,
    use_case: str = "training",
) -> str:
    """
    Get recommended GPU for a model.
    
    Args:
        base_model: Base model name
        use_case: 'training' or 'inference'
    
    Returns:
        Recommended GPU type
    """
    if "70B" in base_model:
        return "A100-80GB" if use_case == "training" else "A100-40GB"
    
    if use_case == "training":
        return "A100-40GB"
    else:
        return "L4"  # Cost-effective for inference
