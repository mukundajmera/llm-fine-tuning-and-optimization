"""
LLMForge Training Configuration
QLoRA hyperparameters with validated defaults and source citations.

Sources:
- [1] QLoRA paper: https://arxiv.org/abs/2305.14314
- [2] Unsloth documentation: https://github.com/unslothai/unsloth
- [3] Hugging Face PEFT: https://huggingface.co/docs/peft
- [4] TRL SFTTrainer: https://huggingface.co/docs/trl
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QLoRAConfig:
    """QLoRA training configuration with validated defaults.
    
    All hyperparameters have been validated against current best practices
    and documented sources.
    """
    
    # ==========================================================================
    # Model Selection
    # ==========================================================================
    base_model: str = "meta-llama/Llama-3.1-8B"
    # Supported: "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B", 
    #            "mistralai/Mistral-7B-v0.3"
    
    # ==========================================================================
    # LoRA Hyperparameters [1][3]
    # ==========================================================================
    lora_r: int = 64  # Rank: 8-64 typical, 64 = high quality
    lora_alpha: int = 16  # Scaling factor: commonly r/4 to r*2
    lora_dropout: float = 0.05  # Dropout: 0.05-0.1 recommended
    lora_target_modules: Optional[list[str]] = None  # Auto-set in __post_init__
    
    # ==========================================================================
    # Quantization (QLoRA Specific) [1]
    # ==========================================================================
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16 > float16 for training
    bnb_4bit_quant_type: str = "nf4"  # NF4 > INT4 for LLMs
    bnb_4bit_use_double_quant: bool = True  # Nested quantization, saves ~0.4GB
    
    # ==========================================================================
    # Training Hyperparameters [2][4]
    # ==========================================================================
    learning_rate: float = 2e-4  # Per Unsloth recommendation
    num_train_epochs: int = 3  # 1-3 typical, more risks overfitting
    per_device_train_batch_size: int = 4  # Fits A100 40GB with 8B model
    gradient_accumulation_steps: int = 4  # Effective batch = 4*4=16
    max_seq_length: int = 2048  # Context window for training
    
    # ==========================================================================
    # Optimization
    # ==========================================================================
    optim: str = "paged_adamw_8bit"  # Memory-efficient AdamW
    warmup_ratio: float = 0.1  # 10% of steps for warmup
    lr_scheduler_type: str = "cosine"  # Cosine decay recommended
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # ==========================================================================
    # Memory Optimization
    # ==========================================================================
    gradient_checkpointing: bool = True  # Save 30% memory, 15% slower
    use_flash_attention_2: bool = True  # 2-3x faster training
    
    # ==========================================================================
    # Logging & Saving
    # ==========================================================================
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3  # Keep only 3 checkpoints
    
    # ==========================================================================
    # MLflow Tracking
    # ==========================================================================
    mlflow_experiment_name: str = "llama-3.1-8b-finetuning"
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Set default LoRA target modules based on model architecture
        if self.lora_target_modules is None:
            # Apply LoRA to all attention + FFN layers (best performance)
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"       # FFN
            ]
        
        # Validate learning rate
        if self.learning_rate > 5e-4:
            print(
                f"⚠️  WARNING: Learning rate {self.learning_rate} is high. "
                f"Recommended: 2e-4. May cause instability."
            )
        elif self.learning_rate < 1e-5:
            print(
                f"⚠️  WARNING: Learning rate {self.learning_rate} is very low. "
                f"Training may be slow."
            )
        
        # Validate batch size vs gradient accumulation
        effective_batch = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )
        if effective_batch < 8:
            print(
                f"⚠️  WARNING: Effective batch size {effective_batch} is low. "
                f"Recommended: 16-32 for stable training."
            )
        
        # Validate epochs
        if self.num_train_epochs > 5:
            print(
                f"⚠️  WARNING: {self.num_train_epochs} epochs may cause "
                f"overfitting. Monitor eval loss closely."
            )
        
        # Validate LoRA rank
        if self.lora_r < 8:
            print(
                f"⚠️  WARNING: LoRA rank {self.lora_r} is very low. "
                f"Model capacity may be limited."
            )
        elif self.lora_r > 128:
            print(
                f"⚠️  WARNING: LoRA rank {self.lora_r} is very high. "
                f"May increase memory usage significantly."
            )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "base_model": self.base_model,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "optim": self.optim,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_flash_attention_2": self.use_flash_attention_2,
        }
    
    @classmethod
    def for_llama_3_1_8b(cls) -> "QLoRAConfig":
        """Optimized config for Llama 3.1 8B on A100 40GB."""
        return cls(
            base_model="meta-llama/Llama-3.1-8B",
            lora_r=64,
            lora_alpha=16,
            per_device_train_batch_size=4,
            max_seq_length=2048,
        )
    
    @classmethod
    def for_llama_3_1_70b(cls) -> "QLoRAConfig":
        """Optimized config for Llama 3.1 70B on A100 80GB."""
        return cls(
            base_model="meta-llama/Llama-3.1-70B",
            lora_r=32,  # Lower rank for larger model
            lora_alpha=16,
            per_device_train_batch_size=1,  # Smaller batch for 70B
            gradient_accumulation_steps=16,  # Compensate with more accumulation
            max_seq_length=1024,  # Shorter context for memory
        )
    
    @classmethod
    def for_mistral_7b(cls) -> "QLoRAConfig":
        """Optimized config for Mistral 7B v0.3 on A100 40GB."""
        return cls(
            base_model="mistralai/Mistral-7B-v0.3",
            lora_r=64,
            lora_alpha=16,
            per_device_train_batch_size=4,
            max_seq_length=2048,
            lora_target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        )


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    dataset_path: str = ""  # GCS or local path
    split_ratio: float = 0.1  # Train/eval split ratio
    max_samples: Optional[int] = None  # Limit samples for debugging
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference configuration for vLLM deployment."""
    
    model_path: str = ""
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    
    # Sampling defaults
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
