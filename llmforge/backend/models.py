"""
LLMForge Backend - Pydantic Models
Request/Response schemas for the API.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field


# =============================================================================
# Training Jobs
# =============================================================================

class HyperparametersSchema(BaseModel):
    """Training hyperparameters."""
    lora_r: int = Field(default=64, ge=8, le=128)
    lora_alpha: int = Field(default=16, ge=8, le=128)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    learning_rate: float = Field(default=0.0002, ge=1e-6, le=1e-3)
    num_epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=4, ge=1, le=32)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    max_seq_length: int = Field(default=2048, ge=256, le=8192)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1)


class TrainingJobCreate(BaseModel):
    """Request to create a training job."""
    job_name: str = Field(..., min_length=3, max_length=50)
    base_model: str = Field(
        default="meta-llama/Llama-3.1-8B",
        description="Base model to fine-tune"
    )
    dataset_path: str = Field(..., description="GCS path to training dataset")
    hyperparameters: HyperparametersSchema = Field(default_factory=HyperparametersSchema)
    gpu_type: str = Field(default="A100-40GB", description="GPU type for training")


class TrainingJobResponse(BaseModel):
    """Training job response."""
    id: UUID
    user_id: str
    job_name: str
    base_model: str
    dataset_path: str
    status: str
    hyperparameters: dict
    metrics: Optional[dict] = None
    output_model_path: Optional[str] = None
    total_cost: Optional[Decimal] = None
    duration_seconds: Optional[int] = None
    gpu_type: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TrainingJobList(BaseModel):
    """List of training jobs."""
    jobs: list[TrainingJobResponse]
    total: int
    page: int
    per_page: int


# =============================================================================
# Deployments
# =============================================================================

class VLLMConfigSchema(BaseModel):
    """vLLM configuration."""
    gpu_count: int = Field(default=1, ge=1, le=8)
    max_model_len: int = Field(default=8192, ge=1024, le=32768)
    dtype: str = Field(default="bfloat16")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.5, le=0.95)


class DeploymentCreate(BaseModel):
    """Request to create a deployment."""
    training_job_id: UUID
    deployment_name: str = Field(..., min_length=3, max_length=50)
    vllm_config: VLLMConfigSchema = Field(default_factory=VLLMConfigSchema)
    replicas: int = Field(default=1, ge=1, le=10)


class DeploymentResponse(BaseModel):
    """Deployment response."""
    id: UUID
    training_job_id: UUID
    deployment_name: str
    model_path: str
    vllm_config: dict
    endpoint_url: Optional[str] = None
    status: str
    replicas: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DeploymentList(BaseModel):
    """List of deployments."""
    deployments: list[DeploymentResponse]
    total: int


# =============================================================================
# Evaluations
# =============================================================================

class EvaluationCreate(BaseModel):
    """Request to create an evaluation."""
    training_job_id: UUID
    eval_dataset_path: str = Field(..., description="GCS path to evaluation dataset")
    num_samples: int = Field(default=100, ge=10, le=1000)


class EvaluationMetrics(BaseModel):
    """Evaluation metrics."""
    rouge_l: float
    bleu_score: float
    perplexity: float
    exact_match_rate: float
    num_samples: int


class EvaluationResponse(BaseModel):
    """Evaluation response."""
    id: UUID
    training_job_id: UUID
    eval_dataset_path: str
    metrics: EvaluationMetrics
    created_at: datetime
    
    class Config:
        from_attributes = True


# =============================================================================
# Inference
# =============================================================================

class InferenceRequest(BaseModel):
    """Request for inference."""
    deployment_id: UUID
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class InferenceResponse(BaseModel):
    """Inference response."""
    id: str
    deployment_id: UUID
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: Decimal


class InferenceLogResponse(BaseModel):
    """Inference log entry."""
    id: UUID
    deployment_id: UUID
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: Decimal
    created_at: datetime
    
    class Config:
        from_attributes = True


# =============================================================================
# Cost Tracking
# =============================================================================

class CostSummary(BaseModel):
    """Cost summary."""
    total_training_cost: Decimal
    total_inference_cost: Decimal
    total_cost: Decimal
    training_jobs_count: int
    inference_requests_count: int
    period_start: datetime
    period_end: datetime


# =============================================================================
# Health & Status
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    mlflow: str
    version: str
