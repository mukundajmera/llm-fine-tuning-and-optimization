"""
LLMForge Backend - Vertex AI Service
Submit and manage training jobs on GCP Vertex AI.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# GCP Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
STAGING_BUCKET = os.getenv("VERTEX_AI_STAGING_BUCKET")
TRAINING_IMAGE = os.getenv(
    "TRAINING_IMAGE",
    "gcr.io/llmforge/training:latest"
)


async def submit_training_job(job) -> Optional[str]:
    """
    Submit a training job to Vertex AI.
    
    Args:
        job: TrainingJob database object
    
    Returns:
        Vertex AI job name or None if failed
    """
    if not PROJECT_ID:
        logger.warning("GCP_PROJECT_ID not set. Skipping Vertex AI submission.")
        return None
    
    try:
        from google.cloud import aiplatform
        
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=STAGING_BUCKET,
        )
        
        # Create custom job
        custom_job = aiplatform.CustomJob(
            display_name=f"llmforge-{job.job_name}",
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": get_machine_type(job.gpu_type),
                        "accelerator_type": get_accelerator_type(job.gpu_type),
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": TRAINING_IMAGE,
                        "args": [
                            job.dataset_path,
                            f"gs://{STAGING_BUCKET}/outputs/{job.id}",
                            "--base-model", job.base_model,
                            "--lora-r", str(job.hyperparameters.get("lora_r", 64)),
                            "--lora-alpha", str(job.hyperparameters.get("lora_alpha", 16)),
                            "--learning-rate", str(job.hyperparameters.get("learning_rate", 0.0002)),
                            "--num-epochs", str(job.hyperparameters.get("num_epochs", 3)),
                            "--batch-size", str(job.hyperparameters.get("batch_size", 4)),
                        ],
                        "env": [
                            {"name": "HF_TOKEN", "value": os.getenv("HF_TOKEN", "")},
                            {"name": "MLFLOW_TRACKING_URI", "value": os.getenv("MLFLOW_TRACKING_URI", "")},
                        ],
                    },
                }
            ],
        )
        
        # Submit job
        custom_job.submit()
        
        logger.info(f"Vertex AI job submitted: {custom_job.display_name}")
        return custom_job.name
        
    except ImportError:
        logger.warning("google-cloud-aiplatform not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to submit Vertex AI job: {e}")
        raise


def get_machine_type(gpu_type: str) -> str:
    """Get GCP machine type for GPU type."""
    machine_types = {
        "A100-40GB": "a2-highgpu-1g",
        "A100-80GB": "a2-ultragpu-1g",
        "L4": "g2-standard-8",
        "T4": "n1-highmem-8",
    }
    return machine_types.get(gpu_type, "a2-highgpu-1g")


def get_accelerator_type(gpu_type: str) -> str:
    """Get GCP accelerator type for GPU type."""
    accelerator_types = {
        "A100-40GB": "NVIDIA_TESLA_A100",
        "A100-80GB": "NVIDIA_A100_80GB",
        "L4": "NVIDIA_L4",
        "T4": "NVIDIA_TESLA_T4",
    }
    return accelerator_types.get(gpu_type, "NVIDIA_TESLA_A100")


async def get_job_status(vertex_job_name: str) -> dict:
    """
    Get status of a Vertex AI job.
    
    Args:
        vertex_job_name: Vertex AI job resource name
    
    Returns:
        Job status dict
    """
    if not PROJECT_ID:
        return {"status": "unknown", "error": "GCP_PROJECT_ID not set"}
    
    try:
        from google.cloud import aiplatform
        
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        job = aiplatform.CustomJob.get(vertex_job_name)
        
        return {
            "name": job.name,
            "display_name": job.display_name,
            "state": job.state.name,
            "create_time": str(job.create_time),
            "start_time": str(job.start_time) if job.start_time else None,
            "end_time": str(job.end_time) if job.end_time else None,
        }
        
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return {"status": "error", "error": str(e)}


async def cancel_job(vertex_job_name: str) -> bool:
    """
    Cancel a Vertex AI job.
    
    Args:
        vertex_job_name: Vertex AI job resource name
    
    Returns:
        True if cancelled successfully
    """
    if not PROJECT_ID:
        return False
    
    try:
        from google.cloud import aiplatform
        
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        job = aiplatform.CustomJob.get(vertex_job_name)
        job.cancel()
        
        logger.info(f"Vertex AI job cancelled: {vertex_job_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return False
