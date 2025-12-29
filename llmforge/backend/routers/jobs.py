"""
LLMForge Backend - Training Jobs Router
CRUD operations for training jobs.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database import get_db, TrainingJob
from models import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobList,
)
from services.vertex_ai import submit_training_job
from services.cost_calculator import calculate_training_cost

logger = logging.getLogger(__name__)

router = APIRouter()

# Default user ID for demo purposes
# TODO: Replace with proper authentication/authorization in production
# This should use JWT tokens, OAuth, or API keys to identify users
DEFAULT_USER_ID = "demo-user"


@router.post("/", response_model=TrainingJobResponse)
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db),
):
    """Create a new training job."""
    logger.info(f"Creating training job: {job.job_name}")
    
    # Create database record
    db_job = TrainingJob(
        user_id=DEFAULT_USER_ID,
        job_name=job.job_name,
        base_model=job.base_model,
        dataset_path=job.dataset_path,
        status="queued",
        hyperparameters=job.hyperparameters.model_dump(),
        gpu_type=job.gpu_type,
    )
    
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Submit to Vertex AI (async)
    try:
        await submit_training_job(db_job)
        db_job.status = "submitted"
        db.commit()
    except Exception as e:
        logger.error(f"Failed to submit job to Vertex AI: {e}")
        db_job.status = "failed"
        db.commit()
    
    logger.info(f"Training job created: {db_job.id}")
    return db_job


@router.get("/", response_model=TrainingJobList)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
):
    """List all training jobs."""
    query = db.query(TrainingJob).filter(TrainingJob.user_id == DEFAULT_USER_ID)
    
    if status:
        query = query.filter(TrainingJob.status == status)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    jobs = query.order_by(desc(TrainingJob.created_at)).offset(
        (page - 1) * per_page
    ).limit(per_page).all()
    
    return TrainingJobList(
        jobs=jobs,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific training job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == DEFAULT_USER_ID,
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return job


@router.delete("/{job_id}")
async def delete_training_job(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete a training job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == DEFAULT_USER_ID,
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Cancel it first."
        )
    
    db.delete(job)
    db.commit()
    
    logger.info(f"Training job deleted: {job_id}")
    return {"message": "Training job deleted successfully"}


@router.post("/{job_id}/cancel")
async def cancel_training_job(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    """Cancel a training job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == DEFAULT_USER_ID,
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status not in ["queued", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # TODO: Cancel on Vertex AI
    job.status = "cancelled"
    db.commit()
    
    logger.info(f"Training job cancelled: {job_id}")
    return {"message": "Training job cancelled successfully"}


@router.get("/{job_id}/metrics")
async def get_training_metrics(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    """Get training metrics for a job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == DEFAULT_USER_ID,
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if not job.metrics:
        return {"message": "No metrics available yet"}
    
    return job.metrics


@router.get("/{job_id}/cost")
async def get_training_cost(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    """Get cost estimate for a training job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == DEFAULT_USER_ID,
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Calculate cost
    cost = calculate_training_cost(
        gpu_type=job.gpu_type,
        duration_seconds=job.duration_seconds or 0,
        model_size_gb=15.0,  # Approximate for 8B model
    )
    
    return {
        "job_id": str(job.id),
        "gpu_type": job.gpu_type,
        "duration_seconds": job.duration_seconds,
        "estimated_cost_usd": cost if job.duration_seconds else "N/A",
        "actual_cost_usd": float(job.total_cost) if job.total_cost else None,
    }
