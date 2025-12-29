"""
LLMForge Backend - Deployments Router
CRUD operations for model deployments.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database import get_db, Deployment, TrainingJob
from models import (
    DeploymentCreate,
    DeploymentResponse,
    DeploymentList,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=DeploymentResponse)
async def create_deployment(
    deployment: DeploymentCreate,
    db: Session = Depends(get_db),
):
    """Create a new model deployment."""
    logger.info(f"Creating deployment: {deployment.deployment_name}")
    
    # Check if training job exists and is completed
    training_job = db.query(TrainingJob).filter(
        TrainingJob.id == deployment.training_job_id
    ).first()
    
    if not training_job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if training_job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Training job must be completed before deployment"
        )
    
    if not training_job.output_model_path:
        raise HTTPException(
            status_code=400,
            detail="Training job has no output model path"
        )
    
    # Check if deployment name is unique
    existing = db.query(Deployment).filter(
        Deployment.deployment_name == deployment.deployment_name
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Deployment name already exists"
        )
    
    # Create deployment
    db_deployment = Deployment(
        training_job_id=deployment.training_job_id,
        deployment_name=deployment.deployment_name,
        model_path=training_job.output_model_path,
        vllm_config=deployment.vllm_config.model_dump(),
        status="deploying",
        replicas=deployment.replicas,
    )
    
    db.add(db_deployment)
    db.commit()
    db.refresh(db_deployment)
    
    # TODO: Trigger actual deployment to GKE
    
    logger.info(f"Deployment created: {db_deployment.id}")
    return db_deployment


@router.get("/", response_model=DeploymentList)
async def list_deployments(
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """List all deployments."""
    query = db.query(Deployment)
    
    if status:
        query = query.filter(Deployment.status == status)
    
    deployments = query.order_by(desc(Deployment.created_at)).all()
    total = query.count()
    
    return DeploymentList(deployments=deployments, total=total)


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific deployment."""
    deployment = db.query(Deployment).filter(
        Deployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    return deployment


@router.delete("/{deployment_id}")
async def delete_deployment(
    deployment_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete a deployment."""
    deployment = db.query(Deployment).filter(
        Deployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # TODO: Delete from GKE
    
    db.delete(deployment)
    db.commit()
    
    logger.info(f"Deployment deleted: {deployment_id}")
    return {"message": "Deployment deleted successfully"}


@router.post("/{deployment_id}/scale")
async def scale_deployment(
    deployment_id: UUID,
    replicas: int = Query(..., ge=0, le=10),
    db: Session = Depends(get_db),
):
    """Scale a deployment."""
    deployment = db.query(Deployment).filter(
        Deployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # Update replicas
    old_replicas = deployment.replicas
    deployment.replicas = replicas
    
    if replicas == 0:
        deployment.status = "inactive"
    elif deployment.status == "inactive":
        deployment.status = "active"
    
    db.commit()
    
    # TODO: Scale on GKE
    
    logger.info(f"Deployment {deployment_id} scaled from {old_replicas} to {replicas}")
    return {
        "message": f"Deployment scaled to {replicas} replicas",
        "deployment_id": str(deployment_id),
        "replicas": replicas,
    }


@router.post("/{deployment_id}/restart")
async def restart_deployment(
    deployment_id: UUID,
    db: Session = Depends(get_db),
):
    """Restart a deployment."""
    deployment = db.query(Deployment).filter(
        Deployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # TODO: Restart on GKE
    
    logger.info(f"Deployment restarted: {deployment_id}")
    return {"message": "Deployment restart initiated"}


@router.get("/{deployment_id}/status")
async def get_deployment_status(
    deployment_id: UUID,
    db: Session = Depends(get_db),
):
    """Get detailed deployment status."""
    deployment = db.query(Deployment).filter(
        Deployment.id == deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # TODO: Get actual status from GKE
    
    return {
        "deployment_id": str(deployment.id),
        "status": deployment.status,
        "replicas": deployment.replicas,
        "endpoint_url": deployment.endpoint_url,
        "vllm_config": deployment.vllm_config,
    }
