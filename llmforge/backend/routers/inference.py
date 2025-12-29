"""
LLMForge Backend - Inference Router
Proxy to vLLM inference server.
"""

import os
import time
import logging
from uuid import UUID
from decimal import Decimal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database import get_db, Deployment, InferenceLog
from models import (
    InferenceRequest,
    InferenceResponse,
    InferenceLogResponse,
)
from services.cost_calculator import calculate_inference_cost_per_token

logger = logging.getLogger(__name__)

router = APIRouter()

# HTTP client for vLLM
http_client = httpx.AsyncClient(timeout=60.0)


@router.post("/completions", response_model=InferenceResponse)
async def create_completion(
    request: InferenceRequest,
    db: Session = Depends(get_db),
):
    """Proxy completion request to vLLM."""
    # Get deployment
    deployment = db.query(Deployment).filter(
        Deployment.id == request.deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    if deployment.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not active. Status: {deployment.status}"
        )
    
    if not deployment.endpoint_url:
        raise HTTPException(
            status_code=400,
            detail="Deployment has no endpoint URL configured"
        )
    
    # Make request to vLLM
    start_time = time.time()
    
    try:
        response = await http_client.post(
            f"{deployment.endpoint_url}/v1/completions",
            json={
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream,
            },
        )
        response.raise_for_status()
        result = response.json()
        
    except httpx.HTTPError as e:
        logger.error(f"vLLM request failed: {e}")
        raise HTTPException(status_code=502, detail=f"vLLM request failed: {e}")
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Extract response data
    generated_text = result["choices"][0]["text"]
    prompt_tokens = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    
    # Calculate cost
    cost_usd = Decimal(str(
        calculate_inference_cost_per_token(prompt_tokens + completion_tokens)
    ))
    
    # Log inference
    inference_log = InferenceLog(
        deployment_id=deployment.id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )
    db.add(inference_log)
    db.commit()
    
    return InferenceResponse(
        id=result["id"],
        deployment_id=deployment.id,
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


@router.post("/chat")
async def create_chat_completion(
    request: InferenceRequest,
    db: Session = Depends(get_db),
):
    """Proxy chat completion request to vLLM."""
    # Get deployment
    deployment = db.query(Deployment).filter(
        Deployment.id == request.deployment_id
    ).first()
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    if deployment.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not active. Status: {deployment.status}"
        )
    
    # For demo/mock mode, return a mock response
    if not deployment.endpoint_url:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "deployment_id": str(deployment.id),
            "message": {
                "role": "assistant",
                "content": f"[Mock response] This is a mock chat response for: {request.prompt[:50]}...",
            },
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": 10,
            "latency_ms": 100,
            "cost_usd": 0.001,
        }
    
    # Make request to vLLM
    start_time = time.time()
    
    try:
        response = await http_client.post(
            f"{deployment.endpoint_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": request.prompt}
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        )
        response.raise_for_status()
        result = response.json()
        
    except httpx.HTTPError as e:
        logger.error(f"vLLM request failed: {e}")
        raise HTTPException(status_code=502, detail=f"vLLM request failed: {e}")
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "id": result["id"],
        "deployment_id": str(deployment.id),
        "message": result["choices"][0]["message"],
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": result["usage"]["completion_tokens"],
        "latency_ms": latency_ms,
    }


@router.get("/logs", response_model=list[InferenceLogResponse])
async def list_inference_logs(
    deployment_id: UUID = Query(None, description="Filter by deployment"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """List inference logs."""
    query = db.query(InferenceLog)
    
    if deployment_id:
        query = query.filter(InferenceLog.deployment_id == deployment_id)
    
    logs = query.order_by(desc(InferenceLog.created_at)).limit(limit).all()
    
    return logs


@router.get("/stats")
async def get_inference_stats(
    deployment_id: UUID = Query(None, description="Filter by deployment"),
    db: Session = Depends(get_db),
):
    """Get inference statistics."""
    from sqlalchemy import func
    
    query = db.query(
        func.count(InferenceLog.id).label("total_requests"),
        func.sum(InferenceLog.prompt_tokens).label("total_prompt_tokens"),
        func.sum(InferenceLog.completion_tokens).label("total_completion_tokens"),
        func.avg(InferenceLog.latency_ms).label("avg_latency_ms"),
        func.sum(InferenceLog.cost_usd).label("total_cost"),
    )
    
    if deployment_id:
        query = query.filter(InferenceLog.deployment_id == deployment_id)
    
    result = query.first()
    
    return {
        "total_requests": result.total_requests or 0,
        "total_prompt_tokens": int(result.total_prompt_tokens or 0),
        "total_completion_tokens": int(result.total_completion_tokens or 0),
        "avg_latency_ms": float(result.avg_latency_ms or 0),
        "total_cost_usd": float(result.total_cost or 0),
    }
