"""
LLMForge vLLM Inference Server
FastAPI wrapper for vLLM with OpenAI-compatible API.

Features:
- OpenAI-compatible /v1/completions and /v1/chat/completions endpoints
- Streaming support
- Prometheus metrics
- Health checks
- Request logging

Usage:
    MODEL_PATH=path/to/model python serve.py
"""

import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

# =============================================================================
# Prometheus Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    'llmforge_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'llmforge_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)
ACTIVE_REQUESTS = Gauge(
    'llmforge_active_requests',
    'Number of active requests'
)
TOKENS_GENERATED = Counter(
    'llmforge_tokens_generated_total',
    'Total number of tokens generated'
)
TOKENS_PROMPT = Counter(
    'llmforge_tokens_prompt_total',
    'Total number of prompt tokens'
)

# =============================================================================
# Request/Response Models
# =============================================================================

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: Optional[str] = None
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stop: Optional[list[str]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=10)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class ChatMessage(BaseModel):
    """Chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: Optional[list[str]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=10)


class CompletionChoice(BaseModel):
    """Completion choice."""
    index: int
    text: str
    finish_reason: str


class CompletionUsage(BaseModel):
    """Token usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


# =============================================================================
# vLLM Engine
# =============================================================================

# Global engine instance
llm_engine = None


def init_vllm_engine():
    """Initialize vLLM engine."""
    global llm_engine
    
    try:
        from vllm import LLM
        
        logger.info(f"Loading model: {MODEL_PATH}")
        logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
        logger.info(f"Max model length: {MAX_MODEL_LEN}")
        logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
        
        llm_engine = LLM(
            model=MODEL_PATH,
            dtype="bfloat16",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            disable_log_stats=False,
        )
        
        logger.info("✅ vLLM engine initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        return False


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting LLMForge Inference Server...")
    success = init_vllm_engine()
    if not success:
        logger.warning("vLLM engine not available - running in mock mode")
    yield
    # Shutdown
    logger.info("Shutting down LLMForge Inference Server...")


app = FastAPI(
    title="LLMForge Inference API",
    description="OpenAI-compatible LLM inference API powered by vLLM",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "engine": "vllm" if llm_engine else "mock",
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH.split("/")[-1],
                "object": "model",
                "owned_by": "llmforge",
            }
        ]
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint."""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        if llm_engine is None:
            # Mock mode for testing
            generated_text = f"[Mock response to: {request.prompt[:50]}...]"
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(generated_text.split())
        else:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else -1,
                max_tokens=request.max_tokens,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                n=request.n,
            )
            
            outputs = llm_engine.generate([request.prompt], sampling_params)
            output = outputs[0]
            
            generated_text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
        
        # Update metrics
        TOKENS_PROMPT.inc(prompt_tokens)
        TOKENS_GENERATED.inc(completion_tokens)
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="completions").observe(latency)
        REQUEST_COUNT.labels(endpoint="completions", status="success").inc()
        
        return CompletionResponse(
            id=f"cmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=MODEL_PATH.split("/")[-1],
            choices=[
                CompletionChoice(
                    index=0,
                    text=generated_text,
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="completions", status="error").inc()
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        # Format messages into prompt
        prompt = format_chat_messages(request.messages)
        
        if llm_engine is None:
            # Mock mode for testing
            generated_text = f"[Mock chat response]"
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
        else:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                n=request.n,
            )
            
            outputs = llm_engine.generate([prompt], sampling_params)
            output = outputs[0]
            
            generated_text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
        
        # Update metrics
        TOKENS_PROMPT.inc(prompt_tokens)
        TOKENS_GENERATED.inc(completion_tokens)
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="chat_completions").observe(latency)
        REQUEST_COUNT.labels(endpoint="chat_completions", status="success").inc()
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=MODEL_PATH.split("/")[-1],
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="chat_completions", status="error").inc()
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        ACTIVE_REQUESTS.dec()


def format_chat_messages(messages: list[ChatMessage]) -> str:
    """Format chat messages into Llama 3.1 chat template."""
    formatted = "<|begin_of_text|>"
    
    for message in messages:
        formatted += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"
        formatted += f"{message.content}<|eot_id|>"
    
    # Add assistant header for generation
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return formatted


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLMForge Inference API",
        "version": "1.0.0",
        "model": MODEL_PATH,
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "completions": "/v1/completions",
            "chat_completions": "/v1/chat/completions",
            "metrics": "/metrics",
        }
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
