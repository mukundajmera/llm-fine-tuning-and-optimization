"""
LLMForge Backend - Database Configuration
SQLAlchemy setup and models.
"""

import os
from uuid import uuid4

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    DateTime,
    Numeric,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
)
from sqlalchemy.sql import func

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://llmforge:password@localhost:5432/llmforge"
)

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# =============================================================================
# Database Models
# =============================================================================

class TrainingJob(Base):
    """Training job model."""
    __tablename__ = "training_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(Text, nullable=False)
    job_name = Column(Text, nullable=False)
    base_model = Column(Text, nullable=False)
    dataset_path = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="queued")
    hyperparameters = Column(JSONB, nullable=False)
    metrics = Column(JSONB)
    output_model_path = Column(Text)
    total_cost = Column(Numeric(10, 2))
    duration_seconds = Column(Integer)
    gpu_type = Column(Text)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    # Relationships
    deployments = relationship("Deployment", back_populates="training_job")
    evaluations = relationship("Evaluation", back_populates="training_job")
    
    __table_args__ = (
        Index("idx_training_jobs_user_id", "user_id"),
        Index("idx_training_jobs_status", "status"),
    )


class Deployment(Base):
    """Deployment model."""
    __tablename__ = "deployments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    training_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False
    )
    deployment_name = Column(Text, nullable=False, unique=True)
    model_path = Column(Text, nullable=False)
    vllm_config = Column(JSONB, nullable=False)
    endpoint_url = Column(Text)
    status = Column(Text, nullable=False, default="deploying")
    replicas = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="deployments")
    inference_logs = relationship("InferenceLog", back_populates="deployment")
    
    __table_args__ = (
        Index("idx_deployments_status", "status"),
    )


class InferenceLog(Base):
    """Inference log model."""
    __tablename__ = "inference_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    deployment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False
    )
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    cost_usd = Column(Numeric(8, 4))
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    deployment = relationship("Deployment", back_populates="inference_logs")
    
    __table_args__ = (
        Index("idx_inference_logs_deployment_id", "deployment_id"),
        Index("idx_inference_logs_created_at", "created_at"),
    )


class Evaluation(Base):
    """Evaluation model."""
    __tablename__ = "evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    training_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False
    )
    eval_dataset_path = Column(Text, nullable=False)
    metrics = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="evaluations")


# =============================================================================
# Database Utilities
# =============================================================================

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)


# =============================================================================
# SQL Schema (for reference)
# =============================================================================

SCHEMA_SQL = """
-- PostgreSQL schema for job tracking

CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    job_name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    status TEXT NOT NULL,
    hyperparameters JSONB NOT NULL,
    metrics JSONB,
    output_model_path TEXT,
    total_cost DECIMAL(10,2),
    duration_seconds INTEGER,
    gpu_type TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE TABLE deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
    deployment_name TEXT NOT NULL UNIQUE,
    model_path TEXT NOT NULL,
    vllm_config JSONB NOT NULL,
    endpoint_url TEXT,
    status TEXT NOT NULL,
    replicas INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE inference_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID REFERENCES deployments(id) ON DELETE CASCADE,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost_usd DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
    eval_dataset_path TEXT NOT NULL,
    metrics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_inference_logs_deployment_id ON inference_logs(deployment_id);
CREATE INDEX idx_inference_logs_created_at ON inference_logs(created_at);
"""
