"""
LLMForge Backend - MLflow Client Service
Interface with MLflow for experiment tracking.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_mlflow_client():
    """Get MLflow client."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed")
        return None


def get_experiment_runs(experiment_name: str, max_results: int = 100) -> list:
    """
    Get runs from an MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        max_results: Maximum number of runs to return
    
    Returns:
        List of run dictionaries
    """
    mlflow = get_mlflow_client()
    if not mlflow:
        return []
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
        )
        
        return runs.to_dict("records")
        
    except Exception as e:
        logger.error(f"Failed to get experiment runs: {e}")
        return []


def get_run_metrics(run_id: str) -> dict:
    """
    Get metrics for an MLflow run.
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        Dictionary of metrics
    """
    mlflow = get_mlflow_client()
    if not mlflow:
        return {}
    
    try:
        run = mlflow.get_run(run_id)
        return run.data.metrics
        
    except Exception as e:
        logger.error(f"Failed to get run metrics: {e}")
        return {}


def get_run_artifacts(run_id: str) -> list:
    """
    Get artifacts for an MLflow run.
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        List of artifact paths
    """
    mlflow = get_mlflow_client()
    if not mlflow:
        return []
    
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        return [a.path for a in artifacts]
        
    except Exception as e:
        logger.error(f"Failed to get run artifacts: {e}")
        return []


def log_evaluation_metrics(
    run_id: str,
    metrics: dict,
) -> bool:
    """
    Log evaluation metrics to an existing run.
    
    Args:
        run_id: MLflow run ID
        metrics: Dictionary of metrics to log
    
    Returns:
        True if successful
    """
    mlflow = get_mlflow_client()
    if not mlflow:
        return False
    
    try:
        with mlflow.start_run(run_id=run_id):
            for key, value in metrics.items():
                mlflow.log_metric(f"eval_{key}", value)
        return True
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        return False


def create_experiment(experiment_name: str) -> Optional[str]:
    """
    Create an MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        Experiment ID or None if failed
    """
    mlflow = get_mlflow_client()
    if not mlflow:
        return None
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        
        return mlflow.create_experiment(experiment_name)
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        return None
