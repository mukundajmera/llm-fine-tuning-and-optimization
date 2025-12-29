# LLMForge Terraform Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "llmforge-cluster"
}

variable "gpu_type" {
  description = "Type of GPU to use (nvidia-l4 or nvidia-tesla-a100)"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_machine_type" {
  description = "Machine type for GPU nodes"
  type        = string
  default     = "g2-standard-8"  # For L4 GPUs
  # Use "a2-highgpu-1g" for A100 GPUs
}

variable "max_gpu_nodes" {
  description = "Maximum number of GPU nodes in autoscaling"
  type        = number
  default     = 5
}

variable "enable_cloud_sql" {
  description = "Whether to create Cloud SQL instance"
  type        = bool
  default     = false
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}
