# LLMForge Terraform Outputs

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.llmforge.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.llmforge.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.llmforge.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "models_bucket" {
  description = "GCS bucket for models"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "GCS bucket URL for models"
  value       = "gs://${google_storage_bucket.models.name}"
}

output "gke_service_account" {
  description = "Service account for GKE nodes"
  value       = google_service_account.gke_nodes.email
}

output "connect_command" {
  description = "Command to connect to the cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.llmforge.name} --region ${var.region} --project ${var.project_id}"
}
