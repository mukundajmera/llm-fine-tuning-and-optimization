# LLMForge GKE Infrastructure
# Terraform configuration for GKE cluster with GPU node pools

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "llmforge-terraform-state"
    prefix = "gke"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# GKE Cluster
resource "google_container_cluster" "llmforge" {
  provider = google-beta
  
  name     = var.cluster_name
  location = var.region
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  networking_mode = "VPC_NATIVE"
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  network    = google_compute_network.llmforge.name
  subnetwork = google_compute_subnetwork.llmforge.name
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  release_channel {
    channel = "REGULAR"
  }
  
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }
  
  cluster_autoscaling {
    enabled = true
    
    resource_limits {
      resource_type = "cpu"
      minimum       = 0
      maximum       = 100
    }
    
    resource_limits {
      resource_type = "memory"
      minimum       = 0
      maximum       = 500
    }
    
    resource_limits {
      resource_type = "nvidia-l4"
      minimum       = 0
      maximum       = 10
    }
  }
}

# CPU Node Pool (for backend services)
resource "google_container_node_pool" "cpu_pool" {
  name       = "cpu-pool"
  location   = var.region
  cluster    = google_container_cluster.llmforge.name
  
  initial_node_count = 2
  
  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }
  
  node_config {
    machine_type = "e2-standard-4"
    
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      "llmforge/node-type" = "cpu"
    }
    
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU Node Pool (for vLLM inference)
resource "google_container_node_pool" "gpu_pool" {
  provider = google-beta
  
  name       = "gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.llmforge.name
  
  initial_node_count = 0
  
  autoscaling {
    min_node_count = 0
    max_node_count = var.max_gpu_nodes
  }
  
  node_config {
    machine_type = var.gpu_machine_type
    
    guest_accelerator {
      type  = var.gpu_type
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }
    
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      "llmforge/node-type" = "gpu"
    }
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
    
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# VPC Network
resource "google_compute_network" "llmforge" {
  name                    = "llmforge-network"
  auto_create_subnetworks = false
}

# Subnet
resource "google_compute_subnetwork" "llmforge" {
  name          = "llmforge-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.llmforge.id
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }
  
  private_ip_google_access = true
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "llmforge-gke-nodes"
  display_name = "LLMForge GKE Nodes Service Account"
}

# IAM roles for GKE nodes
resource "google_project_iam_member" "gke_nodes_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Cloud Storage bucket for models
resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-llmforge-models"
  location      = var.region
  storage_class = "STANDARD"
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud SQL for PostgreSQL (optional - for production)
resource "google_sql_database_instance" "llmforge" {
  count = var.enable_cloud_sql ? 1 : 0
  
  name             = "llmforge-db"
  database_version = "POSTGRES_16"
  region           = var.region
  
  settings {
    tier = "db-custom-2-4096"
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.llmforge.id
    }
    
    backup_configuration {
      enabled = true
    }
  }
  
  deletion_protection = true
}
