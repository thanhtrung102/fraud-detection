# Terraform Variables for Fraud Detection Infrastructure

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# Database Configuration
variable "db_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "db_password" {
  description = "Database password for MLflow"
  type        = string
  sensitive   = true
}

variable "authorized_networks" {
  description = "Authorized networks for Cloud SQL"
  type = list(object({
    name = string
    cidr = string
  }))
  default = []
}

# API Configuration
variable "api_cpu" {
  description = "CPU limit for API container"
  type        = string
  default     = "1"
}

variable "api_memory" {
  description = "Memory limit for API container"
  type        = string
  default     = "2Gi"
}

variable "api_min_instances" {
  description = "Minimum number of API instances"
  type        = number
  default     = 0
}

variable "api_max_instances" {
  description = "Maximum number of API instances"
  type        = number
  default     = 10
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated access to API"
  type        = bool
  default     = false
}

# Docker Configuration
variable "docker_image_tag" {
  description = "Docker image tag for API"
  type        = string
  default     = "latest"
}
