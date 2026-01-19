# Fraud Detection MLOps Infrastructure
# Google Cloud Platform (GCP)

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Backend configuration for state storage
  # Uncomment and configure for production
  # backend "gcs" {
  #   bucket = "fraud-detection-tf-state"
  #   prefix = "terraform/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "cloudrun.googleapis.com",
    "cloudsql.googleapis.com",
    "storage.googleapis.com",
    "bigquery.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "compute.googleapis.com"
  ])

  service            = each.key
  disable_on_destroy = false
}

# Cloud Storage Bucket for MLflow artifacts and data
resource "google_storage_bucket" "mlflow_artifacts" {
  name          = "${var.project_id}-mlflow-artifacts"
  location      = var.region
  force_destroy = var.environment == "dev"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    project     = "fraud-detection"
    component   = "mlflow"
  }

  depends_on = [google_project_service.apis]
}

# Cloud Storage Bucket for training data
resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-fraud-data"
  location      = var.region
  force_destroy = var.environment == "dev"

  uniform_bucket_level_access = true

  labels = {
    environment = var.environment
    project     = "fraud-detection"
    component   = "data"
  }

  depends_on = [google_project_service.apis]
}

# Cloud SQL PostgreSQL for MLflow tracking
resource "google_sql_database_instance" "mlflow" {
  name             = "fraud-mlflow-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = var.db_tier
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"

    backup_configuration {
      enabled            = true
      start_time         = "03:00"
      binary_log_enabled = false

      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled = true

      dynamic "authorized_networks" {
        for_each = var.authorized_networks
        content {
          name  = authorized_networks.value.name
          value = authorized_networks.value.cidr
        }
      }
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  deletion_protection = var.environment == "prod"

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
}

resource "google_sql_user" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
  password = var.db_password
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "fraud_detection" {
  location      = var.region
  repository_id = "fraud-detection"
  description   = "Docker images for fraud detection service"
  format        = "DOCKER"

  labels = {
    environment = var.environment
    project     = "fraud-detection"
  }

  depends_on = [google_project_service.apis]
}

# Secret Manager for sensitive configuration
resource "google_secret_manager_secret" "db_password" {
  secret_id = "mlflow-db-password"

  replication {
    auto {}
  }

  labels = {
    environment = var.environment
    project     = "fraud-detection"
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# Cloud Run Service for API
resource "google_cloud_run_v2_service" "fraud_api" {
  name     = "fraud-detection-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/fraud-detection/api:latest"

      ports {
        container_port = 8000
      }

      env {
        name  = "MODEL_DIR"
        value = "/app/models"
      }

      env {
        name  = "MLFLOW_TRACKING_URI"
        value = "postgresql://mlflow:${var.db_password}@${google_sql_database_instance.mlflow.public_ip_address}:5432/mlflow"
      }

      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.mlflow_artifacts.name
      }

      resources {
        limits = {
          cpu    = var.api_cpu
          memory = var.api_memory
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 3
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds = 30
      }
    }

    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }

    service_account = google_service_account.fraud_api.email
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.fraud_detection
  ]
}

# Service Account for Cloud Run
resource "google_service_account" "fraud_api" {
  account_id   = "fraud-detection-api"
  display_name = "Fraud Detection API Service Account"
}

# IAM bindings for service account
resource "google_storage_bucket_iam_member" "api_storage" {
  bucket = google_storage_bucket.mlflow_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.fraud_api.email}"
}

resource "google_secret_manager_secret_iam_member" "api_secrets" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.fraud_api.email}"
}

# Allow unauthenticated access to API (for demo)
resource "google_cloud_run_v2_service_iam_member" "allow_unauthenticated" {
  count    = var.allow_unauthenticated ? 1 : 0
  location = google_cloud_run_v2_service.fraud_api.location
  name     = google_cloud_run_v2_service.fraud_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# BigQuery Dataset for monitoring data
resource "google_bigquery_dataset" "monitoring" {
  dataset_id  = "fraud_monitoring"
  location    = var.region
  description = "Fraud detection monitoring data"

  labels = {
    environment = var.environment
    project     = "fraud-detection"
  }

  depends_on = [google_project_service.apis]
}

# BigQuery Table for predictions
resource "google_bigquery_table" "predictions" {
  dataset_id = google_bigquery_dataset.monitoring.dataset_id
  table_id   = "predictions"

  schema = <<EOF
[
  {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
  {"name": "transaction_id", "type": "STRING", "mode": "NULLABLE"},
  {"name": "fraud_probability", "type": "FLOAT", "mode": "REQUIRED"},
  {"name": "is_fraud", "type": "BOOLEAN", "mode": "REQUIRED"},
  {"name": "model_version", "type": "STRING", "mode": "NULLABLE"},
  {"name": "latency_ms", "type": "FLOAT", "mode": "NULLABLE"}
]
EOF

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
}

# BigQuery Table for drift metrics
resource "google_bigquery_table" "drift_metrics" {
  dataset_id = google_bigquery_dataset.monitoring.dataset_id
  table_id   = "drift_metrics"

  schema = <<EOF
[
  {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
  {"name": "dataset_drift", "type": "BOOLEAN", "mode": "REQUIRED"},
  {"name": "drift_share", "type": "FLOAT", "mode": "REQUIRED"},
  {"name": "drifted_features", "type": "STRING", "mode": "REPEATED"},
  {"name": "alert_triggered", "type": "BOOLEAN", "mode": "REQUIRED"}
]
EOF

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
}
