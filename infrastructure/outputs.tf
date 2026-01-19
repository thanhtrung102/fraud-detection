# Terraform Outputs for Fraud Detection Infrastructure

output "mlflow_artifacts_bucket" {
  description = "GCS bucket for MLflow artifacts"
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "data_bucket" {
  description = "GCS bucket for training data"
  value       = google_storage_bucket.data.name
}

output "database_instance" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.mlflow.name
}

output "database_connection" {
  description = "Cloud SQL connection string"
  value       = "postgresql://mlflow:***@${google_sql_database_instance.mlflow.public_ip_address}:5432/mlflow"
  sensitive   = true
}

output "database_ip" {
  description = "Cloud SQL public IP"
  value       = google_sql_database_instance.mlflow.public_ip_address
}

output "artifact_registry" {
  description = "Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.fraud_detection.repository_id}"
}

output "api_url" {
  description = "Cloud Run API URL"
  value       = google_cloud_run_v2_service.fraud_api.uri
}

output "bigquery_dataset" {
  description = "BigQuery dataset for monitoring"
  value       = google_bigquery_dataset.monitoring.dataset_id
}

output "service_account_email" {
  description = "Service account email for API"
  value       = google_service_account.fraud_api.email
}

# Useful commands
output "docker_push_command" {
  description = "Command to push Docker image"
  value       = "docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.fraud_detection.repository_id}/api:${var.docker_image_tag}"
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI"
  value       = "postgresql://mlflow:***@${google_sql_database_instance.mlflow.public_ip_address}:5432/mlflow"
  sensitive   = true
}
