# MLOps Guide for Fraud Detection

This guide covers the MLOps components of the fraud detection project, including experiment tracking, workflow orchestration, model deployment, monitoring, and CI/CD.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Prefect Workflow Orchestration](#prefect-workflow-orchestration)
- [Model Deployment](#model-deployment)
- [Model Monitoring](#model-monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Infrastructure as Code](#infrastructure-as-code)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │  GitHub     │      │   Prefect   │      │   MLflow    │                │
│   │  Actions    │─────▶│   Server    │─────▶│   Server    │                │
│   │  (CI/CD)    │      │(Orchestrate)│      │  (Track)    │                │
│   └─────────────┘      └──────┬──────┘      └──────┬──────┘                │
│                               │                    │                        │
│                               ▼                    ▼                        │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │                    Pipelines                             │              │
│   │  ┌───────────┐  ┌───────────┐  ┌───────────────┐        │              │
│   │  │ Training  │  │ Inference │  │  Monitoring   │        │              │
│   │  │  Pipeline │  │  Pipeline │  │   Pipeline    │        │              │
│   │  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘        │              │
│   └────────┼──────────────┼────────────────┼────────────────┘              │
│            │              │                │                                │
│            ▼              ▼                ▼                                │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│   │   Model     │  │   FastAPI   │  │  Evidently  │                        │
│   │  Registry   │──│   Server    │  │   Reports   │                        │
│   └─────────────┘  └──────┬──────┘  └─────────────┘                        │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │                    Deployment                            │              │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │              │
│   │  │   Docker    │  │  Artifact   │  │  Cloud Run  │      │              │
│   │  │   Image     │─▶│  Registry   │─▶│   (GCP)     │      │              │
│   │  └─────────────┘  └─────────────┘  └─────────────┘      │              │
│   └─────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MLflow Experiment Tracking

### Setup

```python
from mlops.tracking import setup_mlflow, start_run, log_metrics, log_params, log_artifacts

# Initialize MLflow
setup_mlflow(
    tracking_uri="sqlite:///mlflow.db",  # Local SQLite
    experiment_name="fraud-detection"
)
```

### Usage in Training

```python
with start_run(run_name="training_experiment_v1") as run:
    # Log hyperparameters
    log_params({
        "model": "stacking_ensemble",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "use_shap_selection": True,
        "n_features": 30
    })

    # Train model...
    model.fit(X_train, y_train)

    # Log metrics
    log_metrics({
        "auc_roc": 0.98,
        "accuracy": 0.98,
        "precision": 0.95,
        "recall": 0.85,
        "f1_score": 0.90
    })

    # Log artifacts (plots, model files)
    log_artifacts("results/", "evaluation_plots")
    log_artifacts("models/", "model_files")

    print(f"Run ID: {run.info.run_id}")
```

### View MLflow UI

```bash
# Start MLflow server
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Open browser: http://localhost:5000
```

### Model Registry

```python
from mlops.registry import ModelRegistry

registry = ModelRegistry(model_name="fraud-detection-stacking")

# Register model from a run
version = registry.register_model(
    run_id="abc123def456",
    model_path="model",
    description="AUC-ROC: 0.98, Accuracy: 0.98"
)

# Transition to production
registry.transition_model_stage(version, "Production")

# Load production model
model = registry.load_model(stage="Production")
```

---

## Prefect Workflow Orchestration

### Training Pipeline

```python
from pipelines.training_pipeline import training_flow

# Run training with all options
result = training_flow(
    config_path="config/config.yaml",
    use_optuna=True,           # Hyperparameter tuning
    use_feature_selection=True, # SHAP feature selection
    n_trials=20,               # Optuna trials
    n_top_features=30,         # Features to select
    register_model=True,       # Register to MLflow
    mlflow_tracking_uri="sqlite:///mlflow.db"
)

print(f"Run ID: {result['run_id']}")
print(f"AUC-ROC: {result['metrics']['auc_roc']:.4f}")
print(f"Threshold: {result['threshold']:.2f}")
```

### Inference Pipeline

```python
from pipelines.inference_pipeline import inference_flow

result = inference_flow(
    data_path="data/new_transactions.csv",
    model_dir="models",
    output_path="results/predictions.csv",
    threshold=0.44
)

print(f"Processed: {result['total_processed']} transactions")
print(f"Fraud rate: {result['report']['fraud_rate']:.2%}")
print(f"High risk: {result['report']['high_risk_count']}")
```

### Monitoring Pipeline

```python
from pipelines.monitoring_pipeline import monitoring_flow

result = monitoring_flow(
    reference_path="data/reference_data.csv",
    production_path="data/production_data.csv",
    output_dir="monitoring/reports",
    drift_threshold=0.3,
    alert_config={
        "slack_webhook": "https://hooks.slack.com/...",
        "email": "alerts@company.com"
    }
)

print(f"Drift detected: {result['drift_detected']}")
print(f"Drift share: {result['drift_share']:.2%}")
print(f"Report: {result['drift_report']}")
```

### Prefect UI

```bash
# Start Prefect server
prefect server start

# View UI: http://localhost:4200
```

### Scheduled Deployments

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from pipelines.monitoring_pipeline import monitoring_flow

# Create daily monitoring deployment
deployment = Deployment.build_from_flow(
    flow=monitoring_flow,
    name="daily-drift-monitoring",
    schedule=CronSchedule(cron="0 6 * * *"),  # Daily at 6 AM
    parameters={
        "reference_path": "data/reference.csv",
        "production_path": "data/production.csv",
        "drift_threshold": 0.3
    }
)
deployment.apply()
```

---

## Model Deployment

### FastAPI Server

```bash
# Development
uvicorn deployment.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn deployment.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Single transaction prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/model/reload` | POST | Reload model from disk |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "TransactionAmt": 150.0,
      "card1": 12345,
      "C14": 1.0,
      "C12": 0.5,
      "V308": 0.3
    },
    "threshold": 0.5
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionAmt": 100.0, "card1": 1234},
      {"TransactionAmt": 500.0, "card1": 5678},
      {"TransactionAmt": 50.0, "card1": 9012}
    ],
    "threshold": 0.44
  }'

# Model info
curl http://localhost:8000/model/info
```

### Response Format

```json
{
  "is_fraud": false,
  "fraud_probability": 0.23,
  "risk_level": "low",
  "threshold_used": 0.5,
  "timestamp": "2026-01-20T10:30:00"
}
```

### Docker Deployment

```bash
# Build image
docker build -f deployment/Dockerfile -t fraud-detection-api .

# Run container
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_DIR=/app/models \
  fraud-detection-api

# Check logs
docker logs -f fraud-api

# Stop
docker stop fraud-api && docker rm fraud-api
```

### Docker Compose (Full Stack)

```yaml
# deployment/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
    environment:
      - MODEL_DIR=/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: python:3.11-slim
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    volumes:
      - ../mlflow.db:/mlflow.db
      - ../mlruns:/mlruns
```

```bash
# Start stack
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop
docker-compose -f deployment/docker-compose.yml down
```

---

## Model Monitoring

### Evidently Integration

```python
from mlops.monitoring import FraudMonitor, create_monitoring_report

# Initialize monitor with reference data
monitor = FraudMonitor(
    reference_data=reference_df,
    target_column="isFraud",
    prediction_column="prediction"
)

# Check for drift
drift_result = monitor.check_drift_threshold(
    current_data=production_df,
    drift_threshold=0.5,
    feature_drift_threshold=0.3
)

if drift_result["alert"]:
    print(f"ALERT: Data drift detected!")
    print(f"Drift share: {drift_result['drift_share']:.2%}")
    print(f"Drifted features: {drift_result['drifted_features']}")
```

### Generate Reports

```python
# Data drift report
report, metrics = monitor.generate_data_drift_report(
    current_data=production_df,
    output_path="monitoring/drift_report.html"
)

# Model performance report
report, metrics = monitor.generate_model_performance_report(
    current_data=production_df,
    output_path="monitoring/performance_report.html"
)

# Data quality report
report, metrics = monitor.generate_data_quality_report(
    current_data=production_df,
    output_path="monitoring/quality_report.html"
)
```

### Automated Monitoring

```python
from pipelines.monitoring_pipeline import monitoring_flow

# Run monitoring pipeline
result = monitoring_flow(
    reference_path="data/reference.csv",
    production_path="data/production.csv",
    drift_threshold=0.3
)

# Results include:
# - drift_detected: bool
# - drift_share: float
# - drifted_features: list
# - drift_report: path to HTML report
# - metrics_path: path to JSON metrics
```

---

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

**Triggered on:** Push/PR to `main` and `mlops` branches

| Job | Description |
|-----|-------------|
| **quality** | Ruff linting, Black formatting, MyPy type checking |
| **test** | Unit tests with pytest, coverage report |
| **integration** | API integration tests |
| **docker** | Validate Docker builds |
| **security** | Bandit security scan |

### Continuous Deployment (`.github/workflows/cd.yml`)

**Triggered on:** Push to `main`, version tags (`v*`)

| Job | Description |
|-----|-------------|
| **build** | Build and push Docker image to Artifact Registry |
| **deploy-staging** | Deploy to Cloud Run staging |
| **deploy-production** | Deploy to Cloud Run production (on tags) |

### Required Secrets

| Secret | Description |
|--------|-------------|
| `GCP_SA_KEY` | GCP service account JSON key |
| `GCP_PROJECT_ID` | GCP project ID |

### Manual Deployment

```bash
# Build and push manually
docker build -f deployment/Dockerfile -t us-central1-docker.pkg.dev/PROJECT/fraud-detection/api:latest .
docker push us-central1-docker.pkg.dev/PROJECT/fraud-detection/api:latest

# Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
  --image us-central1-docker.pkg.dev/PROJECT/fraud-detection/api:latest \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated
```

---

## Infrastructure as Code

### Terraform Setup

```bash
cd infrastructure

# Initialize
terraform init

# Plan
terraform plan \
  -var="project_id=your-gcp-project" \
  -var="region=us-central1"

# Apply
terraform apply \
  -var="project_id=your-gcp-project" \
  -var="region=us-central1"

# Destroy
terraform destroy \
  -var="project_id=your-gcp-project"
```

### Resources Created

| Resource | Purpose |
|----------|---------|
| **Cloud Storage** | MLflow artifacts, model files |
| **Cloud SQL (PostgreSQL)** | MLflow tracking backend |
| **Cloud Run** | API deployment (autoscaling) |
| **Artifact Registry** | Docker images |
| **Secret Manager** | API keys, credentials |
| **BigQuery** | Monitoring data warehouse |

### Variables

```hcl
# infrastructure/variables.tf
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (staging/production)"
  type        = string
  default     = "staging"
}
```

---

## Local Development

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/thanhtrung102/fraud-detection.git
cd fraud-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run training
python pipelines/training_pipeline.py

# 3. Start API
uvicorn deployment.api.main:app --reload

# 4. Test API
curl http://localhost:8000/health
```

### Development Workflow

```bash
# Run tests
pytest tests/ -v

# Check code quality
ruff check src/ mlops/ pipelines/
black --check src/ mlops/ pipelines/
mypy src/ mlops/ pipelines/ --ignore-missing-imports

# Format code
black src/ mlops/ pipelines/ tests/ deployment/

# Run single pipeline
python -c "from pipelines.training_pipeline import training_flow; training_flow(use_optuna=False)"
```

### Using Makefile

```bash
make install      # Install dependencies
make test         # Run all tests
make lint         # Run linters
make format       # Format code
make train        # Run training
make serve        # Start API server
make docker-build # Build Docker image
make docker-run   # Run Docker container
make clean        # Clean artifacts
```

---

## Production Deployment

### GCP Deployment Checklist

1. **Setup GCP Project**
   ```bash
   gcloud projects create fraud-detection-prod
   gcloud config set project fraud-detection-prod
   ```

2. **Enable APIs**
   ```bash
   gcloud services enable \
     run.googleapis.com \
     artifactregistry.googleapis.com \
     cloudbuild.googleapis.com \
     secretmanager.googleapis.com
   ```

3. **Create Service Account**
   ```bash
   gcloud iam service-accounts create fraud-detection-sa \
     --display-name="Fraud Detection Service Account"
   ```

4. **Deploy Infrastructure**
   ```bash
   cd infrastructure
   terraform apply -var="project_id=fraud-detection-prod"
   ```

5. **Configure GitHub Secrets**
   - Add `GCP_SA_KEY` (service account JSON)
   - Add `GCP_PROJECT_ID`

6. **Deploy via CI/CD**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

### Environment Configuration

| Environment | Config | Resources |
|-------------|--------|-----------|
| **Development** | Local SQLite, no auth | Minimal |
| **Staging** | Cloud SQL, basic auth | 2GB RAM, 1 CPU |
| **Production** | Cloud SQL, full auth | 4GB RAM, 2 CPU, autoscaling |

### Monitoring in Production

```python
# Schedule monitoring
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=monitoring_flow,
    name="production-monitoring",
    schedule=CronSchedule(cron="0 */6 * * *"),  # Every 6 hours
    parameters={
        "reference_path": "gs://bucket/reference.csv",
        "production_path": "gs://bucket/production.csv",
        "alert_config": {
            "slack_webhook": "https://hooks.slack.com/..."
        }
    }
)
deployment.apply()
```

---

## Troubleshooting

### Common Issues

**MLflow connection error:**
```bash
# Check MLflow server is running
curl http://localhost:5000/health

# Start if needed
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Model not loading:**
```bash
# Verify model files exist
ls -la models/

# Check permissions
chmod 755 models/*
```

**Docker build fails:**
```bash
# Build with verbose output
docker build -f deployment/Dockerfile -t test . --progress=plain --no-cache
```

**Prefect flow fails:**
```bash
# Check Prefect server
prefect server start

# View flow logs
prefect flow-run logs <flow-run-id>
```

**API returns 503:**
```python
# Model not loaded - check logs
docker logs fraud-api

# Manually reload model
curl -X POST http://localhost:8000/model/reload
```

### Logs and Debugging

```bash
# API logs
docker logs -f fraud-api

# Prefect logs
prefect flow-run logs <run-id>

# MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Monitoring reports
ls -la monitoring/evidently_reports/
```

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prefect Documentation](https://docs.prefect.io/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
