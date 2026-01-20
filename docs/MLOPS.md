# MLOps Guide for Fraud Detection

Complete guide for MLOps components including experiment tracking, workflow orchestration, model deployment, monitoring, and CI/CD.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Monitoring Pipeline](#monitoring-pipeline)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Infrastructure as Code](#infrastructure-as-code)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
+-----------------------------------------------------------------------------+
|                           MLOps Architecture                                 |
+-----------------------------------------------------------------------------+
|                                                                              |
|   +-------------+      +-------------+      +-------------+                  |
|   |  GitHub     |      |   Prefect   |      |   MLflow    |                  |
|   |  Actions    |----->|   Server    |----->|   Server    |                  |
|   |  (CI/CD)    |      |(Orchestrate)|      |  (Track)    |                  |
|   +-------------+      +------+------+      +------+------+                  |
|                               |                    |                         |
|                               v                    v                         |
|   +---------------------------------------------------------------+         |
|   |                        Pipelines                               |         |
|   |  +----------+  +----------+  +--------------+                  |         |
|   |  | Training |  | Inference|  |  Monitoring  |                  |         |
|   |  | Pipeline |  | Pipeline |  |   Pipeline   |                  |         |
|   |  +----+-----+  +----+-----+  +------+-------+                  |         |
|   +-------|-------------|---------------|---------------------------+         |
|           |             |               |                                    |
|           v             v               v                                    |
|   +-------------+  +-------------+  +-------------+                          |
|   |   Model     |  |   FastAPI   |  |  Evidently  |                          |
|   |  Registry   |--|   Server    |  |   Reports   |                          |
|   +-------------+  +------+------+  +-------------+                          |
|                           |                                                  |
|                           v                                                  |
|   +---------------------------------------------------------------+         |
|   |                      Deployment                                |         |
|   |  +-------------+  +-------------+  +-------------+             |         |
|   |  |   Docker    |  |  Artifact   |  |  Cloud Run  |             |         |
|   |  |   Image     |->|  Registry   |->|   (GCP)     |             |         |
|   |  +-------------+  +-------------+  +-------------+             |         |
|   +---------------------------------------------------------------+         |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Experiment Tracking | MLflow | Log metrics, parameters, artifacts |
| Workflow Orchestration | Prefect | Pipeline orchestration and scheduling |
| Model Serving | FastAPI | REST API for predictions |
| Monitoring | Evidently | Data drift and model performance |
| Containerization | Docker | Packaging and deployment |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Infrastructure | Terraform | GCP resource provisioning |
| Cloud Platform | GCP Cloud Run | Serverless deployment |

---

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- GCP account (optional, for cloud deployment)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
# Low memory mode (8GB RAM)
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# Production mode (16GB+ RAM)
python pipelines/training_pipeline.py --config-path config/params_production.yaml --use-optuna
```

### View MLflow Results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### Start API Server

```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

---

## Training Pipeline

### Command Reference

| Use Case | Command |
|----------|---------|
| Default training | `python pipelines/training_pipeline.py` |
| Low memory (8GB) | `python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml` |
| Production (16GB+) | `python pipelines/training_pipeline.py --config-path config/params_production.yaml` |
| With Optuna tuning | `python pipelines/training_pipeline.py --use-optuna` |
| Register to MLflow | `python pipelines/training_pipeline.py --register-model` |
| Skip SHAP selection | `python pipelines/training_pipeline.py --no-feature-selection` |

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config-path` | Path to configuration file | `config/params.yaml` |
| `--use-optuna` | Enable Optuna hyperparameter tuning | Disabled |
| `--no-feature-selection` | Skip SHAP feature selection | Enabled |
| `--register-model` | Register model to MLflow registry | Disabled |

### Configuration Profiles

| Profile | File | Sample Size | RAM | Use Case |
|---------|------|-------------|-----|----------|
| Default | `config/params.yaml` | 590,540 | 16GB+ | Full dataset |
| Production | `config/params_production.yaml` | 300,000 | 16GB | Paper methodology |
| Codespaces | `config/params_codespaces.yaml` | 100,000 | 8GB | Limited resources |

### Paper Methodology vs Default Pipeline

| Aspect | Paper (arXiv:2505.10050) | MLOps Default |
|--------|-------------------------|---------------|
| Optuna tuning | 20 trials per model | Disabled |
| SHAP selection | 30 features | Enabled |
| n_estimators | 400 | 200 (codespaces), 400 (production) |
| max_depth | 8 | 6 (codespaces), 8 (production) |
| Threshold | 0.44 | F1-optimized |

### Full Paper Reproduction

```bash
# Option 1: Research pipeline
python -m src.main

# Option 2: MLOps pipeline with production config
python pipelines/training_pipeline.py \
  --config-path config/params_production.yaml \
  --use-optuna \
  --register-model
```

### Expected Results

| Config | AUC-ROC | Accuracy | Time |
|--------|---------|----------|------|
| Codespaces (100K, no Optuna) | ~0.91 | ~97% | ~2 min |
| Codespaces (100K, with Optuna) | ~0.93 | ~98% | ~15 min |
| Production (300K, with Optuna) | ~0.97 | ~99% | ~45 min |

---

## Inference Pipeline

### Batch Inference

```bash
python pipelines/inference_pipeline.py \
  --data-path data/transactions.csv \
  --model-dir models \
  --output-path results/predictions.csv \
  --threshold 0.44 \
  --sample-size 1000
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path` | Input CSV file | `data/train_transaction.csv` |
| `--model-dir` | Model directory | `models` |
| `--output-path` | Output predictions | `results/predictions.csv` |
| `--threshold` | Classification threshold | `0.44` |
| `--sample-size` | Rows to process (0=all) | `1000` |

### Programmatic Usage

```python
from pipelines.inference_pipeline import inference_flow

result = inference_flow(
    data_path="data/transactions.csv",
    model_dir="models",
    output_path="results/predictions.csv",
    threshold=0.44
)

print(f"Processed: {result['total_processed']} transactions")
print(f"Fraud rate: {result['report']['fraud_rate']:.2%}")
```

---

## Monitoring Pipeline

### Generate Evidently Reports

```bash
python pipelines/monitoring_pipeline.py
```

### Programmatic Usage

```python
from mlops.monitoring import FraudMonitor

# Initialize with reference data
monitor = FraudMonitor(
    reference_data=reference_df,
    target_column="isFraud",
    prediction_column="prediction"
)

# Check for drift
drift_result = monitor.check_drift_threshold(
    current_data=production_df,
    drift_threshold=0.5
)

if drift_result["alert"]:
    print(f"ALERT: Data drift detected!")
    print(f"Drift share: {drift_result['drift_share']:.2%}")
```

### Report Types

| Report | Method | Output |
|--------|--------|--------|
| Data Drift | `generate_data_drift_report()` | HTML report |
| Model Performance | `generate_model_performance_report()` | HTML report |
| Data Quality | `generate_data_quality_report()` | HTML report |

---

## Local Development

### Complete Workflow

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Download dataset from Kaggle
# Place train_transaction.csv and train_identity.csv in data/

# 3. Run training
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# 4. View MLflow results
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000

# 5. Run inference
python pipelines/inference_pipeline.py

# 6. Generate monitoring reports
python pipelines/monitoring_pipeline.py

# 7. Start API server
uvicorn api.main:app --reload --port 8000

# 8. Test API
curl http://localhost:8000/health
```

### Docker Local Development

```bash
# Build image
docker build -f deployment/Dockerfile -t fraud-detection-api .

# Run container
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models fraud-detection-api

# Full stack with Docker Compose
docker-compose -f deployment/docker-compose.yml up -d
# API: http://localhost:8000
# MLflow: http://localhost:5000
```

### Code Quality Checks

```bash
# Run tests
pytest tests/ -v

# Linting
ruff check src/ mlops/ pipelines/

# Formatting
black --check src/ mlops/ pipelines/

# Type checking
mypy src/ mlops/ pipelines/ --ignore-missing-imports
```

---

## Production Deployment

### Deployment Options

| Option | Best For | Complexity | Auto-scaling |
|--------|----------|------------|--------------|
| Docker Compose | Self-hosted, small-scale | Low | Manual |
| GCP Cloud Run | Serverless production | Medium | Automatic |
| Kubernetes | Large-scale, multi-region | High | Automatic |

### Option 1: Docker Compose (Self-Hosted)

```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# Services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000

# Scale API replicas
docker-compose -f deployment/docker-compose.yml up -d --scale api=3

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker-compose.yml down
```

### Option 2: GCP Cloud Run

#### Prerequisites

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### Step-by-Step Deployment

```bash
# 1. Enable required APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# 2. Create Artifact Registry repository
gcloud artifacts repositories create fraud-detection \
  --repository-format=docker \
  --location=us-central1

# 3. Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# 4. Build and push image
docker build -f deployment/Dockerfile -t us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest

# 5. Upload model to Cloud Storage
gsutil mb gs://PROJECT_ID-models
gsutil cp -r models/* gs://PROJECT_ID-models/

# 6. Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

### Environment Configuration

| Environment | Config | RAM | CPU | MLflow Backend |
|-------------|--------|-----|-----|----------------|
| Development | `params_codespaces.yaml` | 8GB | Local | SQLite |
| Staging | `params_production.yaml` | 2GB | 1 | Cloud SQL |
| Production | `params_production.yaml` | 4GB | 2 | Cloud SQL |

### Production Checklist

- [ ] Configure GCP project and enable APIs
- [ ] Set up Artifact Registry for Docker images
- [ ] Deploy infrastructure with Terraform
- [ ] Configure GitHub secrets for CI/CD
- [ ] Upload trained model to Cloud Storage
- [ ] Deploy API to Cloud Run
- [ ] Set up monitoring and alerting
- [ ] Test end-to-end prediction flow

---

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

**Triggered on:** Push/PR to `main` and `mlops` branches

| Job | Description |
|-----|-------------|
| quality | Ruff linting, Black formatting |
| test | Unit tests with pytest |
| integration | API integration tests |
| docker | Validate Docker builds |
| security | Bandit security scan |

### Continuous Deployment (`.github/workflows/cd.yml`)

**Triggered on:** Push to `main`, version tags (`v*`)

| Job | Description |
|-----|-------------|
| build | Build and push Docker image |
| deploy-staging | Deploy to Cloud Run staging |
| deploy-production | Deploy to production (on tags) |

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_SA_KEY` | Service account JSON key (base64) |

### Deployment Triggers

| Trigger | Action |
|---------|--------|
| Push to `main` | Deploy to staging |
| Tag `v*` | Deploy to production |
| PR to `main` | Run CI checks only |

```bash
# Deploy to staging (automatic on merge)
git checkout main
git merge feature-branch
git push origin main

# Deploy to production (tag release)
git tag v1.0.0
git push origin v1.0.0
```

---

## Infrastructure as Code

### Terraform Setup

```bash
cd infrastructure

# Create terraform.tfvars
cat > terraform.tfvars << EOF
project_id  = "your-gcp-project-id"
region      = "us-central1"
environment = "production"
EOF

# Initialize
terraform init

# Plan
terraform plan

# Apply
terraform apply

# Destroy
terraform destroy
```

### Resources Created

| Resource | Purpose |
|----------|---------|
| Cloud Storage | MLflow artifacts, model files |
| Cloud SQL (PostgreSQL) | MLflow tracking backend |
| Cloud Run | API deployment (autoscaling) |
| Artifact Registry | Docker images |
| Secret Manager | API keys, credentials |

---

## API Reference

### Endpoints

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
      "C14": 1.0
    }
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionAmt": 100.0, "card1": 1234},
      {"TransactionAmt": 500.0, "card1": 5678}
    ]
  }'
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
# Expected: xgb_model.joblib, lgbm_model.joblib, catboost_model.cbm, meta_learner.joblib, feature_names.json

# Check permissions
chmod 755 models/*
```

**Memory error during training:**
```bash
# Use low-memory config
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml
```

**Docker build fails:**
```bash
# Build with verbose output
docker build -f deployment/Dockerfile -t test . --progress=plain --no-cache
```

**API returns 503:**
```bash
# Check logs
docker logs fraud-api

# Manually reload model
curl -X POST http://localhost:8000/model/reload
```

**Evidently import error:**
```bash
# Ensure correct version is installed
pip install evidently>=0.7.0
```

### Logs and Debugging

```bash
# API logs
docker logs -f fraud-api

# Prefect flow logs
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
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
