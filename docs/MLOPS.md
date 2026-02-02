# MLOps Guide for Fraud Detection

Complete guide for MLOps components including experiment tracking, workflow orchestration, model deployment, monitoring, and CI/CD.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Storage Layer](#storage-layer)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Monitoring Pipeline](#monitoring-pipeline)
- [Streamlit UI](#streamlit-ui)
- [Data Validation](#data-validation)
- [Visualization Suite](#visualization-suite)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Infrastructure as Code](#infrastructure-as-code)
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
|   |                        Storage Layer                          |         |
|   |  +-------------+  +-------------+  +-------------+            |         |
|   |  | PostgreSQL  |  |    MinIO    |  |   Models    |            |         |
|   |  | (Metadata)  |  | (Artifacts) |  |   (Local)   |            |         |
|   |  +-------------+  +-------------+  +-------------+            |         |
|   +---------------------------------------------------------------+         |
|                               |                                             |
|   +---------------------------------------------------------------+         |
|   |                        Pipelines                               |         |
|   |  +----------+  +----------+  +--------------+                  |         |
|   |  | Training |  | Inference|  |  Monitoring  |                  |         |
|   |  | Pipeline |  | Pipeline |  |   Pipeline   |                  |         |
|   |  +----+-----+  +----+-----+  +------+-------+                  |         |
|   +-------|-------------|---------------|---------------------------+         |
|           |             |               |                                    |
|           v             v               v                                    |
|   +-------------+  +-------------+  +-------------+  +-------------+        |
|   | Streamlit   |  |   FastAPI   |  |  Evidently  |  |Visualization|        |
|   |     UI      |  |   Server    |  |   Reports   |  |    Suite    |        |
|   +-------------+  +------+------+  +-------------+  +-------------+        |
|                           |                                                  |
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
| Artifact Storage | MinIO (S3-compatible) | Store models, charts, reports |
| Metadata Store | PostgreSQL | MLflow backend database |
| Workflow Orchestration | Prefect | Pipeline orchestration and scheduling |
| Model Serving | FastAPI | REST API for predictions |
| Interactive UI | Streamlit | Web interface for fraud analysis |
| Monitoring | Evidently | Data drift and model performance |
| Data Validation | Custom DataValidator | Schema, quality, range validation |
| Visualization | ModelVisualizer | Performance charts and HTML reports |
| Containerization | Docker | Packaging and deployment |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Infrastructure | Terraform | GCP resource provisioning |
| Cloud Platform | GCP Cloud Run | Serverless deployment |

---

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)

### Option 1: Docker Full Stack (Recommended)

```bash
# Clone repository
git clone https://github.com/thanhtrung102/fraud-detection.git
cd fraud-detection

# Start all services (MLflow, MinIO, API, Streamlit UI)
make docker-full

# Access the services:
# - Streamlit UI: http://localhost:8501
# - FastAPI Docs: http://localhost:8000/docs
# - MLflow UI: http://localhost:5000
# - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/

# Run training
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# Start services
make serve      # FastAPI on port 8000
make serve-ui   # Streamlit on port 8501
```

---

## Storage Layer

### MinIO S3-Compatible Storage

MinIO provides S3-compatible object storage for MLflow artifacts, including models, visualizations, and reports.

#### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | MinIO endpoint |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin` | MinIO secret key |

#### S3 Utilities (`mlops/s3_utils.py`)

```python
from mlops.s3_utils import S3ArtifactManager

# Initialize manager
manager = S3ArtifactManager()

# Upload artifact
manager.upload_artifact(
    local_path="results/confusion_matrix.png",
    s3_key="experiments/run_123/confusion_matrix.png"
)

# List artifacts for a run
artifacts = manager.list_artifacts(prefix="experiments/run_123/")

# Sync all MLflow artifacts to S3
from mlops.s3_utils import sync_mlflow_artifacts_to_s3
sync_mlflow_artifacts_to_s3(run_id="abc123")
```

#### Accessing MinIO Console

```bash
# Start MinIO
make docker-mlops

# Access console
open http://localhost:9001
# Login: minioadmin / minioadmin
```

### PostgreSQL Backend

MLflow uses PostgreSQL for metadata storage instead of SQLite:

```yaml
# Connection string
postgresql://mlflow:mlflow@postgres:5432/mlflow
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

## Streamlit UI

The interactive Streamlit UI provides a user-friendly interface for fraud detection.

### Features

- **Model Loading**: Auto-discover models from MLflow or load from local storage
- **Multiple Input Methods**:
  - CSV file upload with validation
  - Manual transaction entry
  - Sample data generation for testing
- **Real-time Predictions**: Single and batch fraud detection
- **Visualizations**: Risk distribution, probability histograms
- **Export**: Download predictions as CSV

### Starting the UI

```bash
# With Docker
make docker-ui

# Locally
make serve-ui
# Open http://localhost:8501
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `MODEL_DIR` | `/app/models` | Local model directory |
| `DEFAULT_THRESHOLD` | `0.5` | Default classification threshold |

### Model Loading Priority

1. **Production Model**: MLflow registry production stage
2. **Latest Run**: Most recent MLflow experiment run
3. **Local Directory**: `models/` folder with joblib files

---

## Data Validation

Comprehensive data validation via `src/validation.py`.

### Validation Layers

1. **Schema Validation**: Required columns and data types
2. **Data Quality**: Missing values, duplicates, outliers
3. **Value Ranges**: Min/max constraints for numeric columns
4. **Distribution Shift**: Detect drift between training and inference data

### Configuration (`config/validation.yaml`)

```yaml
validation:
  required_columns:
    - TransactionDT
    - TransactionAmt
    - card1
    - C1
    - C14

  value_ranges:
    TransactionAmt:
      min: 0.01
      max: 999999.99
    card1:
      min: 1000
      max: 20000

  thresholds:
    max_missing_pct: 30
    max_duplicate_pct: 1
    outlier_zscore_threshold: 3
```

### Usage

```python
from src.validation import DataValidator

# Initialize validator
validator = DataValidator("config/validation.yaml")

# Validate training data
is_valid, report = validator.validate_for_training(df)

if not is_valid:
    print(f"Validation failed with {report['total_issues']} issues")
    print(f"Schema errors: {report['schema_validation']['errors']}")
    print(f"Range errors: {report['value_ranges']['errors']}")

# Validate inference data with drift detection
is_valid, report = validator.validate_for_inference(df, training_stats)

if report.get("distribution_shift", {}).get("drift_detected"):
    print("Distribution shift detected in features:")
    for feature in report["distribution_shift"]["shifted_features"]:
        print(f"  - {feature}")

# Save report
validator.save_report(report, "results/validation_report.json")
```

---

## Visualization Suite

Auto-generate comprehensive model performance reports via `src/visualization.py`.

### Generated Charts

| Chart | Description |
|-------|-------------|
| Confusion Matrix | Heatmap of TP, TN, FP, FN |
| ROC Curve | With AUC score annotation |
| Precision-Recall Curve | With optimal threshold marker |
| Probability Distribution | By class (fraud vs legitimate) |
| Feature Importance | Top 20 features ranked |

### Usage

```python
from src.visualization import ModelVisualizer

visualizer = ModelVisualizer()

# Generate comprehensive report
saved_files = visualizer.create_comprehensive_report(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    metrics={"accuracy": 0.98, "auc_roc": 0.95},
    feature_importance=importance_df,
    save_dir="results/visualizations"
)

# Files created:
# - confusion_matrix.png
# - roc_curve.png
# - precision_recall_curve.png
# - probability_distribution.png
# - feature_importance.png
# - model_report.html (combined report)
```

### HTML Reports

Self-contained HTML reports with embedded base64 images are automatically generated and logged to MLflow artifacts.

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

# 3. Start MLOps infrastructure
make docker-mlops

# 4. Run training with visualization
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# 5. View MLflow results
open http://localhost:5000

# 6. Start Streamlit UI
make serve-ui
open http://localhost:8501

# 7. Run inference
python pipelines/inference_pipeline.py

# 8. Generate monitoring reports
python pipelines/monitoring_pipeline.py

# 9. Start API server
make serve
open http://localhost:8000/docs
```

### Docker Commands

```bash
# Start all services
make docker-full

# Start MLOps stack only (MLflow, MinIO, PostgreSQL)
make docker-mlops

# Start API + UI only
make docker-serve

# Start Streamlit UI only
make docker-ui

# View logs
make docker-logs
make docker-logs-mlflow
make docker-logs-ui

# Stop all services
make docker-down
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
# Start all services with profiles
make docker-full

# Services:
# - Streamlit UI: http://localhost:8501
# - FastAPI: http://localhost:8000
# - MLflow: http://localhost:5000
# - MinIO Console: http://localhost:9001

# Scale API replicas
docker-compose -f deployment/docker-compose.yml up -d --scale api=3

# Stop services
make docker-down
```

### Option 2: GCP Cloud Run

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

# 3. Build and push image
docker build -f deployment/Dockerfile -t us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest

# 4. Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

### Production Checklist

- [ ] Configure GCP project and enable APIs
- [ ] Set up Artifact Registry for Docker images
- [ ] Deploy infrastructure with Terraform
- [ ] Configure GitHub secrets for CI/CD
- [ ] Upload trained model to Cloud Storage
- [ ] Deploy API to Cloud Run
- [ ] Deploy Streamlit UI (optional)
- [ ] Set up monitoring and alerting
- [ ] Configure data validation thresholds
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

# Initialize and apply
terraform init
terraform plan
terraform apply
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

## Troubleshooting

### Common Issues

**MinIO connection error:**
```bash
# Check MinIO is running
curl http://localhost:9000/minio/health/live

# Check environment variables
echo $MLFLOW_S3_ENDPOINT_URL
echo $AWS_ACCESS_KEY_ID
```

**MLflow connection error:**
```bash
# Check MLflow server is running
curl http://localhost:5000/health

# Start if needed with PostgreSQL backend
docker-compose -f deployment/docker-compose.yml --profile mlflow up -d
```

**Streamlit UI not loading models:**
```bash
# Check MLflow connection
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Verify model files exist
ls -la models/
# Expected: xgb_model.joblib, lgbm_model.joblib, catboost_model.cbm,
#           meta_learner.joblib, feature_names.json
```

**Model not loading:**
```bash
# Verify model files exist
ls -la models/

# Check permissions
chmod 755 models/*

# Manually reload model in API
curl -X POST http://localhost:8000/model/reload
```

**Memory error during training:**
```bash
# Use low-memory config
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml
```

**Data validation failures:**
```bash
# Check validation report
python -c "
from src.validation import DataValidator
import pandas as pd

df = pd.read_csv('data/train_transaction.csv', nrows=1000)
validator = DataValidator()
is_valid, report = validator.validate_for_training(df)
print(f'Valid: {is_valid}')
print(f'Issues: {report[\"total_issues\"]}')
"
```

### Logs and Debugging

```bash
# API logs
docker-compose -f deployment/docker-compose.yml logs -f api

# Streamlit UI logs
docker-compose -f deployment/docker-compose.yml logs -f streamlit-ui

# MLflow logs
docker-compose -f deployment/docker-compose.yml logs -f mlflow

# MinIO logs
docker-compose -f deployment/docker-compose.yml logs -f minio

# Prefect flow logs
prefect flow-run logs <run-id>

# Monitoring reports
ls -la monitoring/evidently_reports/
```

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
- [Prefect Documentation](https://docs.prefect.io/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
