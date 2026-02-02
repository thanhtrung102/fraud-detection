# System Architecture

Comprehensive architecture documentation for the Fraud Detection MLOps system.

---

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [Infrastructure](#infrastructure)
- [Directory Structure](#directory-structure)
- [Technology Decisions](#technology-decisions)
- [Scaling Considerations](#scaling-considerations)
- [Security](#security)

---

## Overview

The Fraud Detection system is a production-ready MLOps platform for real-time credit card fraud detection. It implements a stacking ensemble approach combining XGBoost, LightGBM, and CatBoost with comprehensive experiment tracking, monitoring, and deployment capabilities.

### Design Principles

1. **Modularity**: Each component is independent and can be developed/deployed separately
2. **Reproducibility**: All experiments are tracked with full lineage
3. **Scalability**: Containerized services with horizontal scaling support
4. **Observability**: Comprehensive monitoring and drift detection
5. **Simplicity**: Docker Compose profiles for flexible deployment

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │  Streamlit  │    │   FastAPI   │    │   Prefect   │                     │
│   │     UI      │    │     API     │    │   Flows     │                     │
│   │  (Port 8501)│    │  (Port 8000)│    │             │                     │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                     │
│          │                  │                  │                             │
└──────────┼──────────────────┼──────────────────┼─────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SERVICE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │   MLflow    │    │  Evidently  │    │    Data     │                     │
│   │   Server    │    │  Monitoring │    │  Validator  │                     │
│   │  (Port 5000)│    │             │    │             │                     │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                     │
│          │                  │                  │                             │
└──────────┼──────────────────┼──────────────────┼─────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │ PostgreSQL  │    │    MinIO    │    │   Local     │                     │
│   │  (Metadata) │    │ (Artifacts) │    │   Files     │                     │
│   │  (Port 5432)│    │  (Port 9000)│    │             │                     │
│   └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Streamlit UI (`ui/`)

Interactive web interface for fraud analysis.

| File | Purpose |
|------|---------|
| `fraud_app.py` | Main Streamlit application |
| `utils/model_loader.py` | MLflow model discovery and loading |
| `utils/predictor.py` | Prediction logic and sample generation |

**Capabilities:**
- CSV file upload with validation
- Manual transaction entry
- Sample data generation
- Batch predictions with visualizations
- CSV export of results

**Dependencies:**
- MLflow server for model discovery
- Local model files as fallback

### 2. FastAPI Server (`deployment/api/`)

REST API for real-time fraud predictions.

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application and endpoints |
| `schemas.py` | Pydantic request/response models |

**Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /model/reload` - Hot reload model

### 3. MLflow Server

Experiment tracking and model registry.

**Configuration:**
```yaml
Backend Store: PostgreSQL (postgresql://mlflow:mlflow@postgres:5432/mlflow)
Artifact Store: MinIO S3 (s3://mlflow-artifacts/)
Default Experiment: fraud-detection
```

**Tracked Items:**
- Parameters (hyperparameters, thresholds)
- Metrics (AUC-ROC, accuracy, precision, recall)
- Artifacts (models, visualizations, reports)
- Model registry with staging/production stages

### 4. MinIO Object Storage

S3-compatible artifact storage.

**Buckets:**
- `mlflow-artifacts` - MLflow experiment artifacts
- Initialized automatically via `minio-mc` service

**Access:**
- API: `http://localhost:9000`
- Console: `http://localhost:9001`
- Credentials: `minioadmin / minioadmin`

### 5. PostgreSQL Database

Persistent metadata storage for MLflow.

**Schema:**
- Experiments and runs
- Metrics and parameters
- Model registry
- Tags and annotations

### 6. Data Validation (`src/validation.py`)

Comprehensive data quality checks.

**Validation Layers:**
1. **Schema**: Required columns, data types
2. **Quality**: Missing values, duplicates, outliers
3. **Ranges**: Min/max value constraints
4. **Distribution**: Drift detection vs training data

### 7. Visualization Suite (`src/visualization.py`)

Automated performance report generation.

**Generated Charts:**
- Confusion matrix heatmap
- ROC curve with AUC
- Precision-Recall curve
- Probability distributions
- Feature importance rankings
- Combined HTML report

### 8. Monitoring (`mlops/monitoring.py`)

Evidently-based drift detection and performance monitoring.

**Reports:**
- Data drift reports
- Model performance reports
- Data quality reports

---

## Data Flow

### Training Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Data    │───▶│ Preprocessing│───▶│   Feature    │───▶│   Training   │
│  (Kaggle)    │    │  & Merging   │    │  Engineering │    │   Pipeline   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                    ┌──────────────────────────────────────────────┘
                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    SHAP      │───▶│   Stacking   │───▶│   MLflow     │───▶│    Model     │
│  Selection   │    │   Ensemble   │    │   Logging    │    │   Registry   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Inference Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Transaction │───▶│    Data      │───▶│   Feature    │───▶│    Model     │
│    Input     │    │  Validation  │    │  Transform   │    │   Predict    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                    ┌──────────────────────────────────────────────┘
                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Risk       │───▶│   Response   │───▶│   Logging    │
│  Assessment  │    │  Formatting  │    │ & Monitoring │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Model Loading Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Model Loading Priority                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  1. MLflow Production  │
                    │     Stage Model        │
                    └───────────┬────────────┘
                                │ Not Found
                                ▼
                    ┌────────────────────────┐
                    │  2. MLflow Latest      │
                    │     Experiment Run     │
                    └───────────┬────────────┘
                                │ Not Found
                                ▼
                    ┌────────────────────────┐
                    │  3. Local Directory    │
                    │     (models/)          │
                    └────────────────────────┘
```

---

## Model Architecture

### Stacking Ensemble

```
                    ┌─────────────────────────────────────┐
                    │           Input Features            │
                    │        (30 SHAP-selected)           │
                    └─────────────────┬───────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
   ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
   │    XGBoost    │         │   LightGBM    │         │   CatBoost    │
   │  (Base Model) │         │  (Base Model) │         │  (Base Model) │
   └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
           │                          │                          │
           │    predict_proba()       │    predict_proba()       │
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Meta Features (3 probabilities)  │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Logistic Regression            │
                    │        (Meta-Learner)               │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Final Fraud Probability        │
                    └─────────────────────────────────────┘
```

### Model Files

| File | Description | Size |
|------|-------------|------|
| `xgb_model.joblib` | XGBoost base model | ~50MB |
| `lgbm_model.joblib` | LightGBM base model | ~30MB |
| `catboost_model.cbm` | CatBoost base model | ~40MB |
| `meta_learner.joblib` | Logistic regression | ~1KB |
| `feature_names.json` | Feature list | ~1KB |

### Feature Groups

| Group | Features | Description |
|-------|----------|-------------|
| Transaction | `TransactionDT`, `TransactionAmt` | Core transaction data |
| Card | `card1`-`card6` | Card identifiers |
| Address | `addr1`, `addr2`, `dist1`, `dist2` | Location features |
| Count | `C1`-`C14` | Aggregated counts |
| Time Delta | `D1`-`D15` | Time-based features |
| Vesta | `V258`, `V280`, `V282`, etc. | Engineered features |

---

## Infrastructure

### Docker Services

```yaml
services:
  postgres:      # PostgreSQL for MLflow metadata
  minio:         # S3-compatible artifact storage
  minio-mc:      # MinIO client for bucket init
  mlflow:        # MLflow tracking server
  prefect:       # Workflow orchestration
  api:           # FastAPI prediction service
  streamlit-ui:  # Interactive web UI
```

### Docker Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| `full` | All services | Complete deployment |
| `mlflow` | postgres, minio, minio-mc, mlflow | MLOps infrastructure |
| `api` | api | API only |
| `ui` | streamlit-ui | UI only |
| `training` | mlflow profile + prefect | Training workflows |
| `monitoring` | Evidently services | Drift monitoring |

### Port Mapping

| Service | Port | Description |
|---------|------|-------------|
| FastAPI | 8000 | REST API |
| Streamlit | 8501 | Web UI |
| MLflow | 5000 | Experiment tracking |
| MinIO API | 9000 | S3-compatible storage |
| MinIO Console | 9001 | Storage management UI |
| PostgreSQL | 5432 | Database |
| Prefect | 4200 | Workflow UI |

### Resource Requirements

| Environment | CPU | RAM | Storage |
|-------------|-----|-----|---------|
| Development | 2 cores | 8GB | 10GB |
| Production | 4 cores | 16GB | 50GB |
| Training (Optuna) | 4+ cores | 16GB+ | 20GB |

---

## Directory Structure

```
fraud-detection/
├── config/                     # Configuration files
│   ├── params.yaml            # Default parameters
│   ├── params_codespaces.yaml # Low-memory config
│   ├── params_production.yaml # Production config
│   └── validation.yaml        # Data validation rules
│
├── data/                       # Data directory (gitignored)
│   ├── train_transaction.csv
│   └── train_identity.csv
│
├── deployment/                 # Deployment configs
│   ├── api/                   # FastAPI application
│   │   ├── main.py
│   │   └── schemas.py
│   ├── docker-compose.yml     # Docker services
│   ├── Dockerfile             # API Dockerfile
│   └── Dockerfile.ui          # Streamlit Dockerfile
│
├── docs/                       # Documentation
│   ├── MLOPS.md               # MLOps guide
│   ├── API.md                 # API reference
│   └── ARCHITECTURE.md        # This file
│
├── infrastructure/             # Terraform IaC
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
│
├── mlops/                      # MLOps utilities
│   ├── monitoring.py          # Evidently integration
│   ├── tracking.py            # MLflow helpers
│   └── s3_utils.py            # MinIO/S3 utilities
│
├── models/                     # Trained models (gitignored)
│   ├── xgb_model.joblib
│   ├── lgbm_model.joblib
│   ├── catboost_model.cbm
│   ├── meta_learner.joblib
│   └── feature_names.json
│
├── pipelines/                  # Prefect pipelines
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── monitoring_pipeline.py
│
├── src/                        # Core source code
│   ├── data_processor.py      # Data preprocessing
│   ├── feature_engineer.py    # Feature engineering
│   ├── stacking_model.py      # Ensemble model
│   ├── validation.py          # Data validation
│   └── visualization.py       # Chart generation
│
├── templates/                  # HTML templates
│   └── model_report.html      # Report template
│
├── tests/                      # Test suite
│   ├── test_api.py
│   ├── test_model.py
│   └── test_validation.py
│
├── ui/                         # Streamlit UI
│   ├── fraud_app.py           # Main application
│   └── utils/
│       ├── model_loader.py    # Model loading
│       └── predictor.py       # Prediction logic
│
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                 # Continuous integration
│   └── cd.yml                 # Continuous deployment
│
├── Makefile                    # Development commands
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## Technology Decisions

### Why Stacking Ensemble?

| Approach | Pros | Cons |
|----------|------|------|
| Single Model | Simple, fast | Lower accuracy |
| Voting Ensemble | Easy to implement | Limited improvement |
| **Stacking** | Best accuracy, learns optimal weighting | More complex |

**Decision**: Stacking provides 2-3% AUC improvement over single models.

### Why MinIO over Local Storage?

| Storage | Pros | Cons |
|---------|------|------|
| Local SQLite | Simple, no setup | Not scalable, single node |
| Cloud S3 | Scalable, managed | Requires cloud account |
| **MinIO** | S3-compatible, local/cloud, easy migration | Additional service |

**Decision**: MinIO provides S3 compatibility for cloud migration path.

### Why PostgreSQL over SQLite?

| Database | Pros | Cons |
|----------|------|------|
| SQLite | No setup, embedded | Concurrent write issues |
| **PostgreSQL** | ACID, concurrent access, scalable | Additional service |

**Decision**: PostgreSQL supports multiple concurrent MLflow clients.

### Why Streamlit over Custom Frontend?

| Framework | Pros | Cons |
|-----------|------|------|
| React/Vue | Full control, modern UX | Development time |
| **Streamlit** | Rapid development, Python-native | Less customizable |
| Gradio | ML-focused, simple | Limited layouts |

**Decision**: Streamlit enables rapid iteration with Python-only skills.

---

## Scaling Considerations

### Horizontal Scaling

```yaml
# Scale API replicas
docker-compose up -d --scale api=3

# Load balancer (nginx example)
upstream fraud_api {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
}
```

### Vertical Scaling

| Component | Scaling Strategy |
|-----------|------------------|
| API | Add replicas, increase CPU/RAM |
| MLflow | Increase PostgreSQL resources |
| MinIO | Add storage nodes (distributed mode) |
| Training | Use larger instance, GPU support |

### Cloud Migration Path

1. **Phase 1**: Docker Compose on single VM
2. **Phase 2**: Managed services (Cloud SQL, GCS)
3. **Phase 3**: Kubernetes (GKE) for auto-scaling
4. **Phase 4**: Serverless (Cloud Run, Cloud Functions)

---

## Security

### Current Implementation

- CORS enabled for all origins (development)
- No authentication (development)
- Local network isolation via Docker

### Production Recommendations

| Layer | Recommendation |
|-------|----------------|
| Network | VPC, private subnets, load balancer |
| Authentication | OAuth2/JWT, API keys |
| Authorization | Role-based access control |
| Encryption | TLS/HTTPS, encrypted storage |
| Secrets | Cloud Secret Manager, no hardcoded credentials |
| Monitoring | Audit logs, intrusion detection |

### Sensitive Data Handling

- Transaction data should be anonymized
- PII columns excluded from logging
- Model artifacts encrypted at rest
- Access logs retained for compliance

---

## References

- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MinIO Documentation](https://min.io/docs/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
