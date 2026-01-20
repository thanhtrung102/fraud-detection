# Credit Card Fraud Detection MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Kaggle-IEEE--CIS%20Fraud-20BEFF.svg)](https://www.kaggle.com/c/ieee-fraud-detection)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/thanhtrung102/fraud-detection)

A production-ready fraud detection system using stacking ensemble methods combined with explainable AI techniques. Based on the paper ["Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods"](https://arxiv.org/html/2505.10050v1) (arXiv:2505.10050).

## Overview

### Key Features

- **Stacking Ensemble Model**: Combines XGBoost, LightGBM, and CatBoost with XGBoost meta-learner
- **Explainable AI**: Full transparency with SHAP, LIME, and Partial Dependence Plots
- **MLOps Integration**: MLflow tracking, Prefect orchestration, Evidently monitoring
- **Production Ready**: FastAPI serving, Docker deployment, GCP Cloud Run support
- **Optuna Tuning**: Automated hyperparameter optimization (20 trials per model)
- **SHAP Feature Selection**: Top 30 features selected based on SHAP importance

---

## Architecture

```
                              +------------------+
                              |   Input Data     |
                              | (590K+ records)  |
                              +--------+---------+
                                       |
                    +------------------v------------------+
                    |           Preprocessing             |
                    |  - Missing value imputation         |
                    |  - Label encoding                   |
                    |  - SMOTE class balancing            |
                    +------------------+------------------+
                                       |
                    +------------------v------------------+
                    |      SHAP Feature Selection         |
                    |         (Top 30 features)           |
                    +------------------+------------------+
                                       |
                    +------------------v------------------+
                    |        Stacking Ensemble            |
                    |  +--------+ +--------+ +--------+   |
                    |  |XGBoost | |LightGBM| |CatBoost|   |
                    |  +---+----+ +---+----+ +---+----+   |
                    |      |          |          |        |
                    |      +----------+----------+        |
                    |                 |                   |
                    |         +-------v-------+           |
                    |         |   XGBoost     |           |
                    |         | Meta-Learner  |           |
                    |         +---------------+           |
                    +------------------+------------------+
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
+--------v--------+         +----------v----------+       +----------v----------+
|    MLflow       |         |      FastAPI        |       |     Evidently       |
|   Experiment    |         |     Inference       |       |    Monitoring       |
|    Tracking     |         |       API           |       |     Reports         |
+-----------------+         +---------------------+       +---------------------+
```

### Technology Stack

| Category | Technologies |
|----------|-------------|
| **ML Models** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Explainability** | SHAP, LIME, Partial Dependence Plots |
| **MLOps** | MLflow, Prefect, Evidently |
| **API** | FastAPI, Uvicorn |
| **Deployment** | Docker, Docker Compose, GCP Cloud Run |
| **Infrastructure** | Terraform, GitHub Actions CI/CD |
| **Data Processing** | Pandas, NumPy, Imbalanced-learn (SMOTE) |

---

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- ~2GB storage for dataset

### 1. Clone and Setup

```bash
git clone https://github.com/thanhtrung102/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection):

```bash
# Using Kaggle CLI
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

### 3. Run Training Pipeline

```bash
# Quick start (low memory mode)
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# Full paper methodology
python -m src.main
```

### 4. Start API Server

```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

---

## ML Pipeline Features

### Training Options

| Use Case | Command |
|----------|---------|
| **Research (paper reproduction)** | `python -m src.main` |
| **MLOps with tracking** | `python pipelines/training_pipeline.py` |
| **Low memory (8GB RAM)** | `python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml` |
| **Production (16GB+ RAM)** | `python pipelines/training_pipeline.py --config-path config/params_production.yaml --use-optuna` |

### CLI Arguments

| Flag | Description |
|------|-------------|
| `--config-path PATH` | Config file path (default: `config/params.yaml`) |
| `--use-optuna` | Enable Optuna hyperparameter tuning (20 trials) |
| `--no-feature-selection` | Disable SHAP feature selection |
| `--register-model` | Register trained model to MLflow registry |

### Configuration Profiles

| Profile | Sample Size | RAM Required | Use Case |
|---------|-------------|--------------|----------|
| `params.yaml` | Full (590K) | 16GB+ | Default |
| `params_production.yaml` | 300,000 | 16GB | Paper reproduction |
| `params_codespaces.yaml` | 100,000 | 8GB | Limited resources |

---

## Inference System

### Batch Inference

```bash
python pipelines/inference_pipeline.py \
  --data-path data/transactions.csv \
  --model-dir models \
  --output-path results/predictions.csv \
  --threshold 0.44
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Single transaction prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |

### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "TransactionAmt": 150.0,
      "card1": 12345,
      "C14": 1.0
    }
  }'
```

---

## Performance & Metrics

### Achieved Results (100K Sample)

| Metric | Achieved | Paper Target |
|--------|----------|--------------|
| **Accuracy** | 97.89% | 99% |
| **AUC-ROC** | 0.9195 | 0.99 |
| **Precision** | 78.88% | 99% |
| **Recall** | 54.26% | 99% |

### Top Features (SHAP Importance)

1. C14 (transaction category)
2. card6 (card type)
3. TransactionAmt
4. card1
5. V308

---

## Local Development

### Complete Local Workflow

```bash
# 1. Train model
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# 2. View MLflow results
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000

# 3. Run batch inference
python pipelines/inference_pipeline.py

# 4. Generate monitoring reports
python pipelines/monitoring_pipeline.py

# 5. Start API server
uvicorn api.main:app --reload --port 8000

# 6. Test prediction
curl http://localhost:8000/health
```

### Docker Development

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

---

## Production Deployment

### Deployment Options

| Option | Best For | Complexity |
|--------|----------|------------|
| **Docker Compose** | Self-hosted, small-scale | Low |
| **GCP Cloud Run** | Serverless, auto-scaling | Medium |
| **Kubernetes** | Large-scale, multi-region | High |

### Docker Compose Deployment

```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# Scale API replicas
docker-compose -f deployment/docker-compose.yml up -d --scale api=3
```

### GCP Cloud Run Deployment

```bash
# 1. Build and push image
docker build -f deployment/Dockerfile -t us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest

# 2. Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest \
  --region us-central1 \
  --memory 2Gi \
  --allow-unauthenticated
```

### Terraform Infrastructure

```bash
cd infrastructure

# Initialize and deploy
terraform init
terraform plan -var="project_id=your-gcp-project"
terraform apply -var="project_id=your-gcp-project"
```

See [MLOps Documentation](docs/MLOPS.md) for detailed production deployment guide.

---

## Troubleshooting

### Common Issues

**Memory Error during training:**
```bash
# Use low-memory config
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml
```

**Model not loading in inference:**
```bash
# Verify model files exist
ls -la models/
# Should contain: xgb_model.joblib, lgbm_model.joblib, catboost_model.cbm, meta_learner.joblib
```

**API returns 500 error:**
```bash
# Check logs
docker logs fraud-api

# Reload model
curl -X POST http://localhost:8000/model/reload
```

---

## Documentation

- [MLOps Guide](docs/MLOPS.md) - Detailed MLOps setup, deployment, and monitoring
- [API Reference](docs/API.md) - API endpoints and usage
- [Architecture](docs/ARCHITECTURE.md) - System design and components

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset from Kaggle
- [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/) teams
- [SHAP](https://shap.readthedocs.io/) and [LIME](https://github.com/marcotcr/lime) for explainability tools
- [MLflow](https://mlflow.org/), [Prefect](https://www.prefect.io/), [Evidently](https://www.evidentlyai.com/) for MLOps tools
