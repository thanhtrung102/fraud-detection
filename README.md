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
- **Interactive UI**: Streamlit web interface for real-time fraud analysis
- **Production Storage**: MinIO (S3-compatible) artifact storage with PostgreSQL backend
- **Data Validation**: Comprehensive schema and quality validation pipeline
- **Visualization Suite**: Auto-generated HTML reports with performance charts
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
                    |        Data Validation              |
                    |  - Schema validation                |
                    |  - Quality checks                   |
                    |  - Range validation                 |
                    +------------------+------------------+
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
    +----------------------------------+----------------------------------+
    |                    |                    |                           |
+---v----+         +-----v-----+        +-----v-----+              +------v------+
|Streamlit|        |  FastAPI  |        |  MLflow   |              |  Evidently  |
|   UI    |        |    API    |        | + MinIO   |              |  Monitoring |
+---------+        +-----------+        +-----------+              +-------------+
```

### Technology Stack

| Category | Technologies |
|----------|-------------|
| **ML Models** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Explainability** | SHAP, LIME, Partial Dependence Plots |
| **MLOps** | MLflow, Prefect, Evidently |
| **Storage** | MinIO (S3-compatible), PostgreSQL |
| **API** | FastAPI, Uvicorn |
| **UI** | Streamlit, Plotly |
| **Deployment** | Docker, Docker Compose, GCP Cloud Run |
| **Infrastructure** | Terraform, GitHub Actions CI/CD |
| **Data Processing** | Pandas, NumPy, Imbalanced-learn (SMOTE) |

---

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- ~2GB storage for dataset

### Option 1: Docker (Recommended)

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
# Clone and setup
git clone https://github.com/thanhtrung102/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

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

## Streamlit UI

The interactive Streamlit UI provides a user-friendly interface for fraud detection:

### Features

- **Model Loading**: Auto-discover models from MLflow or load from local storage
- **Multiple Input Methods**:
  - CSV file upload with validation
  - Manual transaction entry
  - Sample data generation for testing
- **Real-time Predictions**: Single and batch fraud detection
- **Visualizations**: Risk distribution, probability histograms
- **Export**: Download predictions as CSV

### Usage

```bash
# Start with Docker
make docker-ui

# Or locally
make serve-ui
# Open http://localhost:8501
```

### Screenshots

The UI includes:
- Sidebar for model configuration and threshold adjustment
- Tabbed interface for different input methods
- Interactive charts showing fraud probability distribution
- Detailed results table with risk levels
- Export functionality for predictions

---

## Docker Services

### Service Profiles

| Command | Services Started | Use Case |
|---------|-----------------|----------|
| `make docker-full` | All services | Full development environment |
| `make docker-mlops` | MLflow, MinIO, PostgreSQL | MLOps infrastructure only |
| `make docker-serve` | API, Streamlit UI | Serving only (requires MLflow) |
| `make docker-ui` | Streamlit UI | UI development |
| `make docker-train` | Training worker | Run training in container |

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit UI | http://localhost:8501 | - |
| FastAPI | http://localhost:8000/docs | - |
| MLflow | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Prefect | http://localhost:4200 | - |

### Docker Commands

```bash
# Build all images
make docker-build

# Start full stack
make docker-full

# View logs
make docker-logs
make docker-logs-mlflow
make docker-logs-ui

# Stop all services
make docker-down
```

---

## Data Validation

The platform includes comprehensive data validation via `src/validation.py`:

### Validation Layers

1. **Schema Validation**: Required columns and data types
2. **Data Quality**: Missing values, duplicates, outliers
3. **Value Ranges**: Min/max constraints for numeric columns
4. **Distribution Shift**: Detect drift between training and inference data

### Configuration

Edit `config/validation.yaml`:

```yaml
validation:
  required_columns:
    - TransactionDT
    - TransactionAmt
    - card1

  value_ranges:
    TransactionAmt:
      min: 0.01
      max: 999999.99

  thresholds:
    max_missing_pct: 30
    max_duplicate_pct: 1
```

### Usage

```python
from src.validation import DataValidator

validator = DataValidator("config/validation.yaml")
is_valid, report = validator.validate_for_training(df)
```

---

## Visualization Suite

Auto-generate comprehensive model performance reports:

### Generated Charts

- Confusion Matrix Heatmap
- ROC Curve with AUC
- Precision-Recall Curve
- Probability Distribution by Class
- Feature Importance (Top 20)

### HTML Reports

The `ModelVisualizer` class generates self-contained HTML reports with embedded charts:

```python
from src.visualization import ModelVisualizer

visualizer = ModelVisualizer()
saved_files = visualizer.create_comprehensive_report(
    y_true, y_pred, y_proba, metrics,
    feature_importance=importance_df,
    save_dir="results/visualizations"
)
```

Reports are automatically logged to MLflow artifacts.

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

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Single transaction prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/model/reload` | POST | Reload model from disk |

### Example Request

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

## Project Structure

```
fraud-detection/
├── config/                 # Configuration files
│   ├── params.yaml         # Default parameters
│   ├── params_production.yaml
│   ├── params_codespaces.yaml
│   └── validation.yaml     # Data validation config
├── data/                   # Data files (git-ignored)
├── deployment/
│   ├── api/               # FastAPI application
│   ├── docker-compose.yml # Docker orchestration
│   ├── Dockerfile         # API container
│   ├── Dockerfile.ui      # Streamlit container
│   └── Dockerfile.training
├── src/                   # Core ML modules
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── stacking_model.py
│   ├── validation.py      # Data validation
│   └── visualization.py   # Chart generation
├── mlops/                 # MLOps utilities
│   ├── tracking.py
│   ├── registry.py
│   ├── monitoring.py
│   └── s3_utils.py        # MinIO/S3 utilities
├── pipelines/             # Prefect workflows
├── ui/                    # Streamlit application
│   ├── fraud_app.py
│   └── utils/
├── templates/             # HTML report templates
├── tests/                 # Unit + integration tests
├── infrastructure/        # Terraform configs
└── Makefile              # Build automation
```

---

## Makefile Commands

```bash
# Setup
make install          # Install production dependencies
make install-dev      # Install development dependencies

# Development
make test             # Run all tests
make lint             # Run linting
make format           # Format code

# Training
make train            # Run full training pipeline
make train-quick      # Run without Optuna

# Serving
make serve            # Start FastAPI server
make serve-ui         # Start Streamlit UI
make mlflow-ui        # Start MLflow UI

# Docker
make docker-build     # Build all images
make docker-full      # Start all services
make docker-mlops     # Start MLOps stack only
make docker-serve     # Start API + UI only
make docker-down      # Stop all services

# Infrastructure
make terraform-init   # Initialize Terraform
make terraform-apply  # Deploy infrastructure
```

---

## Production Deployment

### GCP Cloud Run

```bash
# Build and push image
docker build -f deployment/Dockerfile -t us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest

# Deploy
gcloud run deploy fraud-detection-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/fraud-detection/api:latest \
  --region us-central1 \
  --memory 2Gi \
  --allow-unauthenticated
```

### Terraform

```bash
cd infrastructure
terraform init
terraform plan -var="project_id=your-gcp-project"
terraform apply -var="project_id=your-gcp-project"
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
- [Streamlit](https://streamlit.io/) for the interactive UI framework
- [MinIO](https://min.io/) for S3-compatible object storage
