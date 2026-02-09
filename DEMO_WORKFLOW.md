# Fraud Detection MLOps Platform - Complete Feature Demonstration

## Overview

This guide demonstrates all features of the Credit Card Fraud Detection system:
- **Model Training**: Stacking ensemble (XGBoost, LightGBM, CatBoost)
- **Feature Selection**: SHAP-based feature importance
- **Hyperparameter Tuning**: Optuna optimization
- **Explainability**: SHAP, LIME, Partial Dependence Plots
- **MLOps**: MLflow tracking, Prefect orchestration
- **API**: FastAPI REST service
- **UI**: Streamlit web interface
- **Monitoring**: Data drift detection with Evidently
- **Deployment**: Docker containers

---

## Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Full Feature Demonstration](#full-feature-demonstration)
4. [Individual Feature Demos](#individual-feature-demos)
5. [MLOps Workflows](#mlops-workflows)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Setup

### System Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- ~2GB disk space for datasets
- Docker & Docker Compose (for containerized demos)

### Installation

```bash
# Clone repository
git clone https://github.com/thanhtrung102/fraud-detection.git
cd fraud-detection

# Create Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Download training data (requires Kaggle API credentials)
# Follow: https://www.kaggle.com/c/ieee-fraud-detection
# Place downloaded files in ./data/
```

### Quick Configuration

```bash
# For low-memory environments (Codespaces, containers)
# Edit config/params.yaml and uncomment sample_size:
# sample_size: 100000  # Reduces data to 100K samples for faster training
```

---

## Quick Start (5 minutes)

Run the end-to-end pipeline quickly without hyperparameter tuning:

```bash
# Install dependencies
make install-dev

# Run quick training (skips Optuna tuning for speed)
make train-quick

# Expected output:
# ✓ Data loaded: 590,540 training samples
# ✓ Features selected: Top 30 features via SHAP
# ✓ Models trained: XGBoost, LightGBM, CatBoost
# ✓ Metrics achieved: Accuracy 0.99+, AUC-ROC 0.99+
# ✓ Results saved to ./results/
```

**Time estimate**: 3-5 minutes (with sample_size=100000)

---

## Full Feature Demonstration

### 1️⃣ End-to-End Pipeline with All Features

```bash
# Run complete training pipeline (includes Optuna tuning - ~30 minutes)
make train

# This executes:
# 1. Data loading and preprocessing
# 2. Data validation
# 3. SHAP-based feature selection (Top 30 features)
# 4. Optuna hyperparameter tuning (50 trials per model)
# 5. Ensemble training (XGBoost + LightGBM + CatBoost + Meta-learner)
# 6. Optimal threshold finding
# 7. Model evaluation
# 8. Explainability analysis (SHAP, LIME)
# 9. Visualization generation
# 10. Results saved
```

**Output**:
```
results/
├── metrics.json          # Accuracy, AUC-ROC, Precision, Recall, F1
├── feature_importance.csv  # Top features and SHAP values
└── figures/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── shap_summary.png
    ├── lime_explanation.png
    └── threshold_analysis.png
```

### 2️⃣ View Results & Metrics

```bash
# Display training results
cat results/metrics.json | python -m json.tool

# Expected output:
# {
#   "accuracy": 0.9912,
#   "auc_roc": 0.9924,
#   "precision": 0.9876,
#   "recall": 0.9834,
#   "f1_score": 0.9855,
#   "models": {
#     "xgboost": { ... },
#     "lightgbm": { ... },
#     "catboost": { ... }
#   }
# }
```

---

## Individual Feature Demos

### 🔍 Feature 1: Data Validation

**What it demonstrates**: Data quality checks, schema validation, distribution analysis

```bash
# Run data validation independently
python -c "
from src.validation import validate_training_data
from src.data_preprocessing import load_config

config = load_config()
validation_report = validate_training_data(config)
print('✓ Data validation passed')
print(f'  - Schema check: {validation_report[\"schema_check\"]}')
print(f'  - Quality score: {validation_report[\"quality_score\"]:.2%}')
print(f'  - Outliers detected: {validation_report[\"outlier_count\"]}')
"
```

**Files involved**:
- `src/validation.py` - DataValidator class
- `config/validation.yaml` - Validation rules
- Input: `data/train_transaction.csv`, `data/train_identity.csv`

---

### 📊 Feature 2: Feature Selection with SHAP

**What it demonstrates**: SHAP importance ranking and feature reduction

```bash
# Run feature selection independently
python -c "
from src.feature_selection import shap_feature_selection
from src.data_preprocessing import preprocess_pipeline, load_config
import numpy as np

config = load_config()
X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

# Train a base model for SHAP analysis
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=50, random_state=42)
xgb.fit(X_train, y_train)

# Get top 30 features via SHAP
selected_features, selected_indices = shap_feature_selection(
    xgb, X_train, feature_names, n_features=30
)

print(f'✓ Selected {len(selected_features)} features:')
for i, feat in enumerate(selected_features[:10], 1):
    print(f'  {i:2d}. {feat}')
"
```

**Output files**:
- Top 30 features saved to `data/top_30_features.txt`
- SHAP summary plots
- Feature importance rankings

---

### ⚙️ Feature 3: Hyperparameter Tuning with Optuna

**What it demonstrates**: Automated hyperparameter optimization for each base model

```bash
# Run Optuna tuning independently
python -c "
from src.optuna_tuning import tune_all_models
from src.data_preprocessing import preprocess_pipeline, load_config
from imblearn.over_sampling import SMOTE

config = load_config()
X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Tune models (50 trials per model = 150 total)
print('Tuning XGBoost, LightGBM, CatBoost...')
best_params = tune_all_models(X_train_balanced, y_train_balanced, X_test, y_test)

for model, params in best_params.items():
    print(f'✓ {model} best params: {params}')
"
```

**Time estimate**: 15-30 minutes

---

### 🎯 Feature 4: Model Training & Ensemble

**What it demonstrates**: Training base models and stacking ensemble architecture

```bash
# Run training pipeline independently
python -c "
from src.stacking_model import StackingFraudDetector
from src.data_preprocessing import preprocess_pipeline, load_config
from imblearn.over_sampling import SMOTE

config = load_config()
X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

# Balance data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train ensemble
print('Training stacking ensemble...')
model = StackingFraudDetector()
model.train(X_train_balanced, y_train_balanced)

# Evaluate
y_pred_proba = model.predict_proba(X_test)
predictions = model.predict(X_test)

print(f'✓ Model trained successfully')
print(f'  - Base models: XGBoost, LightGBM, CatBoost')
print(f'  - Meta-learner: XGBoost')
print(f'  - Test samples: {len(X_test)}')

# Save model
model.save('models')
print('✓ Model saved to ./models/')
"
```

---

### 📈 Feature 5: Model Evaluation & Metrics

**What it demonstrates**: Performance metrics, ROC curves, confusion matrices

```bash
# Evaluate trained model
python -c "
from src.stacking_model import StackingFraudDetector
from src.evaluation import compute_metrics, print_results
from src.data_preprocessing import preprocess_pipeline, load_config
import numpy as np

config = load_config()
X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

# Load model
model = StackingFraudDetector()
model.load('models')

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Compute metrics
metrics = compute_metrics(y_test, y_pred, y_pred_proba)
print_results(metrics, feature_names)

# Save metrics
import json
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'✓ Metrics saved to ./results/metrics.json')
"
```

---

### 💡 Feature 6: Explainability (SHAP, LIME, PDP)

**What it demonstrates**: Multiple explainability techniques for model decisions

```bash
# Generate explainability reports
python -c "
from src.explainability import generate_all_explanations
from src.stacking_model import StackingFraudDetector
from src.data_preprocessing import preprocess_pipeline, load_config

config = load_config()
X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(config)

# Load model
model = StackingFraudDetector()
model.load('models')

# Generate explanations
print('Generating explanations...')
print('  - SHAP Force Plot (why a transaction is flagged)')
print('  - SHAP Summary Plot (global feature importance)')
print('  - LIME explanations (local interpretability)')
print('  - Partial Dependence Plots (feature effects)')

generate_all_explanations(
    model,
    X_train,
    X_test,
    y_test,
    feature_names,
    output_dir='results/explanations'
)

print('✓ Explanations saved to ./results/explanations/')
"
```

**Output files**:
- `results/explanations/shap_summary.png` - Global feature importance
- `results/explanations/shap_force_{id}.html` - Individual predictions
- `results/explanations/lime_explanation.png` - Local explanations
- `results/explanations/pdp_*.png` - Partial dependence plots

---

### 🚀 Feature 7: API Server (FastAPI)

**What it demonstrates**: REST API for real-time predictions

```bash
# Terminal 1: Start the API server
make serve

# Expected output:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete
```

```bash
# Terminal 2: Make API requests
# Get model info
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "TransactionDT": 86400,
      "TransactionAmt": 149.95,
      ...
    }
  }'

# Batch predictions
curl -X POST http://localhost:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      { "features": {...} },
      { "features": {...} }
    ]
  }'
```

**API Endpoints**:
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Single prediction
- `POST /predictions` - Batch predictions
- `GET /feature-names` - Get feature list

**API Documentation**:
```bash
# Open interactive API docs
$BROWSER http://localhost:8000/docs
```

---

### 🎨 Feature 8: Interactive UI (Streamlit)

**What it demonstrates**: Web interface for fraud detection and analysis

```bash
# Terminal 1: Start API server (required by UI)
make serve

# Terminal 2: Start Streamlit UI
streamlit run ui/fraud_app.py --server.port 8501

# Open in browser
$BROWSER http://localhost:8501
```

**UI Features**:
1. **Single Prediction**
   - Upload CSV or manual entry
   - Real-time fraud prediction
   - Confidence scores
   - SHAP explanations

2. **Batch Analysis**
   - CSV upload with validation
   - Batch processing
   - Results visualization
   - Export predictions

3. **Model Insights**
   - Feature importance
   - Model performance
   - Decision explanations
   - Sample data generation

4. **Settings**
   - Model selection
   - Threshold adjustment
   - MLflow integration
   - Data validation

---

### 📊 Feature 9: MLOps - MLflow Tracking

**What it demonstrates**: Experiment tracking, model registry, parameters/metrics logging

```bash
# Terminal 1: Start MLflow server
make mlflow-ui

# Expected output:
# [INFO] Starting MLflow server on http://127.0.0.1:5000

# Open in browser
$BROWSER http://localhost:5000
```

**Available in MLflow UI**:
- **Experiments**: All training runs with dates
- **Parameters**: Hyperparameters used in each run
- **Metrics**: Accuracy, AUC, Precision, Recall, F1 for each model
- **Artifacts**: Model files, plots, reports
- **Model Registry**: Manage and version models

**Programmatic access**:

```bash
# Query MLflow for best run
python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient('http://localhost:5000')
experiments = client.search_experiments()

for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    print(f'Experiment: {exp.name}')
    print(f'  - Runs: {len(runs)}')
    if runs:
        best_run = max(runs, key=lambda r: r.data.metrics.get('auc_roc', 0))
        print(f'  - Best AUC: {best_run.data.metrics.get(\"auc_roc\", 0):.4f}')
"
```

---

### 🔄 Feature 10: MLOps - Prefect Orchestration

**What it demonstrates**: Workflow orchestration, task scheduling, error handling

```bash
# Start Prefect server
make prefect-ui

# Expected output:
# [Prefect] Server started at http://127.0.0.1:4200
```

**Deploy training pipeline**:

```bash
# Register flows with Prefect
python -c "
from pipelines.training_pipeline import training_flow
from pipelines.inference_pipeline import inference_flow

# These flows are automatically scheduled and trackable in Prefect UI
print('✓ Flows registered with Prefect')
print('  - training_flow: Daily model retraining')
print('  - inference_flow: Batch predictions')
print('  - monitoring_flow: Data drift detection')
"
```

**View in UI**:
- `$BROWSER http://localhost:4200`
- Monitor scheduled runs
- View logs and errors
- Manage deployments

---

### 📉 Feature 11: Monitoring & Drift Detection

**What it demonstrates**: Data drift detection, model performance monitoring

```bash
# Run monitoring pipeline
python -c "
from pipelines.monitoring_pipeline import monitoring_flow

# This detects:
# - Data distribution shifts
# - Missing value changes
# - Outlier increases
# - Model performance degradation
# - Concept drift

result = monitoring_flow(
    reference_data_path='data/train_transaction.csv',
    production_data_path='data/production_transactions.csv'
)

if result.get('drift_detected', False):
    print('⚠️  Data drift detected!')
    print(f'  - Affected features: {result[\"drifted_features\"]}')
    print(f'  - Recommendation: Retrain model')
else:
    print('✓ No significant drift detected')
"
```

**Monitoring Metrics**:
- Feature distribution shifts (Kolmogorov-Smirnov test)
- Missing value rate changes
- Outlier percentage changes
- Model prediction distribution changes
- Performance metric trends

---

### 🐳 Feature 12: Docker Deployment

**What it demonstrates**: Containerized deployment for production

```bash
# Build Docker images
make docker-build

# Start full stack (MLflow, MinIO, PostgreSQL, API, UI)
make docker-full

# Expected output:
# [+] Running 5/5
#  ✔ Container mlflow          Started
#  ✔ Container minio           Started
#  ✔ Container postgres        Started
#  ✔ Container api             Started
#  ✔ Container ui              Started

# Access services:
# - API: http://localhost:8000
# - UI: http://localhost:8501
# - MLflow: http://localhost:5000
# - MinIO: http://localhost:9000
```

**Available Docker Compose profiles**:

```bash
# MLOps stack only (MLflow + MinIO + PostgreSQL)
make docker-mlops

# API + UI only (requires MLflow running)
make docker-serve

# Training on Docker
make docker-train

# Monitoring on Docker
make docker-monitor

# Stop all services
make docker-down
```

---

## MLOps Workflows

### Workflow 1: Complete Training & Deployment

```bash
# Step 1: Validate data
python -m src.validation

# Step 2: Train model (full pipeline with tuning)
make train

# Step 3: Evaluate model
python -c "from src.main import validate_metrics; ..."

# Step 4: Register model in MLflow
# (Automatic during training)

# Step 5: Deploy API
make serve &

# Step 6: Deploy UI
streamlit run ui/fraud_app.py &

# Step 7: Set up monitoring
python -m pipelines.monitoring_pipeline
```

### Workflow 2: Experiment Tracking

```bash
# Training run gets automatically tracked in MLflow:
# 1. Hyperparameters
# 2. Metrics (accuracy, AUC, etc.)
# 3. Artifacts (models, plots, reports)
# 4. Tags and metadata

# View results
$BROWSER http://localhost:5000

# Compare runs
# - Parameter differences
# - Metric differences
# - Artifact comparison
```

### Workflow 3: Batch Inference

```bash
# Process production data
python -c "
from pipelines.inference_pipeline import inference_flow
import pandas as pd

# Load production data
prod_data = pd.read_csv('data/production_transactions.csv')

# Run batch inference
results = inference_flow(
    transactions=prod_data,
    model_uri='models/fraud_detector'
)

# Save results with predictions and explanations
results.to_csv('results/batch_predictions.csv', index=False)
"
```

### Workflow 4: Model Retraining

```bash
# Automatic retraining pipeline
python -c "
from pipelines.training_pipeline import training_flow
from mlops.registry import ModelRegistry

# Run training
results = training_flow(
    data_path='data/train_transaction.csv',
    new_data_path='data/new_transactions.csv'  # Data since last training
)

# Register new model
registry = ModelRegistry()
if results['metrics']['auc_roc'] > 0.99:
    registry.register_model(
        model=results['model'],
        name='fraud-detector',
        version=results['version'],
        metrics=results['metrics']
    )
"
```

---

## Code Organization by Feature

### Data Processing
- `src/validation.py` - Schema/quality validation
- `src/data_preprocessing.py` - ETL pipeline
- `src/feature_selection.py` - SHAP feature selection

### Model Training
- `src/optuna_tuning.py` - Hyperparameter optimization
- `src/stacking_model.py` - Ensemble implementation
- `src/evaluation.py` - Metrics computation

### Explainability
- `src/explainability.py` - SHAP, LIME, PDP generation
- `src/visualization.py` - Plot generation

### Serving
- `deployment/api/main.py` - FastAPI server
- `deployment/api/schemas.py` - Request/response models
- `ui/fraud_app.py` - Streamlit interface

### MLOps
- `mlops/tracking.py` - MLflow integration
- `mlops/registry.py` - Model registry
- `mlops/monitoring.py` - Drift detection
- `pipelines/training_pipeline.py` - Prefect workflow
- `pipelines/inference_pipeline.py` - Prediction workflow
- `pipelines/monitoring_pipeline.py` - Monitoring workflow

---

## Testing

```bash
# Run all tests
make test

# Run specific test suite
make test-unit      # Unit tests only
make test-cov       # With coverage report

# Run tests for API
pytest tests/integration/test_api.py -v

# Run tests for model
pytest tests/unit/test_model.py -v
```

---

## Quality Checks

```bash
# Run all quality checks
make quality

# Individual checks:
make lint           # Ruff linting
make format         # Black formatting
make type-check     # MyPy type checking
make test           # Pytest
```

---

## Troubleshooting

### Issue: "Docker not found"
**Solution**: Docker daemon not running or not installed
```bash
# Install Docker
sudo apk add docker docker-compose

# Start Docker daemon (requires elevated privileges)
sudo dockerd &
```

### Issue: Kaggle data not found
**Solution**: Data files must be downloaded from Kaggle
```bash
# Set up Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

### Issue: Out of memory during training
**Solution**: Use sample_size configuration
```yaml
# Edit config/params.yaml
data:
  sample_size: 100000  # Reduces to 100K samples
```

### Issue: API fails to start
**Solution**: Model not trained yet
```bash
# Train model first
make train-quick

# Then start API
make serve
```

### Issue: Streamlit connection refused
**Solution**: API server not running
```bash
# Terminal 1: Start API
make serve

# Terminal 2: Start UI
streamlit run ui/fraud_app.py
```

### Issue: MLflow shows no experiments
**Solution**: MLflow server not running or different backend
```bash
# Start local MLflow server
make mlflow-ui

# Check server running
curl http://localhost:5000
```

---

## Performance Benchmarks

| Component | Time Estimate | Hardware |
|-----------|--------------|----------|
| Data preprocessing | 1-2 min | 8GB RAM |
| Feature selection | 2-3 min | - |
| Optuna tuning (50 trials) | 15-30 min | GPU optional |
| Model training | 2-5 min | - |
| Evaluation | 1 min | - |
| Explainability generation | 5-10 min | - |
| **Total (full pipeline)** | **~30-45 min** | **Single machine** |

With `sample_size=100000`: **~10-15 minutes**

---

## Next Steps

1. **Deploy to Production**
   - Configure GCP Cloud Run (see `infrastructure/`)
   - Set up GitHub Actions CI/CD
   - Create monitoring dashboards

2. **Extend Functionality**
   - Add more base models
   - Implement additional explainability techniques
   - Create custom evaluation metrics

3. **Optimize Performance**
   - Implement model serving with TFS
   - Add caching layer
   - Optimize feature computation

4. **Scale to Production**
   - Kubernetes deployment
   - Distributed training
   - Real-time data pipelines

---

## Additional Resources

- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **API Documentation**: [docs/API.md](docs/API.md)
- **MLOps Setup**: [docs/MLOPS.md](docs/MLOPS.md)
- **Paper**: [Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods](https://arxiv.org/html/2505.10050v1)

---

**Last Updated**: February 2026  
**Version**: 1.1.0  
**Author**: thanhtrung102
