# Credit Card Fraud Detection with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Kaggle-IEEE--CIS%20Fraud-20BEFF.svg)](https://www.kaggle.com/c/ieee-fraud-detection)

A fraud detection system using stacking ensemble methods combined with explainable AI techniques. Based on the paper ["Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods"](https://arxiv.org/html/2505.10050v1) (arXiv:2505.10050).

> **Looking for MLOps & Production Deployment?**
> Check out the [`mlops` branch](https://github.com/thanhtrung102/fraud-detection/tree/mlops) for:
> - MLflow experiment tracking & model registry
> - Prefect workflow orchestration
> - FastAPI model serving
> - Evidently monitoring & drift detection
> - Docker & GCP Cloud Run deployment
> - CI/CD with GitHub Actions
>
> See the [MLOps Documentation](https://github.com/thanhtrung102/fraud-detection/blob/mlops/docs/MLOPS.md) for details.

## Highlights

- **Stacking Ensemble**: Combines XGBoost, LightGBM, and CatBoost with XGBoost meta-learner
- **Explainable AI**: Full transparency with SHAP, LIME, and Partial Dependence Plots
- **Optuna Tuning**: Automated hyperparameter optimization (20 trials per model)
- **SHAP Feature Selection**: Top 30 features selected based on SHAP importance
- **Achieved**: ~98% accuracy, ~92% AUC-ROC on 100K sample (Codespaces)
- **Paper Target**: 99% accuracy, 0.99 AUC-ROC on full 590K dataset

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [MLOps & Production](#mlops--production)
- [Contributing](#contributing)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/thanhtrung102/fraud-detection.git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (with SHAP feature selection + Optuna tuning)
python -m src.main
```

### Pipeline Options

```bash
# Full pipeline (SHAP + Optuna) - Recommended for best results
python -m src.main

# Skip Optuna tuning (faster, uses config defaults)
python -m src.main --no-optuna

# Skip SHAP feature selection (use all features)
python -m src.main --no-feature-selection

# Quick run (skip both)
python -m src.main --no-optuna --no-feature-selection
```

### Production Mode (Replicate Paper Results)

For machines with **4 cores, 16GB RAM, 32GB storage**:

```bash
# Copy production config (300K samples, optimized for paper results)
cp config/params_production.yaml config/params.yaml

# Run full pipeline with Optuna tuning (20 trials)
python -m src.main
```

**Target Metrics:** 99% Accuracy, 0.99 AUC-ROC (as per [arXiv:2505.10050](https://arxiv.org/html/2505.10050v1))

### GitHub Codespaces (8GB RAM)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/thanhtrung102/fraud-detection)

For Codespaces (limited to 8GB RAM):

```bash
# Copy Codespaces config (100K samples)
cp config/params_codespaces.yaml config/params.yaml

# Run pipeline (skip Optuna for faster execution)
python -m src.main --no-optuna
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Data                                │
│              (590K+ transactions, 400+ features)                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessing                                 │
│  • Missing value imputation (median/mode)                        │
│  • Categorical encoding (Label Encoding)                         │
│  • Class balancing (SMOTE)                                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Selection                               │
│           SHAP-based top 30 feature selection                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Optuna Hyperparameter Tuning                        │
│     20 trials per model, 5-fold CV, optimizing AUC-ROC          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Stacking Ensemble                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│  │  XGBoost  │  │ LightGBM  │  │ CatBoost  │   Base Learners    │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                    │
│        │              │              │                           │
│        └──────────────┼──────────────┘                           │
│                       ▼                                          │
│               ┌───────────────┐                                  │
│               │    XGBoost    │  Meta-Learner                    │
│               │  Meta-Learner │                                  │
│               └───────────────┘                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Success Metrics Validation                          │
│         Validate against paper targets (99% accuracy)            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Explainability                                 │
│         SHAP │ LIME │ Partial Dependence Plots                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.8+
- 16GB+ RAM (recommended)
- ~2GB storage for dataset

### Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt
```

### Dataset

Download the IEEE-CIS Fraud Detection dataset from Kaggle:

```bash
# Configure Kaggle API first (place kaggle.json in ~/.kaggle/)
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

**Dataset Statistics:**
- 590,540 transactions
- 3.5% fraud rate (highly imbalanced)
- 394 transaction features + 41 identity features

---

## Usage

### Full Pipeline (Recommended)

```bash
# Run the complete pipeline with all optimizations
python -m src.main
```

The pipeline runs 6 phases:
1. **Data Preprocessing**: Load, impute, encode, split, and SMOTE balance
2. **SHAP Feature Selection**: Select top 30 most important features
3. **Optuna Tuning**: 20 trials per model optimizing AUC-ROC
4. **Model Training**: Train stacking ensemble with tuned hyperparameters
5. **Prediction**: Generate predictions with optimized threshold
6. **Evaluation**: Compute metrics and validate against paper targets

### Programmatic Usage

```python
from src.data_preprocessing import preprocess_pipeline
from src.feature_selection import shap_feature_selection, apply_feature_selection
from src.optuna_tuning import tune_all_models
from src.stacking_model import StackingFraudDetector
from src.evaluation import compute_metrics, print_results

# Load and preprocess data
X_train, X_test, y_train, y_test, features = preprocess_pipeline()

# SHAP Feature Selection (top 30 features)
indices, selected_features, importance = shap_feature_selection(
    X_train, y_train, features, n_top_features=30
)
X_train_selected = apply_feature_selection(X_train, indices)
X_test_selected = apply_feature_selection(X_test, indices)

# Optuna Hyperparameter Tuning (20 trials)
best_params = tune_all_models(X_train_selected, y_train, n_trials=20)

# Train model with optimized parameters
model = StackingFraudDetector(
    xgb_params=best_params['xgboost'],
    lgbm_params=best_params['lightgbm'],
    catboost_params=best_params['catboost']
)
model.fit(X_train_selected, y_train)

# Evaluate
y_proba = model.predict_proba(X_test_selected)[:, 1]
y_pred = model.predict(X_test_selected, threshold=0.44)

metrics = compute_metrics(y_test, y_pred, y_proba)
print_results(metrics)
```

### Hyperparameter Tuning Only

```python
from src.optuna_tuning import tune_xgboost, tune_lightgbm, tune_catboost

# Tune individual models
xgb_params = tune_xgboost(X_train, y_train, n_trials=20)
lgbm_params = tune_lightgbm(X_train, y_train, n_trials=20)
catboost_params = tune_catboost(X_train, y_train, n_trials=20)
```

### Feature Selection Only

```python
from src.feature_selection import shap_feature_selection

# Get SHAP-based feature importance
indices, names, importance_df = shap_feature_selection(
    X_train, y_train, feature_names, n_top_features=30
)

# View top features
print(importance_df.head(10))
```

---

## Results

### Achieved Performance (100K Sample - Codespaces)

| Metric | Achieved | Paper Target | Status |
|--------|----------|--------------|--------|
| **Accuracy** | 0.9789 | 0.99 | 98.9% of target |
| **Precision** | 0.7888 | 0.99 | Below target |
| **Recall** | 0.5426 | 0.99 | Below target |
| **F1-Score** | 0.6430 | 0.99 | Below target |
| **AUC-ROC** | 0.9195 | 0.99 | 92.9% of target |

### Optuna Cross-Validation Scores

During hyperparameter tuning (on SMOTE-balanced training data):
```
XGBoost CV AUC:  0.9969
LightGBM CV AUC: 0.9973
CatBoost CV AUC: 0.9970
```

### Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|---|:-------------------:|:---------------:|
| **Actual Legitimate** | 57,596 | 305 |
| **Actual Fraud** | 960 | 1,139 |

- **True Negative Rate**: 99.5% (legitimate correctly classified)
- **Fraud Detection Rate**: 54.3% (recall)
- **Threshold Used**: 0.67 (F1-optimized)

### Top Features by SHAP Importance

1. C14 (transaction category): 0.4719
2. C12: 0.4405
3. card6: 0.4312
4. C1: 0.4239
5. V308: 0.4019

### Notes on Results Gap

The gap between CV scores (~0.997) and test results (~0.92 AUC) is due to:
- **Distribution mismatch**: Training on SMOTE-balanced data (50/50), testing on imbalanced data (3.5% fraud)
- **Sample size**: Using 100K samples vs paper's 590K full dataset
- **Threshold selection**: F1-optimized threshold (0.67) prioritizes precision over recall

For better recall, consider using the paper's threshold of 0.44 instead of F1-optimization.

---

## Explainability

### SHAP Analysis

SHAP (SHapley Additive exPlanations) provides global and local feature importance:

```python
import shap
from src.explainability import plot_shap_summary

# Generate SHAP summary plot
plot_shap_summary(shap_values, X_test, feature_names,
                  save_path='results/shap_summary.png')
```

**Top Features by SHAP Importance:**
1. C14 (transaction category)
2. TransactionAmt
3. card1
4. addr1
5. D15

### LIME Explanations

Local Interpretable Model-agnostic Explanations for individual predictions:

```python
from src.explainability import explain_single_prediction_lime

# Explain a specific fraud prediction
explain_single_prediction_lime(
    model, X_train, X_test[fraud_idx], feature_names,
    save_path='results/lime_explanation.html'
)
```

### Partial Dependence Plots

Visualize feature effects on predictions:

```python
from src.explainability import plot_partial_dependence

plot_partial_dependence(model, X_test, feature_names,
                       features_to_plot=['C14', 'TransactionAmt', 'card1', 'D15'],
                       save_path='results/pdp.png')
```

---

## Project Structure

```
fraud-detection/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── config/
│   ├── params.yaml            # Default configuration
│   ├── params_production.yaml # Production config (16GB RAM)
│   └── params_codespaces.yaml # GitHub Codespaces config (8GB RAM)
│
├── data/                      # Dataset (download from Kaggle)
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── main.py                # Main pipeline entry point
│   ├── data_preprocessing.py  # Data loading & preprocessing
│   ├── feature_selection.py   # SHAP-based feature selection
│   ├── optuna_tuning.py       # Hyperparameter optimization
│   ├── stacking_model.py      # Stacking ensemble implementation
│   ├── evaluation.py          # Metrics & visualization
│   └── explainability.py      # SHAP, LIME, PDP analysis
│
├── notebooks/
│   └── exploration.ipynb      # Data exploration notebook
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── lime_explanation.html
│
├── models/                    # Saved model files
│   ├── xgb_model.joblib
│   ├── lgbm_model.joblib
│   ├── catboost_model.joblib
│   └── meta_learner.joblib
│
└── .devcontainer/
    └── devcontainer.json      # GitHub Codespaces configuration
```

---

## Configuration

All hyperparameters are managed in `config/params.yaml`:

```yaml
# Data Configuration
data:
  sample_size: 300000  # Adjust based on available RAM

# SHAP Feature Selection (as per paper)
feature_selection:
  method: shap
  n_top_features: 30
  shap_sample_size: 50000

# Optuna Hyperparameter Tuning (as per paper - 20 trials)
optuna:
  n_trials: 20
  direction: maximize
  metric: roc_auc
  cv_folds: 5

# Base Models (defaults, can be overridden by Optuna)
base_models:
  xgboost:
    n_estimators: 400
    max_depth: 8
    learning_rate: 0.1

  lightgbm:
    n_estimators: 400
    max_depth: 8
    learning_rate: 0.1

  catboost:
    iterations: 400
    depth: 8
    learning_rate: 0.1

# Meta-Learner (XGBoost as per paper)
meta_learner:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.1

# Classification Threshold (paper found 0.44 optimal)
threshold:
  default: 0.44
```

---

## Key Techniques

### 1. SMOTE for Class Imbalance

With only 3.5% fraud cases, we use SMOTE to balance the training data:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Note:** SMOTE is applied only to training data, never test data.

### 2. Stacking Ensemble

Three diverse gradient boosting models combined via meta-learning:

- **XGBoost**: Efficient on large datasets, strong regularization
- **LightGBM**: Fast training, handles many features well
- **CatBoost**: Excellent with categorical features

### 3. Threshold Optimization

Default 0.5 threshold is suboptimal for imbalanced data. We optimize for F1-score:

```python
best_threshold = 0.44  # Optimized via grid search
```

---

## MLOps & Production

This branch (`main`) contains the core ML experiment code. For production deployment and MLOps features, switch to the [`mlops` branch](https://github.com/thanhtrung102/fraud-detection/tree/mlops).

### MLOps Branch Features

| Feature | Description |
|---------|-------------|
| **MLflow** | Experiment tracking, model registry |
| **Prefect** | Workflow orchestration, scheduled pipelines |
| **FastAPI** | REST API for model serving |
| **Evidently** | Data drift detection, model monitoring |
| **Docker** | Containerized deployment |
| **GCP Cloud Run** | Serverless production deployment |
| **Terraform** | Infrastructure as Code |
| **GitHub Actions** | CI/CD pipelines |

### Quick Start (MLOps)

```bash
# Switch to mlops branch
git checkout mlops

# Run training with MLflow tracking
python pipelines/training_pipeline.py --config-path config/params_codespaces.yaml

# Start API server
uvicorn api.main:app --reload --port 8000
```

See [MLOps Documentation](https://github.com/thanhtrung102/fraud-detection/blob/mlops/docs/MLOPS.md) for complete setup guide.

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
