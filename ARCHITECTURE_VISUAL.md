# Fraud Detection Workflow - Visual Guide

## System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Raw Fraud Data                              │
│                    (590K+ transactions + identity)                         │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │   DATA VALIDATION & QUALITY CHECKS      │
        │  (Schema, missing values, outliers)    │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │    DATA PREPROCESSING PIPELINE          │
        │  (Imputation, encoding, scaling)       │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │      CLASS BALANCING (SMOTE)            │
        │   (Address imbalanced dataset)         │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  FEATURE SELECTION (SHAP)               │
        │  (400+ features → 30 key features)     │
        └────────────────────┬───────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
   │  TUNE       │   │  TUNE       │   │  TUNE        │
   │  XGBoost    │   │ LightGBM    │   │ CatBoost     │
   │(50 trials)  │   │(50 trials)  │   │(50 trials)   │
   └──────┬──────┘   └──────┬──────┘   └───────┬──────┘
          │                 │                  │
          ▼                 ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
   │   TRAIN     │   │   TRAIN     │   │  TRAIN       │
   │  XGBoost    │   │ LightGBM    │   │ CatBoost     │
   └──────┬──────┘   └──────┬──────┘   └───────┬──────┘
          │                 │                  │
          └─────────────────┼──────────────────┘
                            │
                            ▼
            ┌──────────────────────────────┐
            │  CREATE META-FEATURES         │
            │(Predictions from 3 models)   │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   TRAIN META-LEARNER         │
            │    (XGBoost on meta-feat.)  │
            └──────────────┬───────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
    ┌────────┐         ┌────────┐         ┌─────────┐
    │ MODEL  │         │ EXPLAIN│         │ MONITOR │
    │ EVAL   │         │ (SHAP) │         │ (DRIFT) │
    └────┬───┘         └────┬───┘         └────┬────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  DEPLOY  │      │   VIEW   │      │   TRACK  │
    │  API     │      │   UI     │      │  MLflow  │
    │ (FastAPI)│      │(Streamlit)      │          │
    └──────────┘      └──────────┘      └──────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                ┌───────────┴────────────┐
                │                        │
                ▼                        ▼
        ┌───────────────┐      ┌──────────────────┐
        │  PREDICTIONS  │      │  MONITORING &    │
        │  (Real-time)  │      │  RETRAINING      │
        └───────────────┘      └──────────────────┘
```

---

## Feature Implementation Timeline

```
Step 1: Data Loading & Validation (2 min)
├─ Load train/test data
├─ Validate schema
├─ Check data quality
└─ Generate validation report

Step 2: Preprocessing (1 min)
├─ Handle missing values
├─ Encode categorical features
├─ Apply SMOTE for class balance
└─ Prepare training set

Step 3: Feature Selection (2-3 min)
├─ Train initial XGBoost model
├─ Calculate SHAP importance
├─ Select top 30 features
└─ Reduce feature space

Step 4: Hyperparameter Tuning (15-30 min)
├─ XGBoost: 50 Optuna trials
├─ LightGBM: 50 Optuna trials
├─ CatBoost: 50 Optuna trials
└─ Save best parameters

Step 5: Model Training (2-5 min)
├─ Train XGBoost with best params
├─ Train LightGBM with best params
├─ Train CatBoost with best params
└─ Prepare base model predictions

Step 6: Ensemble Creation (1 min)
├─ Create meta-features
├─ Train meta-learner (XGBoost)
└─ Validate ensemble performance

Step 7: Evaluation (1 min)
├─ Compute metrics (AUC, ACC, F1)
├─ Generate confusion matrices
├─ Create ROC curves
└─ Generate performance plots

Step 8: Explainability (5-10 min)
├─ SHAP summary plots
├─ SHAP force plots
├─ LIME explanations
└─ Partial Dependence Plots

Step 9: Model Storage (< 1 min)
├─ Save model artifacts
├─ Save metadata
├─ Register in MLflow
└─ Save visualizations

Step 10: Serve & Monitor (ongoing)
├─ Start FastAPI server
├─ Launch Streamlit UI
├─ Monitor performance
└─ Detect data drift
```

---

## User Journey Through Features

### 👤 Data Scientist

```
START: Install && Configure
  ↓
EXPLORE: Run DEMO_WORKFLOW.md
  ├─ Load data
  ├─ Validate quality
  ├─ Explore statistics
  ├─ Train models
  ├─ Evaluate results
  └─ Generate explanations
  ↓
EXPERIMENT: Modify DEMO_WORKFLOW.ipynb
  ├─ Adjust hyperparameters
  ├─ Change feature count
  ├─ Try different models
  └─ Compare results
  ↓
OPTIMIZE: Run full training with Optuna
  ├─ 50 trials per model
  ├─ Find optimal parameters
  ├─ Compare benchmarks
  └─ Register in MLflow
  ↓
MONITOR: Check drift detection
  ├─ Compare distributions
  ├─ Track performance metrics
  ├─ Retrain if needed
  └─ Deploy new version
```

### 🔧 ML Engineer / DevOps

```
START: Review architecture docs
  ├─ Read DEMO_WORKFLOW.md
  └─ Understand components
  ↓
SETUP: Configure deployment
  ├─ Edit config files
  ├─ Set environment variables
  └─ Prepare infrastructure
  ↓
DEPLOY: Use Docker stack
  ├─ make docker-full
  ├─ Start all services
  ├─ Verify endpoints
  └─ Test API
  ↓
ORCHESTRATE: Setup Prefect flows
  ├─ Deploy training pipeline
  ├─ Deploy inference pipeline
  ├─ Deploy monitoring pipeline
  └─ Schedule runs
  ↓
MONITOR: Track system health
  ├─ Check MLflow experiments
  ├─ Monitor data drift
  ├─ Review error logs
  └─ Maintain SLAs
```

### 📊 Business Analyst

```
START: Access Streamlit UI
  ├─ Connect to fraud_app.py
  └─ Load interface
  ↓
UPLOAD: Process transaction data
  ├─ Upload CSV
  └─ Validate format
  ↓
ANALYZE: View predictions
  ├─ See fraud probability
  ├─ Review explanations
  ├─ Understand risk factors
  └─ Export results
  ↓
REPORT: Generate insights
  ├─ Feature importance
  ├─ Model performance
  ├─ Fraud patterns
  └─ Cost-benefit analysis
  ↓
DECIDE: Take action
  ├─ Block high-risk transactions
  ├─ Investigate medium-risk
  ├─ Monitor trends
  └─ Update rules
```

---

## Training Time Estimates

```
Configuration          | Time    | Hardware     | Features
─────────────────────┼─────────┼──────────────┼──────────
Quick Demo           | 5 min   | 8GB RAM      | Basic ML
(sample_size=50K)    |         |              | No tuning
─────────────────────┼─────────┼──────────────┼──────────
Standard Training    | 15 min  | 8GB RAM      | Full pipeline
(sample_size=100K)   |         |              | No tuning
─────────────────────┼─────────┼──────────────┼──────────
Production Training  | 30-45m  | 16GB RAM     | Full features
(full dataset)       |         | GPU optional | Optuna tuning
─────────────────────┼─────────┼──────────────┼──────────
With Hypertuning     | 45-60m  | 16GB RAM     | 150 trials
(full dataset)       |         | GPU optional | Optimization
```

---

## Decision Tree - Which Feature to Demo

```
                    ╔═ Start Here ═╗
                    │              │
                    └──────┬───────┘
                           │
                    ┌──────▼──────┐
                    │ Have 5 min? │
                    └──┬──────┬───┘
                      └─Yes─┐ └─No─┐
                            │      │
                      Run   │      └─────────────┐
                      quick │                   │
                      train │        ┌──────────▼──────────┐
                            │        │ Have 15-30 min?     │
                            │        └───┬───────┬─────────┘
                            │          Yes│       └─No─┐
                            │            │             │
                            └─┐      ┌───▼──┐    ┌─────▼────────┐
                              │      │ Read │    │ Read full    │
                              │      │QUICK │    │DEMO_WORKFLOW │
                              │      │REF   │    │              │
                              │      └──┬───┘    └──────────────┘
                              │         │
                              └────┬────┘
                                   │
                            ┌──────▼──────┐
                            │ Try coding? │
                            └───┬──────┬─┘
                              Yes│      └─No─┐
                               │            │
                        ┌──────▼────┐   ┌───▼────────────┐
                        │ Run       │   │ Explore API at │
                        │IPYNB      │   │ localhost:8000 │
                        │notebook   │   │                │
                        └───────────┘   └────────────────┘
```

---

## API Endpoints Overview

```
┌─────────────────────────────────────────────────────────┐
│              FRAUD DETECTION API                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  GET /health                                           │
│  └─ Health check & service status                      │
│                                                         │
│  GET /models                                           │
│  └─ List available models & versions                   │
│                                                         │
│  POST /predict                                         │
│  └─ Single prediction                                  │
│     ├─ Input: Transaction features                     │
│     ├─ Output: Fraud probability (0-1)                 │
│     └─ Response time: <100ms                           │
│                                                         │
│  POST /predictions                                     │
│  └─ Batch predictions                                  │
│     ├─ Input: Array of transactions                    │
│     ├─ Output: Array of probabilities                  │
│     └─ Supports 1K+ transactions                       │
│                                                         │
│  GET /feature-names                                    │
│  └─ Get list of 30 selected features                   │
│                                                         │
│  POST /explain                                         │
│  └─ Get explanation for prediction                     │
│     ├─ SHAP force plot                                 │
│     ├─ Top risk factors                                │
│     └─ Feature contributions                           │
│                                                         │
│  GET /metrics                                          │
│  └─ Current model performance                          │
│     ├─ Accuracy, AUC, Precision, Recall, F1            │
│     └─ Test set statistics                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Dashboard Views

### MLFlow Dashboard
```
/experiments
├─ Training_2026_02_09    [50 runs]
├─ Hypertuning_XGBoost    [50 runs]
├─ Hypertuning_LightGBM   [50 runs]
└─ Hypertuning_CatBoost   [50 runs]

View: Parameters, Metrics, Artifacts, Model Registry
```

### Streamlit Dashboard
```
┌─ Fraud Detection System ─────────────┐
│                                      │
│ [Sidebar]            [Main Content]  │
│ ┌────────────┐    [Upload CSV]       │
│ │ Model Cfg  │    [Manual Entry]     │
│ ├────────────┤    [Sample Data]      │
│ │ Single Pred│    [Batch Process]    │
│ │ Batch Pred │    [Explanations]     │
│ │ Analytics  │    [Performance]      │
│ │ Monitor    │    [Feature Import]   │
│ └────────────┘    [Export Results]   │
│                                      │
└──────────────────────────────────────┘
```

---

## Monitoring Dashboard

```
Real-Time Monitoring
├─ Model Health
│  ├─ Accuracy: 0.9912 ✓
│  ├─ AUC: 0.9924 ✓
│  ├─ Recall: 0.9834 ✓
│  └─ Precision: 0.9876 ✓
│
├─ Prediction Stats
│  ├─ Total predictions: 590,540
│  ├─ Fraud rate: 3.5%
│  ├─ Avg confidence: 0.85
│  └─ Std confidence: 0.18
│
└─ Data Drift Detection
   ├─ Feature drift: 0 detected ✓
   ├─ Distribution shift: None ✓
   ├─ Missing values: 0.0% ✓
   └─ Outliers: 0.1% ✓
```

---

## Performance Checklist

- [ ] Data validation: 0 schema errors
- [ ] Feature selection: 30 features selected
- [ ] Model training: XGBoost + LightGBM + CatBoost trained
- [ ] Ensemble accuracy: > 0.99
- [ ] Ensemble AUC: > 0.99
- [ ] API response time: < 100ms
- [ ] Memory usage: < 2GB (with sample_size=100K)
- [ ] Training time: < 30 minutes

---

**Navigate to:**
- **Getting Started**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Full Guide**: [DEMO_WORKFLOW.md](DEMO_WORKFLOW.md)
- **Interactive Demo**: [DEMO_WORKFLOW.ipynb](DEMO_WORKFLOW.ipynb)
