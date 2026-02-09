# Fraud Detection MLOps - Feature Demonstration Guide

## Welcome! 👋

You now have access to a comprehensive, production-ready fraud detection system with built-in MLOps capabilities. This guide explains how to explore all features.

---

## 📚 Documentation Structure

There are **three ways** to explore the system:

### 1. **QUICK_REFERENCE.md** - For Quick Navigation
- 5-minute demos
- Common use cases
- Copy-paste commands
- Quick troubleshooting

**Best for**: Getting started, running specific tasks, quick answers

### 2. **DEMO_WORKFLOW.md** - For Comprehensive Learning  
- Feature-by-feature walkthrough
- Detailed explanations
- Code examples
- Performance benchmarks

**Best for**: Understanding architecture, exploring features in depth, learning patterns

### 3. **DEMO_WORKFLOW.ipynb** - For Hands-On Practice
- Interactive Jupyter notebook
- Runnable code cells
- Visualizations
- Real data exploration

**Best for**: Learning by doing, experimenting, understanding concepts

---

## 🎯 Choose Your Learning Path

### Path 1: I Want to Demo Everything (30 minutes)

```bash
# 1. Read this first
cat QUICK_REFERENCE.md

# 2. Run the quick demo
make install-dev
make train-quick

# 3. Start services
make serve &
streamlit run ui/fraud_app.py &
make mlflow-ui &

# 4. Explore
# - API: http://localhost:8000/docs
# - UI: http://localhost:8501
# - MLflow: http://localhost:5000
```

### Path 2: I Want to Understand the Architecture (45 minutes)

```bash
# 1. Read architecture docs
cat DEMO_WORKFLOW.md

# 2. Understand key features
# - Section: "Individual Feature Demos"
# - Section: "MLOps Workflows"

# 3. Run specific feature demos as shown in DEMO_WORKFLOW.md

# 4. Read code
# - src/stacking_model.py (model architecture)
# - pipelines/training_pipeline.py (orchestration)
# - deployment/api/main.py (serving)
```

### Path 3: I Want to Experiment Hands-On (60 minutes)

```bash
# 1. Open Jupyter notebook
jupyter notebook DEMO_WORKFLOW.ipynb

# 2. Run through all cells
# - Load data
# - Explore statistics
# - Train models
# - Make predictions
# - Monitor performance

# 3. Modify code to experiment
# - Change hyperparameters
# - Try different thresholds
# - Analyze feature importance
```

---

## 📋 Feature Checklist

Check off features as you explore them:

### Core ML Features
- [ ] Data validation and quality checks
- [ ] Feature selection with SHAP
- [ ] Hyperparameter tuning with Optuna
- [ ] Stacking ensemble training
- [ ] Model evaluation and metrics
- [ ] Explainability (SHAP, LIME, PDP)

### Serving Features
- [ ] FastAPI REST server
- [ ] Streamlit interactive UI
- [ ] Single predictions
- [ ] Batch predictions
- [ ] Real-time explanations

### MLOps Features
- [ ] MLflow experiment tracking
- [ ] Model registry
- [ ] Prefect orchestration
- [ ] Data drift detection
- [ ] Performance monitoring

### Deployment Features
- [ ] Docker containerization
- [ ] Docker Compose orchestration
- [ ] Environment configuration
- [ ] Production settings

---

## 🚀 Quick Start Commands

### Fastest Demo (5 minutes)
```bash
make install-dev && make train-quick && cat results/metrics.json
```

### Full Feature Demo (15 minutes)
```bash
make install-dev
make train-quick
make serve &
curl http://localhost:8000/health
```

### Complete System (30 minutes)
```bash
make install-dev
make train-quick
make serve &
streamlit run ui/fraud_app.py &
make mlflow-ui &
# Open browsers to localhost:8000, 8501, 5000
```

---

## 📁 File Organization

```
fraud-detection/
├── DEMO_WORKFLOW.md          ← Read first! Comprehensive guide
├── DEMO_WORKFLOW.ipynb       ← Interactive notebook for hands-on
├── QUICK_REFERENCE.md        ← Quick lookup for commands
├── GUIDE.md                  ← This file
│
├── src/                       # Core model code
│   ├── main.py              # End-to-end pipeline
│   ├── stacking_model.py    # Ensemble implementation
│   ├── feature_selection.py # SHAP feature selection
│   ├── optuna_tuning.py     # Hyperparameter optimization
│   ├── explainability.py    # SHAP, LIME, PDP
│   └── ...
│
├── pipelines/                # MLOps workflows
│   ├── training_pipeline.py # Prefect training flow
│   ├── inference_pipeline.py # Prediction flow
│   └── monitoring_pipeline.py # Drift detection
│
├── deployment/               # Production code
│   ├── api/main.py          # FastAPI server
│   ├── docker-compose.yml   # Container orchestration
│   └── Dockerfile*          # Container definitions
│
├── ui/                       # Web Interface
│   ├── fraud_app.py         # Streamlit app
│   └── utils/               # Helper modules
│
├── mlops/                    # MLOps infrastructure
│   ├── tracking.py          # MLflow integration
│   ├── registry.py          # Model registry
│   └── monitoring.py        # Monitoring utils
│
├── config/                   # Configuration files
│   ├── params.yaml          # Model parameters
│   ├── validation.yaml      # Validation rules
│   └── ...
│
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
│
└── docs/                     # Additional documentation
    ├── ARCHITECTURE.md      # System architecture
    ├── API.md              # API documentation
    └── MLOPS.md            # MLOps setup
```

---

## 🎓 Learning Objectives

By following this guide, you'll understand:

### Machine Learning
✓ Stacking ensemble methods  
✓ Feature selection with SHAP  
✓ Class imbalance handling  
✓ Explainable AI techniques  
✓ Model evaluation strategies  

### MLOps
✓ Experiment tracking  
✓ Model versioning  
✓ Workflow orchestration  
✓ Data validation  
✓ Monitoring and drift detection  

### Production
✓ REST API design  
✓ Web UI development  
✓ Docker containerization  
✓ Multi-service deployment  
✓ Configuration management  

---

## 📊 System Architecture Overview

```
Input Data
    ↓
┌─────────────────────────┐
│  Data Validation        │ ← Feature 1
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  Preprocessing          │ ← Feature 2
│  (SMOTE, Encoding)      │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Feature Selection (SHAP)│ ← Feature 3
│ (Top 30 features)       │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Optuna Tuning           │ ← Feature 4
│ (3 models × 50 trials)  │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Train Ensemble          │ ← Feature 5
│ XGB + LGB + CB + Meta   │
└──────────┬──────────────┘
           ↓
      ┌────┴────┬────────┬─────────┐
      ↓         ↓        ↓         ↓
    API    Streamlit  MLflow   Monitoring  ← Features 6-9
    UI                         
      ↓         ↓        ↓         ↓
  Predictions  Analysis  Tracking  Drift
```

---

## 🔄 Common Use Cases

| Use Case | Time | Command/Guide |
|----------|------|-----------|
| Train model | 15-30 min | `make train` |
| Quick demo | 5 min | `make train-quick` |
| Make prediction | 1 min | See API section in QUICK_REFERENCE.md |
| Run tests | 5 min | `make test` |
| Code quality checks | 2 min | `make quality` |
| View MLflow | 1 min | `make mlflow-ui` |
| Deploy API | 1 min | `make serve` |
| Run UI | 1 min | `streamlit run ui/fraud_app.py` |
| Full stack | 3 min | `make docker-full` |

---

## ✅ Success Indicators

Your setup is successful when:

- [ ] `make train-quick` completes without errors
- [ ] `results/metrics.json` shows AUC > 0.98
- [ ] `make serve` server responds to `curl http://localhost:8000/health`
- [ ] Streamlit UI loads at `http://localhost:8501`
- [ ] MLflow shows experiments at `http://localhost:5000`
- [ ] Can make predictions via API and UI

---

## 🐛 Troubleshooting

### Problem: "Docker not found"
→ See QUICK_REFERENCE.md, Debugging section

### Problem: Data files missing
→ See DEMO_WORKFLOW.md, Prerequisites section

### Problem: Out of memory
→ Edit `config/params.yaml`, set `sample_size: 50000`

### Problem: Port already in use
→ Change port in respective start command

### Problem: Model not found
→ Run `make train` or `make train-quick` first

**More help**: See DEMO_WORKFLOW.md Troubleshooting section

---

## 📞 Need Help?

1. **For quick answers**: Check QUICK_REFERENCE.md
2. **For detailed guides**: Read DEMO_WORKFLOW.md
3. **For code examples**: See DEMO_WORKFLOW.ipynb
4. **For architecture details**: Read docs/ARCHITECTURE.md
5. **For API specifics**: See docs/API.md
6. **For MLOps setup**: See docs/MLOPS.md

---

## 🎯 Next Steps

1. **Start Here**: Run `make train-quick` (5 minutes)
2. **Explore**: Read DEMO_WORKFLOW.md sections 1-3
3. **Experiment**: Run DEMO_WORKFLOW.ipynb notebook
4. **Deploy**: Follow "MLOps Workflows" in DEMO_WORKFLOW.md
5. **Extend**: Modify code for your use case

---

## 📈 What You'll Build

By the end, you'll have:

✓ Trained fraud detection ensemble model (99%+ accuracy)  
✓ Explainability reports (SHAP, LIME)  
✓ REST API for predictions  
✓ Interactive web UI  
✓ MLflow experiment tracking  
✓ Data quality monitoring  
✓ Production deployment setup  
✓ Docker containerization  

---

## 💡 Key Insights

- **Stacking Ensemble**: Combines 3 models (XGBoost, LightGBM, CatBoost) with XGBoost meta-learner
- **Feature Selection**: SHAP-based selection reduces 400+ features to 30 key features
- **Explainability**: SHAP Force plots explain individual predictions
- **Monitoring**: Automatic drift detection for production data
- **MLOps**: Full tracking with MLflow and orchestration with Prefect

---

## 📝 Citation

If you use this system, please cite:

```
@article{fraud_detection_mlops,
  title={Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods},
  author={thanhtrung102},
  year={2026},
  url={https://github.com/thanhtrung102/fraud-detection}
}
```

---

## 📞 Support

- **Repository**: https://github.com/thanhtrung102/fraud-detection
- **Issues**: https://github.com/thanhtrung102/fraud-detection/issues
- **Paper**: https://arxiv.org/html/2505.10050v1

---

**Ready to get started?**

👉 [Read QUICK_REFERENCE.md](QUICK_REFERENCE.md)  
👉 [Read DEMO_WORKFLOW.md](DEMO_WORKFLOW.md)  
👉 [Open DEMO_WORKFLOW.ipynb](DEMO_WORKFLOW.ipynb)  

---

**Last Updated**: February 2026  
**Version**: 1.1.0
