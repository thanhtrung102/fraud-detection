# Quick Reference - Common Workflows

## 🚀 Start Here

### 5-Minute Demo
```bash
# Install and run quick training
make install-dev
make train-quick  # No hyperparameter tuning for speed

# View metrics
cat results/metrics.json
```

### 15-Minute Demo
```bash
# Install dependencies
make install-dev

# Run full training with tuning
make train  # ~15 minutes with sample_size=100000

# Start API
make serve &

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```

### 30-Minute Complete Demo
```bash
# 1. Install & Setup (2 min)
make install-dev

# 2. Train Model (15 min)
make train

# 3. Start Services (3 min)
make serve &
make mlflow-ui &
streamlit run ui/fraud_app.py &

# 4. Explore Features (10 min)
# - Visit http://localhost:8000/docs
# - Visit http://localhost:8501
# - Visit http://localhost:5000
```

---

## 📋 Common Use Cases

### Use Case 1: Check Model Performance
```bash
cat results/metrics.json | python -m json.tool
```

### Use Case 2: Make a Prediction
```bash
# Option 1: Via API (requires server running)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 100.0,
    "transaction_dt": 86400,
    ...
  }'

# Option 2: Programmatically
python -c "
from src.stacking_model import StackingFraudDetector
model = StackingFraudDetector()
model.load('models')
pred = model.predict([[...]])  # feature array
"
```

### Use Case 3: Understand a Prediction
```bash
# Generate SHAP explanation
python -c "
from src.explainability import explain_prediction
from src.stacking_model import StackingFraudDetector

model = StackingFraudDetector()
model.load('models')

# Explain why transaction was flagged
explanation = explain_prediction(model, transaction_features)
print(f'Risk Score: {explanation[\"score\"]}')
print(f'Top risk factors: {explanation[\"top_factors\"]}')
"
```

### Use Case 4: Batch Predictions
```bash
# Using Python
python -c "
from src.stacking_model import StackingFraudDetector
import pandas as pd

model = StackingFraudDetector()
model.load('models')

data = pd.read_csv('transactions.csv')
predictions = model.predict(data)
"

# Using API
curl -X POST http://localhost:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [...],
    "return_explanations": true
  }'
```

### Use Case 5: Monitor Data Quality
```bash
# Check if new data has drifted from training data
python -c "
from src.validation import DataValidator
from src.data_preprocessing import load_config

validator = DataValidator()
config = load_config()

# Validate new production data
is_valid, errors = validator.validate_for_inference(
    new_data,
    reference_data=training_data
)

if not is_valid:
    print(f'⚠️  Data issues detected: {errors}')
"
```

### Use Case 6: Retrain Model
```bash
# Full retraining with hyperparameter tuning
make train

# Quick retraining without tuning
make train-quick

# Retraining via Prefect orchestration
python -c "
from pipelines.training_pipeline import training_flow

results = training_flow(
    new_training_data_path='data/updated_train_transaction.csv'
)
print(f'New model AUC: {results[\"metrics\"][\"auc_roc\"]}')
"
```

---

## 🔧 Maintenance Tasks

### Task 1: Run Tests
```bash
# All tests
make test

# Specific test file
pytest tests/unit/test_model.py -v

# With coverage
make test-cov
```

### Task 2: Format Code
```bash
# Format code files
make format

# Check formatting without changes
make format-check
```

### Task 3: Lint Code
```bash
# Check for linting issues
make lint

# Fix linting issues automatically
make lint-fix
```

### Task 4: Type Checking
```bash
# Check type hints
make type-check
```

### Task 5: Full Quality Check
```bash
# Run all quality checks
make quality
```

---

## 🐳 Docker Commands

### Build Images
```bash
# Build all Docker images
make docker-build
```

### Run Services

```bash
# Full stack (all services)
make docker-full

# MLOps infrastructure only
make docker-mlops

# API + UI only (requires MLflow)
make docker-serve

# Stop all services
make docker-down
```

### Check Service Status
```bash
# List running containers
docker ps

# View logs
docker logs <container_name>

# Access service shell
docker exec -it <container_name> /bin/sh
```

---

## 📊 View Results

### Metrics
```bash
# View as JSON
cat results/metrics.json

# View as formatted table
python -c "
import json
with open('results/metrics.json') as f:
    metrics = json.load(f)
    for key, val in metrics.items():
        if key != 'models':
            print(f'{key:15} {val:.4f}' if isinstance(val, float) else f'{key:15} {val}')
"
```

### Visualizations
```bash
# List all generated plots
ls -lh results/figures/

# View specific plot
$BROWSER results/figures/roc_curve.png
$BROWSER results/figures/confusion_matrix.png
$BROWSER results/figures/shap_summary.png
```

### Feature Importance
```bash
# View top features
cat data/top_30_features.txt | head -10

# Full feature importance
cat results/feature_importance.csv
```

---

## 🔍 Debugging

### Issue: Training fails
```bash
# 1. Check data files exist
ls -la data/

# 2. Validate data
python -c "
from src.validation import validate_training_data
from src.data_preprocessing import load_config
validate_training_data(load_config())
"

# 3. Check config
cat config/params.yaml

# 4. See full error
python src/main.py -v  # Verbose mode
```

### Issue: API doesn't start
```bash
# 1. Check model exists
ls -la models/

# 2. Try loading model manually
python -c "
from src.stacking_model import StackingFraudDetector
m = StackingFraudDetector()
m.load('models')
"

# 3. Check port 8000 not in use
netstat -tulpn | grep 8000

# 4. Try different port
python deployment/api/main.py --port 8001
```

### Issue: Streamlit won't connect to API
```bash
# 1. Ensure API is running
curl http://localhost:8000/health

# 2. Check API URL in Streamlit settings
# (Should be http://localhost:8000)

# 3. Restart both services
make serve &
streamlit run ui/fraud_app.py --server.port 8501
```

---

## 💡 Performance Tips

### For Faster Training
```yaml
# Edit config/params.yaml
data:
  sample_size: 50000  # Smaller sample
  
optuna:
  n_trials: 10  # Fewer tuning trials (default 50)
```

### For Better Accuracy
```yaml
# Edit config/params.yaml
optuna:
  n_trials: 100  # More tuning trials
  
feature_selection:
  n_top_features: 50  # More features
```

### For Production
```yaml
# Use validated params in config/params_production.yaml
# Ensure reproducibility with fixed random_state
```

---

## 📚 Manual Feature Demos

### 1. Feature Selection
```bash
python -c "
from src.feature_selection import shap_feature_selection
from src.data_preprocessing import preprocess_pipeline, load_config
from xgboost import XGBClassifier

config = load_config()
X_train, X_test, y_train, y_test, features = preprocess_pipeline(config)
xgb = XGBClassifier(n_estimators=50)
xgb.fit(X_train, y_train)
selected, indices = shap_feature_selection(xgb, X_train, features, n_features=30)
print('Top features:', selected[:5])
"
```

### 2. Model Evaluation
```bash
python -c "
from src.stacking_model import StackingFraudDetector
from src.evaluation import compute_metrics
from src.data_preprocessing import preprocess_pipeline, load_config

config = load_config()
X_train, X_test, y_train, y_test, _ = preprocess_pipeline(config)
model = StackingFraudDetector()
model.load('models')
y_pred = model.predict(X_test)
metrics = compute_metrics(y_test, y_pred, model.predict_proba(X_test))
print(f'AUC: {metrics[\"auc_roc\"]:.4f}')
"
```

### 3. Data Validation
```bash
python -c "
from src.validation import DataValidator
from src.data_preprocessing import load_config, load_data

validator = DataValidator()
config = load_config()
df = load_data(
    config['data']['train_transaction'],
    config['data']['train_identity']
)
is_valid, errors = validator.validate_schema(df)
print(f'Valid: {is_valid}, Errors: {len(errors)}')
"
```

---

## 📞 Getting Help

- **Documentation**: [DEMO_WORKFLOW.md](DEMO_WORKFLOW.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **API Docs**: [docs/API.md](docs/API.md)
- **MLOps Info**: [docs/MLOPS.md](docs/MLOPS.md)
- **Issues**: [GitHub Issues](https://github.com/thanhtrung102/fraud-detection/issues)

---

**Pro Tip**: Start with `make train-quick` to see all features in action in ~5-10 minutes!
