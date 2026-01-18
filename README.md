# Reproducing: Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods

[![Paper](https://img.shields.io/badge/arXiv-2505.10050-b31b1b.svg)](https://arxiv.org/abs/2505.10050)
[![Dataset](https://img.shields.io/badge/Kaggle-IEEE--CIS%20Fraud-20BEFF.svg)](https://www.kaggle.com/c/ieee-fraud-detection)

## Paper Overview

**Title:** Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods
**Authors:** Fahad Almalki, Mehedi Masud (Taif University)
**Published:** May 2025
**Source:** [arXiv:2505.10050v1](https://arxiv.org/html/2505.10050v1)

### Key Claims to Reproduce
- **99% Accuracy** on IEEE-CIS Fraud Detection dataset
- **0.99 AUC-ROC** score
- Cross-validation AUC scores: [0.9979, 0.9979, 0.9979, 0.9981, 0.9980]
- Training time: ~278 seconds on full dataset

---

## Reproduction Plan

### Phase 1: Environment Setup

#### 1.1 Required Dependencies
```bash
# Create virtual environment
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # Linux/Mac
# fraud_detection_env\Scripts\activate  # Windows

# Install required packages
pip install numpy pandas scikit-learn
pip install xgboost lightgbm catboost
pip install optuna
pip install shap lime
pip install imbalanced-learn  # For SMOTE
pip install matplotlib seaborn
pip install kaggle  # For dataset download
```

#### 1.2 Hardware Requirements
- RAM: 16GB+ recommended (dataset is 590K+ records)
- GPU: Optional but beneficial for XGBoost/LightGBM
- Storage: ~2GB for dataset

### Phase 2: Data Acquisition

#### 2.1 Download IEEE-CIS Fraud Detection Dataset
```bash
# Configure Kaggle API credentials first
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

#### 2.2 Expected Data Files
- `train_transaction.csv` - Transaction features
- `train_identity.csv` - Identity features
- `test_transaction.csv` - Test transaction features
- `test_identity.csv` - Test identity features

### Phase 3: Data Preprocessing Pipeline

#### 3.1 Data Loading and Merging
```python
import pandas as pd

# Load transaction and identity data
train_transaction = pd.read_csv('data/train_transaction.csv')
train_identity = pd.read_csv('data/train_identity.csv')

# Merge on TransactionID
train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')
```

#### 3.2 Missing Value Imputation
| Data Type | Imputation Strategy |
|-----------|---------------------|
| Numerical | Median imputation |
| Categorical | Mode imputation |

```python
from sklearn.impute import SimpleImputer

# Separate numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = train_df.select_dtypes(include=['object']).columns

# Impute numerical with median
num_imputer = SimpleImputer(strategy='median')
train_df[numerical_cols] = num_imputer.fit_transform(train_df[numerical_cols])

# Impute categorical with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
```

#### 3.3 Categorical Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col].astype(str))
```

#### 3.4 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
y = train_df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

#### 3.5 Class Imbalance Handling (SMOTE)
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Important:** Only apply SMOTE to training data, not test data.

### Phase 4: Feature Selection with SHAP

#### 4.1 Initial Model for Feature Importance
```python
import xgboost as xgb
import shap

# Train initial XGBoost for feature selection
initial_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
initial_model.fit(X_train_balanced, y_train_balanced)

# Calculate SHAP values
explainer = shap.TreeExplainer(initial_model)
shap_values = explainer.shap_values(X_train_balanced)
```

#### 4.2 Select Top 30 Features
```python
import numpy as np

# Get mean absolute SHAP values for each feature
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = X_train.columns

# Select top 30 features
top_30_idx = np.argsort(feature_importance)[-30:]
top_30_features = feature_names[top_30_idx].tolist()

# Filter datasets to top 30 features
X_train_selected = X_train_balanced[top_30_features]
X_test_selected = X_test[top_30_features]
```

### Phase 5: Hyperparameter Optimization with Optuna

#### 5.1 XGBoost Hyperparameter Tuning
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train_selected, y_train_balanced,
                            cv=5, scoring='roc_auc')
    return scores.mean()

# Run 20 trials as per paper
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_params = study.best_params
```

### Phase 6: Stacking Ensemble Model

#### 6.1 Train Base Learners
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Initialize base learners
xgb_model = XGBClassifier(**best_params)
lgbm_model = LGBMClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
catboost_model = CatBoostClassifier(
    iterations=300,
    depth=7,
    learning_rate=0.1,
    random_seed=42,
    verbose=0
)

# Train base models
xgb_model.fit(X_train_selected, y_train_balanced)
lgbm_model.fit(X_train_selected, y_train_balanced)
catboost_model.fit(X_train_selected, y_train_balanced)
```

#### 6.2 Generate Meta-Features
```python
# Get base model predictions for meta-learner training
xgb_pred = xgb_model.predict_proba(X_train_selected)[:, 1]
lgbm_pred = lgbm_model.predict_proba(X_train_selected)[:, 1]
catboost_pred = catboost_model.predict_proba(X_train_selected)[:, 1]

# Stack predictions
meta_train = np.column_stack([xgb_pred, lgbm_pred, catboost_pred])
```

#### 6.3 Train Meta-Learner (XGBoost)
```python
meta_learner = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
meta_learner.fit(meta_train, y_train_balanced)
```

### Phase 7: Evaluation

#### 7.1 Generate Test Predictions
```python
# Base model predictions on test set
xgb_test_pred = xgb_model.predict_proba(X_test_selected)[:, 1]
lgbm_test_pred = lgbm_model.predict_proba(X_test_selected)[:, 1]
catboost_test_pred = catboost_model.predict_proba(X_test_selected)[:, 1]

meta_test = np.column_stack([xgb_test_pred, lgbm_test_pred, catboost_test_pred])

# Final predictions
y_pred_proba = meta_learner.predict_proba(meta_test)[:, 1]
```

#### 7.2 Threshold Optimization
```python
from sklearn.metrics import f1_score

# Find optimal threshold (paper suggests 0.44)
thresholds = np.arange(0.3, 0.6, 0.01)
best_f1 = 0
best_threshold = 0.5

for thresh in thresholds:
    y_pred = (y_pred_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

y_pred_final = (y_pred_proba >= best_threshold).astype(int)
```

#### 7.3 Compute Metrics
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)

print("="*50)
print("CLASSIFICATION RESULTS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_final):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Optimal Threshold: {best_threshold:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))
```

#### 7.4 Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Full pipeline CV (simplified - base model only for CV)
cv_scores = cross_val_score(xgb_model, X_train_selected, y_train_balanced,
                           cv=cv, scoring='roc_auc')
print(f"\nCross-Validation AUC Scores: {cv_scores}")
print(f"Mean AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
```

### Phase 8: Explainability Analysis

#### 8.1 SHAP Analysis
```python
import shap

# SHAP for meta-learner interpretability
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_selected)

# Summary plot
shap.summary_plot(shap_values, X_test_selected, show=False)
plt.savefig('results/shap_summary.png', dpi=300, bbox_inches='tight')
```

#### 8.2 LIME Analysis
```python
import lime
import lime.lime_tabular

# Initialize LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_selected.values,
    feature_names=top_30_features,
    class_names=['Legitimate', 'Fraud'],
    mode='classification'
)

# Explain a single prediction
idx = 0  # Select a fraud case
exp = lime_explainer.explain_instance(
    X_test_selected.iloc[idx].values,
    meta_learner.predict_proba,  # Use stacking model
    num_features=10
)
exp.save_to_file('results/lime_explanation.html')
```

#### 8.3 Partial Dependence Plots
```python
from sklearn.inspection import PartialDependenceDisplay

# Plot partial dependence for top features
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    xgb_model, X_test_selected, features=[0, 1, 2, 3],  # Top 4 features
    ax=ax
)
plt.savefig('results/partial_dependence.png', dpi=300, bbox_inches='tight')
```

---

## Expected Results (Target Metrics)

| Metric | Paper Result | Acceptable Range |
|--------|-------------|------------------|
| Accuracy | 0.99 | 0.97 - 0.99 |
| Precision | 0.99 | 0.97 - 0.99 |
| Recall | 0.99 | 0.97 - 0.99 |
| F1-Score | 0.99 | 0.97 - 0.99 |
| AUC-ROC | 0.99 | 0.97 - 0.99 |

### Confusion Matrix Target
| | Predicted Legitimate | Predicted Fraud |
|---|---------------------|-----------------|
| Actual Legitimate | ~113,453 | ~481 |
| Actual Fraud | ~1,825 | ~112,192 |

---

## Potential Reproduction Challenges

### 1. Exact Feature Selection
- Paper identifies **C14** as most important feature
- Top 30 features may vary slightly based on random seed
- **Mitigation:** Document exact feature list obtained

### 2. SMOTE Implementation
- Class ratio after SMOTE not explicitly specified
- Different SMOTE variants may yield different results
- **Mitigation:** Test with `sampling_strategy='auto'` and `1.0`

### 3. Optuna Randomness
- 20 trials may not find global optimum
- **Mitigation:** Use fixed seed, consider more trials (50-100)

### 4. Stacking Architecture
- Exact meta-learner hyperparameters not fully specified
- **Mitigation:** Use simple XGBoost configuration

### 5. Memory Requirements
- Full dataset may require 16GB+ RAM
- **Mitigation:** Use chunked processing or sample data for testing

---

## Project Structure

```
fraud-detection-reproduction/
├── README.md
├── requirements.txt
├── config/
│   └── params.yaml           # Hyperparameter configurations
├── data/
│   └── .gitkeep              # Placeholder (data from Kaggle)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Phase 3 implementation
│   ├── feature_selection.py   # Phase 4 implementation
│   ├── hyperparameter_tuning.py # Phase 5 implementation
│   ├── stacking_model.py      # Phase 6 implementation
│   ├── evaluation.py          # Phase 7 implementation
│   └── explainability.py      # Phase 8 implementation
├── notebooks/
│   └── full_pipeline.ipynb    # End-to-end notebook
├── results/
│   ├── metrics.json           # Final metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── lime_explanation.html
└── tests/
    └── test_pipeline.py       # Unit tests
```

---

## Verification Checklist

- [ ] Dataset downloaded successfully (590K+ records)
- [ ] Missing value imputation completed
- [ ] Categorical encoding applied
- [ ] SMOTE applied to training data only
- [ ] Top 30 features selected via SHAP
- [ ] Optuna tuning completed (20 trials)
- [ ] XGBoost, LightGBM, CatBoost base models trained
- [ ] Meta-learner (XGBoost) trained on stacked predictions
- [ ] Threshold optimization performed (target: ~0.44)
- [ ] Accuracy >= 0.97 achieved
- [ ] AUC-ROC >= 0.97 achieved
- [ ] SHAP analysis completed
- [ ] LIME explanations generated
- [ ] Results documented and compared to paper

---

## References

1. **Original Paper:** Almalki, F., & Masud, M. (2025). Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods. arXiv:2505.10050
2. **Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
3. **XGBoost:** [Documentation](https://xgboost.readthedocs.io/)
4. **LightGBM:** [Documentation](https://lightgbm.readthedocs.io/)
5. **CatBoost:** [Documentation](https://catboost.ai/docs/)
6. **SHAP:** [Documentation](https://shap.readthedocs.io/)
7. **Optuna:** [Documentation](https://optuna.org/)

---

## License

This reproduction study is for educational and research purposes only.

## Contributing

Feel free to open issues or submit PRs if you find improvements or encounter challenges during reproduction.
