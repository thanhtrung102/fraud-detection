"""
Fraud Detection Package
=======================

A high-performance credit card fraud detection system using
stacking ensemble methods and explainable AI.

Paper: "Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods"
Target: 99% Accuracy, 0.99 AUC-ROC

Modules:
    - data_preprocessing: Data loading, cleaning, and SMOTE balancing
    - feature_selection: SHAP-based feature selection (top 30 features)
    - optuna_tuning: Hyperparameter optimization with Optuna (20 trials)
    - stacking_model: XGBoost + LightGBM + CatBoost stacking ensemble
    - evaluation: Metrics computation and visualization
    - explainability: SHAP, LIME, and PDP analysis
"""

__version__ = "1.1.0"
__author__ = "thanhtrung102"
