"""
Feature Selection Module
========================

SHAP-based feature selection for fraud detection.
Selects top N features based on SHAP importance values.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap


def shap_feature_selection(X_train: np.ndarray, y_train: np.ndarray,
                           feature_names: list, n_top_features: int = 30,
                           sample_size: int = 50000) -> tuple:
    """
    Select top features using SHAP importance values.

    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        n_top_features: Number of top features to select
        sample_size: Sample size for SHAP calculation (for memory efficiency)

    Returns:
        (selected_feature_indices, selected_feature_names, shap_importance)
    """
    print(f"Running SHAP feature selection (selecting top {n_top_features} features)...")

    # Sample data if needed for memory efficiency
    if len(X_train) > sample_size:
        print(f"  Sampling {sample_size:,} rows for SHAP calculation...")
        indices = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices] if isinstance(X_train, np.ndarray) else X_train.iloc[indices]
        y_sample = y_train[indices] if isinstance(y_train, np.ndarray) else y_train.iloc[indices]
    else:
        X_sample = X_train
        y_sample = y_train

    # Train a quick XGBoost model for SHAP
    print("  Training XGBoost for SHAP values...")
    quick_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    quick_model.fit(X_sample, y_sample)

    # Calculate SHAP values
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(quick_model)

    # Use a smaller sample for SHAP calculation if still too large
    shap_sample_size = min(10000, len(X_sample))
    X_shap = X_sample[:shap_sample_size] if isinstance(X_sample, np.ndarray) else X_sample.iloc[:shap_sample_size]

    shap_values = explainer.shap_values(X_shap)

    # Calculate mean absolute SHAP values per feature
    if isinstance(shap_values, list):
        # For multi-class, use the positive class
        shap_importance = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)

    # Select top N features
    top_features = importance_df.head(n_top_features)
    selected_feature_names = top_features['feature'].tolist()
    selected_feature_indices = [feature_names.index(f) for f in selected_feature_names]

    print(f"  Selected {n_top_features} features with highest SHAP importance")
    print(f"  Top 5 features: {selected_feature_names[:5]}")

    return selected_feature_indices, selected_feature_names, importance_df


def apply_feature_selection(X: np.ndarray, feature_indices: list) -> np.ndarray:
    """
    Apply feature selection to dataset.

    Args:
        X: Feature matrix
        feature_indices: Indices of selected features

    Returns:
        Filtered feature matrix
    """
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, feature_indices].values
    return X[:, feature_indices]


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=50, random_state=42)
    feature_names = [f"feature_{i}" for i in range(50)]

    indices, names, importance = shap_feature_selection(
        X, y, feature_names, n_top_features=10
    )

    print(f"\nSelected features: {names}")
